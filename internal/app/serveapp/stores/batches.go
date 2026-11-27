package stores

import (
	"bytes"
	"context"
	"fmt"
	"os"
	"runtime/debug"
	"slices"
	"sync"
	"sync/atomic"
	"time"

	"github.com/townsendmerino/goinfer/internal/app/serveapp/jobs"
	"github.com/townsendmerino/goinfer/internal/batchio"
)

// MaxBatchLines bounds the requests in one batch. Each line becomes a goroutine and a parsed request before any
// admission queue sees it, and the body cap alone admits millions of minimal lines.
const MaxBatchLines = 10000

// MaxServerActiveBatchLines bounds the batch lines admitted and not yet finished across every batch of both dialects.
const MaxServerActiveBatchLines = 2 * MaxBatchLines

// BatchLineResult is one line's outcome, dialect-neutral — each HTTP layer translates it into its dialect's shape.
type BatchLineResult struct {
	CustomID   string
	StatusCode int            // 200 on success
	Body       map[string]any // the full per-endpoint response body; nil on error
	ErrType    string         // "" on success
	ErrMsg     string
}

// LineError constructs a non-200 BatchLineResult for customID.
func LineError(customID string, status int, errType, msg string) BatchLineResult {
	return BatchLineResult{CustomID: customID, StatusCode: status, ErrType: errType, ErrMsg: msg}
}

// BatchRecord is the unit of work over jobs.Store: N lines submitted as N ordinary jobs, tracked here for aggregate
// status, original input order, and assembled output.
type BatchRecord struct {
	ID               string
	Kind             string // "openai_chat" | "anthropic_messages"
	Endpoint         string // openai only, echoed back ("/v1/chat/completions")
	CompletionWindow string // openai only, echoed back verbatim
	InputFileID      string // openai only
	CustomIDs        []string
	Created          time.Time

	wg       sync.WaitGroup
	store    *BatchStore
	reserved atomic.Int64

	mu           sync.Mutex
	status       string // in_progress | finalizing | completed | cancelling | cancelled | failed
	jobIDs       []string
	results      []BatchLineResult
	outputFileID string
	errorFileID  string
	completedAt  *time.Time
}

// IsCancelling reports whether cancellation has been requested for this batch.
func (b *BatchRecord) IsCancelling() bool {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.status == "cancelling" || b.status == "cancelled"
}

// Status returns the batch's current lifecycle status under its lock.
func (b *BatchRecord) Status() string {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.status
}

// OutputFileIDs returns the assembled OpenAI output and error file IDs under the batch lock.
func (b *BatchRecord) OutputFileIDs() (outputFileID, errorFileID string) {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.outputFileID, b.errorFileID
}

// CompletedAt returns a copy of the completion timestamp under the batch lock, or nil if still live.
func (b *BatchRecord) CompletedAt() *time.Time {
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.completedAt == nil {
		return nil
	}
	t := *b.completedAt
	return &t
}

// ResultAt returns line i's recorded outcome under the batch lock.
func (b *BatchRecord) ResultAt(i int) BatchLineResult {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.results[i]
}

// ResultCount returns the number of result slots sized at creation.
func (b *BatchRecord) ResultCount() int {
	b.mu.Lock()
	defer b.mu.Unlock()
	return len(b.results)
}

// SetResult records line i's outcome under the batch lock.
func (b *BatchRecord) SetResult(i int, r BatchLineResult) {
	b.mu.Lock()
	b.results[i] = r
	b.mu.Unlock()
}

// RegisterJob publishes jobID into slot i under the batch lock after registering cancel in js. If the batch is already
// cancelling, it cancels immediately and returns false, closing the cancel-vs-registration race.
func (b *BatchRecord) RegisterJob(i int, jobID string, js *jobs.Store, cancel context.CancelFunc) bool {
	js.SetCancel(jobID, cancel)
	b.mu.Lock()
	if b.status == "cancelling" || b.status == "cancelled" {
		b.mu.Unlock()
		cancel()
		return false
	}
	b.jobIDs[i] = jobID
	b.mu.Unlock()
	return true
}

// LineFinished returns one line's reservation to the server-wide budget. Safe to call more than once per line: only the
// first N calls across a batch release budget.
func (b *BatchRecord) LineFinished() {
	if b.store != nil && b.reserved.Add(-1) >= 0 {
		b.store.releaseLines(1)
	}
}

// DoneLine marks one constituent line finished on the batch's WaitGroup.
func (b *BatchRecord) DoneLine() { b.wg.Done() }

// Wait blocks until all constituent lines have called DoneLine.
func (b *BatchRecord) Wait() { b.wg.Wait() }

// RequestCancel marks the batch cancelling and cancels every constituent job in js that has registered an ID. Terminal
// batches ("completed", "cancelled", "failed") are left unchanged.
func (b *BatchRecord) RequestCancel(js *jobs.Store) {
	b.mu.Lock()
	if b.status == "completed" || b.status == "cancelled" || b.status == "failed" {
		b.mu.Unlock()
		return
	}
	b.status = "cancelling"
	ids := slices.Clone(b.jobIDs)
	b.mu.Unlock()
	for _, id := range ids {
		if id != "" {
			js.Cancel(id)
		}
	}
}

// MarkFailed sets the batch's terminal status to "failed" and stamps CompletedAt under the batch lock.
func (b *BatchRecord) MarkFailed() {
	b.mu.Lock()
	b.status = "failed"
	now := time.Now()
	b.completedAt = &now
	b.mu.Unlock()
}

// Finalize waits for every line to finish, assembles OpenAI output/error files via putFiles when Kind is "openai_chat",
// and transitions the batch to its terminal status. Any panic during finalization marks the batch "failed" under b.mu.
func (b *BatchRecord) Finalize(putFiles func(...NewFile) []*StoredFile) {
	b.wg.Wait()

	b.mu.Lock()
	defer b.mu.Unlock()
	defer b.recoverFinalizeLocked()
	cancelled := b.status == "cancelling"
	now := time.Now()
	b.completedAt = &now
	if b.Kind == "openai_chat" {
		out, errs := b.assembleOpenAIOutputLocked()
		files := []NewFile{{Filename: b.ID + "_output.jsonl", Purpose: "batch_output", Data: out}}
		if len(errs) > 0 {
			files = append(files, NewFile{Filename: b.ID + "_error.jsonl", Purpose: "batch_error", Data: errs})
		}
		stored := putFiles(files...)
		b.outputFileID = stored[0].ID
		if len(stored) > 1 {
			b.errorFileID = stored[1].ID
		}
	}
	if cancelled {
		b.status = "cancelled"
	} else {
		b.status = "completed"
	}
}

func (b *BatchRecord) recoverFinalizeLocked() {
	if r := recover(); r != nil {
		fmt.Fprintf(os.Stderr, "serve: finalizing batch %s panicked: %v\n%s", b.ID, r, debug.Stack())
		b.status = "failed"
		now := time.Now()
		b.completedAt = &now
	}
}

func (b *BatchRecord) assembleOpenAIOutputLocked() (output, errors []byte) {
	var out, errs bytes.Buffer
	for _, r := range b.results {
		id := "batch_req_" + MintID()
		if r.ErrType != "" || r.Body == nil {
			errs.Write(batchio.ErrLine(id, r.CustomID, r.ErrType, r.ErrMsg))
		} else {
			out.Write(batchio.OKLine(id, r.CustomID, r.StatusCode, r.Body))
		}
	}
	return out.Bytes(), errs.Bytes()
}

// TerminalResults returns a snapshot of results if the batch is in a terminal state, or (nil, false) while still live.
func (b *BatchRecord) TerminalResults() ([]BatchLineResult, bool) {
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.status != "completed" && b.status != "cancelled" && b.status != "failed" {
		return nil, false
	}
	return slices.Clone(b.results), true
}

// OpenAIObject projects b into OpenAI's /v1/batches response shape.
func (b *BatchRecord) OpenAIObject() map[string]any {
	b.mu.Lock()
	defer b.mu.Unlock()
	status := b.status
	completed, failed := 0, 0
	for _, r := range b.results {
		if r.CustomID == "" {
			continue
		}
		if r.ErrType == "" && r.Body != nil {
			completed++
		} else {
			failed++
		}
	}
	out := map[string]any{
		"id": b.ID, "object": "batch", "endpoint": b.Endpoint, "input_file_id": b.InputFileID,
		"completion_window": b.CompletionWindow, "status": status,
		"created_at": b.Created.Unix(),
		"request_counts": map[string]any{
			"total": len(b.CustomIDs), "completed": completed, "failed": failed,
		},
	}
	if b.outputFileID != "" {
		out["output_file_id"] = b.outputFileID
	}
	if b.errorFileID != "" {
		out["error_file_id"] = b.errorFileID
	}
	if b.completedAt != nil {
		out["completed_at"] = b.completedAt.Unix()
	}
	return out
}

// AnthropicObject projects b into Anthropic's /v1/messages/batches response shape.
func (b *BatchRecord) AnthropicObject() map[string]any {
	b.mu.Lock()
	defer b.mu.Unlock()
	status := "in_progress"
	switch b.status {
	case "cancelling":
		status = "canceling"
	case "completed", "cancelled", "failed":
		status = "ended"
	}
	processing, succeeded, errored, canceled := 0, 0, 0, 0
	for _, r := range b.results {
		if r.CustomID == "" {
			processing++
			continue
		}
		switch {
		case r.ErrType == "cancelled":
			canceled++
		case r.ErrType != "" || r.Body == nil:
			errored++
		default:
			succeeded++
		}
	}
	out := map[string]any{
		"id": b.ID, "type": "message_batch", "processing_status": status,
		"created_at": b.Created.Format(time.RFC3339),
		"request_counts": map[string]any{
			"processing": processing, "succeeded": succeeded, "errored": errored,
			"canceled": canceled, "expired": 0,
		},
		"results_url": nil,
	}
	if status == "ended" {
		out["results_url"] = "/v1/messages/batches/" + b.ID + "/results"
	}
	if b.completedAt != nil {
		out["ended_at"] = b.completedAt.Format(time.RFC3339)
	}
	return out
}

// BatchStore is a bounded, in-memory, process-wide registry mirroring jobs.Store's FIFO eviction that skips live batches.
type BatchStore struct {
	mu             sync.Mutex
	batches        map[string]*BatchRecord
	order          []string
	cap            int
	activeLines    int
	maxActiveLines int
}

// NewBatchStore constructs a BatchStore holding at most cap terminal batches (0 = unbounded) with the default
// MaxServerActiveBatchLines active-line cap.
func NewBatchStore(cap int) *BatchStore {
	return NewBatchStoreWithMaxActive(cap, MaxServerActiveBatchLines)
}

// NewBatchStoreWithMaxActive constructs a BatchStore with an explicit server-wide active-line cap.
func NewBatchStoreWithMaxActive(cap, maxActiveLines int) *BatchStore {
	return &BatchStore{batches: map[string]*BatchRecord{}, cap: cap, maxActiveLines: maxActiveLines}
}

// MaxActiveLines returns the server-wide active-line cap this store enforces.
func (bs *BatchStore) MaxActiveLines() int {
	bs.mu.Lock()
	defer bs.mu.Unlock()
	return bs.maxActiveLines
}

// ActiveLineCount reports the number of batch lines currently reserved across all live batches.
func (bs *BatchStore) ActiveLineCount() int {
	bs.mu.Lock()
	defer bs.mu.Unlock()
	return bs.activeLines
}

// Len reports how many batches are currently held in the store.
func (bs *BatchStore) Len() int {
	bs.mu.Lock()
	defer bs.mu.Unlock()
	return len(bs.batches)
}

func (bs *BatchStore) releaseLines(n int) {
	bs.mu.Lock()
	bs.activeLines -= n
	bs.mu.Unlock()
}

// Create mints a BatchRecord in "in_progress", reserving len(customIDs) lines against MaxActiveLines (returning nil if
// the reservation would exceed it).
func (bs *BatchStore) Create(kind, endpoint, completionWindow, inputFileID string, customIDs []string) *BatchRecord {
	b := &BatchRecord{
		ID: "batch_" + MintID(), Kind: kind, Endpoint: endpoint, CompletionWindow: completionWindow,
		InputFileID: inputFileID, CustomIDs: customIDs, jobIDs: make([]string, len(customIDs)),
		Created: time.Now(), status: "in_progress", results: make([]BatchLineResult, len(customIDs)),
		store: bs,
	}
	n := len(customIDs)
	b.reserved.Store(int64(n))
	bs.mu.Lock()
	defer bs.mu.Unlock()
	if bs.activeLines+n > bs.maxActiveLines {
		return nil
	}
	bs.activeLines += n
	b.wg.Add(n)
	bs.batches[b.ID] = b
	bs.order = append(bs.order, b.ID)
	bs.evictLocked()
	return b
}

// Get returns id's BatchRecord, or nil.
func (bs *BatchStore) Get(id string) *BatchRecord {
	bs.mu.Lock()
	defer bs.mu.Unlock()
	return bs.batches[id]
}

func (bs *BatchStore) evictLocked() {
	if bs.cap <= 0 {
		return
	}
	for i := 0; len(bs.order) > bs.cap && i < len(bs.order); {
		id := bs.order[i]
		b := bs.batches[id]
		live := true
		if b != nil {
			b.mu.Lock()
			live = b.status != "completed" && b.status != "cancelled" && b.status != "failed"
			b.mu.Unlock()
		}
		if b == nil || !live {
			delete(bs.batches, id)
			bs.order = append(bs.order[:i], bs.order[i+1:]...)
			continue
		}
		i++
	}
}
