package serveapp

import (
	"sync"
	"time"
)

// batchLineResult is one line's outcome, dialect-neutral — each HTTP layer (batches_http.go)
// translates it into its own dialect's shape (OpenAI's output/error JSONL line, Anthropic's
// results JSONL line) rather than this struct trying to be both at once.
type batchLineResult struct {
	CustomID   string
	StatusCode int            // 200 on success
	Body       map[string]any // the full per-endpoint response body; nil on error
	ErrType    string         // "" on success
	ErrMsg     string
}

// batchRecord is J4's unit of work over J2's job store (task-work-queue-2026-09.md): N lines,
// submitted as N ordinary jobs (see batches_run.go), tracked here only for what a job alone
// cannot answer — aggregate status, original input order, and the assembled output.
//
// wg is Add(n) once at creation (batchStore.create) and Done() once per line as its own last act
// (batches_run.go) — not a job-store concept, this record's own completion signal, so
// finalizeBatch (batches_finalize.go) needs no polling: wg.Wait() IS "every line is terminal."
//
// mu guards Status/Results/OutputFileID/ErrorFileID/CompletedAt: many line goroutines write
// Results[i] concurrently (disjoint indices, so the writes themselves never race each other) but
// Status and the two file-id fields are read by every GET and written by exactly one finalizer,
// so they need the same discipline job.go's own mu gained in J3 after -race caught a real bug
// there (job.go:142's doc comment) — written locked from the start here instead of finding the
// same class of defect twice.
type batchRecord struct {
	ID               string
	Kind             string // "openai_chat" | "anthropic_messages"
	Endpoint         string // openai only, echoed back ("/v1/chat/completions")
	CompletionWindow string // openai only, echoed back verbatim — never enforced (task doc's own
	// carve-out: "it is not a promise this product can make on one worker")
	InputFileID string   // openai only
	CustomIDs   []string // input order
	JobIDs      []string // parallel to CustomIDs
	Created     time.Time
	wg          sync.WaitGroup

	mu           sync.Mutex
	Status       string // in_progress | finalizing | completed | cancelling | cancelled
	Results      []batchLineResult
	OutputFileID string // openai, set at finalize
	ErrorFileID  string // openai, set at finalize, only if any line errored
	CompletedAt  *time.Time
}

// requestCancel is the batch-level analogue of jobStore.cancel: mark cancelling, then cancel
// every constituent job not yet terminal (checked live against jobStore, not against Results,
// since Results is only written by a line's OWN goroutine as it finishes — a line still running
// has no Results entry yet either way).
//
// JobIDs is copied under b.mu before ranging over it — found by -race, not by inspection: each
// line's goroutine writes its own slot via setJobID (batches_run.go) under the same lock, and a
// plain read of the slice here raced those writes even though every write lands at a disjoint
// index (job.go:142's doc comment names the same class of bug in J3's job store).
func (b *batchRecord) requestCancel(jobs *jobStore) {
	b.mu.Lock()
	if b.Status == "completed" || b.Status == "cancelled" {
		b.mu.Unlock()
		return
	}
	b.Status = "cancelling"
	ids := append([]string(nil), b.JobIDs...)
	b.mu.Unlock()
	for _, id := range ids {
		if id != "" { // a line still validating (before setJobID) has no job to cancel yet
			jobs.cancel(id)
		}
	}
}

// batchStore is a bounded, in-memory, process-wide registry — mirrors jobStore's own shape
// (job.go:69) closely enough that the FIFO-skip-a-live-one eviction logic is copied, not
// reinvented: a batch a client is still polling must never disappear the same way a job must not
// (job.go:198's own comment).
type batchStore struct {
	mu      sync.Mutex
	batches map[string]*batchRecord
	order   []string
	cap     int
}

func newBatchStore(cap int) *batchStore {
	return &batchStore{batches: map[string]*batchRecord{}, cap: cap}
}

// create mints a batch record in "in_progress" (this pass validates synchronously before
// creating one, so there is no observable "validating" state to represent — see batches_http.go).
func (bs *batchStore) create(kind, endpoint, completionWindow, inputFileID string, customIDs []string) *batchRecord {
	b := &batchRecord{
		ID: "batch_" + reqID(), Kind: kind, Endpoint: endpoint, CompletionWindow: completionWindow,
		InputFileID: inputFileID, CustomIDs: customIDs, JobIDs: make([]string, len(customIDs)),
		Created: time.Now(), Status: "in_progress", Results: make([]batchLineResult, len(customIDs)),
	}
	b.wg.Add(len(customIDs))
	bs.mu.Lock()
	defer bs.mu.Unlock()
	bs.batches[b.ID] = b
	bs.order = append(bs.order, b.ID)
	bs.evictLocked()
	return b
}

// get returns id's batchRecord, or nil. Safe to read Status/Results etc. off the returned pointer
// only under b.mu — callers that just need the pointer (batches_run.go registering a job id,
// requestCancel) don't need the lock; callers projecting to HTTP (batches_http.go) take it.
func (bs *batchStore) get(id string) *batchRecord {
	bs.mu.Lock()
	defer bs.mu.Unlock()
	return bs.batches[id]
}

// evictLocked drops the oldest COMPLETED/CANCELLED batches once len(batches) exceeds cap. Called
// with bs.mu held. Mirrors jobStore.evictLocked (job.go:202) exactly, including the reason: "a
// batch in flight must never disappear out from under a client polling it."
func (bs *batchStore) evictLocked() {
	if bs.cap <= 0 {
		return
	}
	for i := 0; len(bs.order) > bs.cap && i < len(bs.order); {
		id := bs.order[i]
		b := bs.batches[id]
		live := true
		if b != nil {
			b.mu.Lock()
			live = b.Status != "completed" && b.Status != "cancelled"
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
