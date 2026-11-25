package serveapp

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/internal/app/serveapp/jobs"
)

func TestBatchStore_createAndGet(t *testing.T) {
	bs := newBatchStore(0)
	b := bs.create("openai_chat", "/v1/chat/completions", "24h", "file-1", []string{"a", "b"})
	if b.ID == "" {
		t.Fatal("create returned an empty id")
	}
	if b.Status != "in_progress" {
		t.Fatalf("initial status = %q, want in_progress", b.Status)
	}
	if len(b.Results) != 2 || len(b.JobIDs) != 2 {
		t.Fatalf("Results/JobIDs not sized to len(customIDs): %+v", b)
	}
	if got := bs.get(b.ID); got != b {
		t.Fatal("get returned a different record than create")
	}
}

func TestBatchStore_getUnknownReturnsNil(t *testing.T) {
	bs := newBatchStore(0)
	if bs.get("does-not-exist") != nil {
		t.Fatal("get for an unknown id returned non-nil")
	}
}

// TestBatchStore_evictionNeverDropsALiveBatch mirrors TestJobStore_evictionNeverDropsALiveJob
// (jobs_test.go) — the identical eviction shape, copied on purpose (batches.go's own comment).
func TestBatchStore_evictionNeverDropsALiveBatch(t *testing.T) {
	bs := newBatchStore(2)
	old := bs.create("openai_chat", "/v1/chat/completions", "", "f", []string{"x"}) // stays in_progress

	a := bs.create("openai_chat", "/v1/chat/completions", "", "f", []string{"x"})
	a.mu.Lock()
	a.Status = "completed"
	a.mu.Unlock()
	b := bs.create("openai_chat", "/v1/chat/completions", "", "f", []string{"x"})
	b.mu.Lock()
	b.Status = "completed"
	b.mu.Unlock()
	bs.mu.Lock()
	bs.evictLocked()
	bs.mu.Unlock()

	if bs.get(old.ID) == nil {
		t.Fatal("the still in_progress oldest batch was evicted — a live batch must never be dropped")
	}
	remaining := 0
	for _, id := range []string{old.ID, a.ID, b.ID} {
		if bs.get(id) != nil {
			remaining++
		}
	}
	if remaining != 2 {
		t.Fatalf("batches remaining = %d, want 2 (old survives; exactly one completed batch evicted)", remaining)
	}
}

func TestBatchStore_unboundedWhenCapIsZero(t *testing.T) {
	bs := newBatchStore(0)
	var ids []string
	for range 10 {
		b := bs.create("openai_chat", "/v1/chat/completions", "", "f", []string{"x"})
		b.mu.Lock()
		b.Status = "completed"
		b.mu.Unlock()
		ids = append(ids, b.ID)
	}
	for _, id := range ids {
		if bs.get(id) == nil {
			t.Fatalf("batch %s evicted despite cap=0 (unbounded)", id)
		}
	}
}

// TestBatchRecord_requestCancel_marksCancellingAndCancelsJobs: cancelling a batch must mark it
// cancelling AND reach every constituent job's cancel func — the batch-level analogue of
// DELETE /v1/jobs/{id} (jobs.Store.Cancel), applied to every line at once.
func TestBatchRecord_requestCancel_marksCancellingAndCancelsJobs(t *testing.T) {
	jobs, err := jobs.NewStore("", 0)
	if err != nil {
		t.Fatalf("jobs.NewStore: %v", err)
	}
	j1 := jobs.CreatePending("j1", "m", nil)
	j2 := jobs.CreatePending("j2", "m", nil)
	_ = j1
	_ = j2
	var cancelled1, cancelled2 bool
	jobs.SetCancel("j1", func() { cancelled1 = true })
	jobs.SetCancel("j2", func() { cancelled2 = true })

	bs := newBatchStore(0)
	b := bs.create("openai_chat", "/v1/chat/completions", "", "f", []string{"a", "b"})
	b.JobIDs = []string{"j1", "j2"}

	b.requestCancel(jobs)

	b.mu.Lock()
	status := b.Status
	b.mu.Unlock()
	if status != "cancelling" {
		t.Fatalf("status = %q, want cancelling", status)
	}
	if !cancelled1 || !cancelled2 {
		t.Fatalf("cancelled1=%v cancelled2=%v, want both true", cancelled1, cancelled2)
	}
}

// TestBatchRecord_requestCancel_noOpOnceTerminal: a batch that already finished must not be
// reopened into "cancelling" by a late cancel request — mirrors jobs.Store.Cancel's own
// idempotence (TestJobStore_cancelUnknownIDIsNotAnError, jobs_test.go).
func TestBatchRecord_requestCancel_noOpOnceTerminal(t *testing.T) {
	jobs, err := jobs.NewStore("", 0)
	if err != nil {
		t.Fatalf("jobs.NewStore: %v", err)
	}
	bs := newBatchStore(0)
	b := bs.create("openai_chat", "/v1/chat/completions", "", "f", []string{"a"})
	b.mu.Lock()
	b.Status = "completed"
	b.mu.Unlock()

	b.requestCancel(jobs)

	b.mu.Lock()
	status := b.Status
	b.mu.Unlock()
	if status != "completed" {
		t.Fatalf("status = %q, want completed (a terminal batch must not be reopened)", status)
	}
}

// A batch line with an image or audio part is refused by its part type, before any payload is decoded: a payload that is not
// valid base64 gets the unsupported-media refusal, naming what it refuses, never a base64 decode error.
func TestBatchLines_imagePartsRefusedBeforeDecode(t *testing.T) {
	bs := newBatchStore(0)
	s := &server{batches: bs}
	b := bs.create("openai_chat", "/v1/chat/completions", "", "f", []string{"oa-image", "oa-audio"})
	image := chatReq{Messages: []chatMessage{{Role: "user", Content: json.RawMessage(
		`[{"type":"text","text":"describe"},{"type":"image_url","image_url":{"url":"data:image/png;base64,!!!not-base64!!!"}}]`)}}}
	audio := chatReq{Messages: []chatMessage{{Role: "user", Content: json.RawMessage(
		`[{"type":"input_audio","input_audio":{"data":"!!!not-base64!!!","format":"wav"}}]`)}}}
	runBatchChatLine(s, b, 0, image)
	runBatchChatLine(s, b, 1, audio)
	maxTok := 8
	ab := bs.create("anthropic_messages", "/v1/messages", "", "", []string{"an-image"})
	areq := anthropicReq{MaxTokens: &maxTok, Messages: []anthropicMessage{{Role: "user", Content: json.RawMessage(
		`[{"type":"image","source":{"type":"base64","media_type":"image/png","data":"!!!not-base64!!!"}}]`)}}}
	runBatchMessageLine(s, ab, 0, areq)
	for _, c := range []struct {
		r    batchLineResult
		want string
	}{
		{b.Results[0], "do not support image or audio inputs"},
		{b.Results[1], "do not support image or audio inputs"},
		{ab.Results[0], "do not support image inputs"},
	} {
		if c.r.StatusCode != http.StatusBadRequest || !strings.Contains(c.r.ErrMsg, c.want) {
			t.Errorf("line %s: status %d, message %q; want 400 with %q", c.r.CustomID, c.r.StatusCode, c.r.ErrMsg, c.want)
		}
	}
}

// The server-wide line budget: create reserves every line of a batch at once and refuses a batch that would exceed the budget, and each
// line returns its reservation when it finishes, whether it completed, failed validation, or was cancelled.
func TestBatchStore_activeLineBudgetIsReservedAndReleased(t *testing.T) {
	bs := newBatchStore(0)
	bs.maxActiveLines = 3
	s := &server{batches: bs}
	b1 := bs.create("openai_chat", "/v1/chat/completions", "", "f", []string{"a", "b"})
	if b1 == nil {
		t.Fatal("a batch inside the budget was refused")
	}
	if b := bs.create("openai_chat", "/v1/chat/completions", "", "f", []string{"c", "d"}); b != nil {
		t.Fatal("a batch past the server-wide line budget was created")
	}
	if got := bs.activeLineCount(); got != 2 {
		t.Fatalf("active lines after one admitted and one refused batch = %d, want 2", got)
	}
	js, err := jobs.NewStore("", 0)
	if err != nil {
		t.Fatal(err)
	}
	b1.requestCancel(js)
	runBatchPool(2, func(i int) { runBatchChatLine(s, b1, i, chatReq{}) })
	b1.wg.Wait()
	if got := bs.activeLineCount(); got != 0 {
		t.Fatalf("active lines after every line of a cancelled batch finished = %d, want 0", got)
	}
	b2 := bs.create("anthropic_messages", "/v1/messages", "", "", []string{"x", "y", "z"})
	if b2 == nil {
		t.Fatal("the released budget was not available to the next batch")
	}
	runBatchPool(3, func(i int) { runBatchMessageLine(s, b2, i, anthropicReq{}) })
	b2.wg.Wait()
	if got := bs.activeLineCount(); got != 0 {
		t.Fatalf("active lines after every line failed validation = %d, want 0", got)
	}
	// A registered model with nothing loaded behind it: the line panics past validation, and recoverBatchLine fails it.
	ps := &server{batches: bs, models: map[string]*loadedModel{"m": {name: "m"}}}
	b3 := bs.create("openai_chat", "/v1/chat/completions", "", "f", []string{"p"})
	runBatchChatLine(ps, b3, 0, chatReq{Model: "m", Messages: []chatMessage{{Role: "user", Content: json.RawMessage(`"hi"`)}}})
	if r := b3.Results[0]; r.StatusCode != http.StatusInternalServerError {
		t.Fatalf("the panicking line's result = %+v, want a recovered 500", r)
	}
	if got := bs.activeLineCount(); got != 0 {
		t.Fatalf("active lines after a line panicked = %d, want 0", got)
	}
	b3.lineFinished() // a second release of the same line is a no-op, never a negative count
	if got := bs.activeLineCount(); got != 0 {
		t.Fatalf("active lines after a repeated release = %d, want 0", got)
	}
}

// A saturated line budget is a 429 in each dialect's error shape, and no batch is created.
func TestBatches_serverWideLineBudgetIs429(t *testing.T) {
	srv := &server{files: newFileStore(8, 0), batches: newBatchStore(8)}
	srv.batches.maxActiveLines = 1
	ts := httptest.NewServer(batchesTestMux(srv))
	defer ts.Close()

	row := func(id string) string {
		return mustJSONLine(t, map[string]any{"custom_id": id, "method": "POST", "url": "/v1/chat/completions", "body": map[string]any{}})
	}
	fileID := uploadJSONLFile(t, ts, "in.jsonl", []string{row("a"), row("b")})
	resp := postJSON(t, ts.URL+"/v1/batches", map[string]any{"input_file_id": fileID, "endpoint": "/v1/chat/completions", "completion_window": "24h"})
	defer resp.Body.Close()
	var oa struct {
		Error struct{ Type, Message string } `json:"error"`
	}
	_ = json.NewDecoder(resp.Body).Decode(&oa)
	if resp.StatusCode != http.StatusTooManyRequests || oa.Error.Type != "rate_limit_error" {
		t.Errorf("OpenAI dialect: status %d, error %+v; want 429 rate_limit_error", resp.StatusCode, oa.Error)
	}

	req := map[string]any{"params": map[string]any{}}
	resp2 := postJSON(t, ts.URL+"/v1/messages/batches", map[string]any{"requests": []map[string]any{
		{"custom_id": "x", "params": req["params"]}, {"custom_id": "y", "params": req["params"]}}})
	defer resp2.Body.Close()
	var an struct {
		Type  string                         `json:"type"`
		Error struct{ Type, Message string } `json:"error"`
	}
	_ = json.NewDecoder(resp2.Body).Decode(&an)
	if resp2.StatusCode != http.StatusTooManyRequests || an.Type != "error" || an.Error.Type != "rate_limit_error" {
		t.Errorf("Anthropic dialect: status %d, body %+v; want 429 rate_limit_error", resp2.StatusCode, an)
	}
	srv.batches.mu.Lock()
	n := len(srv.batches.batches)
	srv.batches.mu.Unlock()
	if n != 0 {
		t.Errorf("a refused batch must not be created (%d in the store)", n)
	}
}
