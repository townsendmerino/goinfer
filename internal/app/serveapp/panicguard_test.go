package serveapp

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// One batch may hold maxBatchLines requests; the body cap alone admits millions of minimal lines, each a goroutine and a parsed request.
// No model is needed: the 413 is decided before any line runs.
func TestBatches_lineCountIsCapped(t *testing.T) {
	srv := &server{files: newFileStore(8), batches: newBatchStore(8)}
	ts := httptest.NewServer(batchesTestMux(srv))
	defer ts.Close()

	rows := make([]string, maxBatchLines+1)
	for i := range rows {
		rows[i] = fmt.Sprintf(`{"custom_id":"c%d","method":"POST","url":"/v1/chat/completions","body":{}}`, i)
	}
	fileID := uploadJSONLFile(t, ts, "big.jsonl", rows)
	resp := postJSON(t, ts.URL+"/v1/batches", map[string]any{"input_file_id": fileID, "endpoint": "/v1/chat/completions", "completion_window": "24h"})
	defer resp.Body.Close()
	var oa struct {
		Error struct{ Message string } `json:"error"`
	}
	_ = json.NewDecoder(resp.Body).Decode(&oa)
	if resp.StatusCode != http.StatusRequestEntityTooLarge || !strings.Contains(oa.Error.Message, "at most") {
		t.Errorf("OpenAI dialect: status %d, message %q; want 413 naming the cap", resp.StatusCode, oa.Error.Message)
	}

	reqs := make([]map[string]any, maxBatchLines+1)
	for i := range reqs {
		reqs[i] = map[string]any{"custom_id": fmt.Sprintf("c%d", i), "params": map[string]any{}}
	}
	resp2 := postJSON(t, ts.URL+"/v1/messages/batches", map[string]any{"requests": reqs})
	defer resp2.Body.Close()
	if resp2.StatusCode != http.StatusRequestEntityTooLarge {
		t.Errorf("Anthropic dialect: status %d, want 413", resp2.StatusCode)
	}
	srv.batches.mu.Lock()
	n := len(srv.batches.batches)
	srv.batches.mu.Unlock()
	if n != 0 {
		t.Errorf("a refused batch must not be created (%d in the store)", n)
	}
	if batchTooLarge(httptest.NewRecorder(), maxBatchLines) {
		t.Errorf("a batch of exactly %d lines must be accepted", maxBatchLines)
	}
}

// A panic in a spawned goroutine is not recovered by net/http: it ends the process. Each test below drives the REAL goroutine body
// (runBatchChatLine, runJob, finalizeBatch) with a server that is missing what the body dereferences, so the panic is the code's own,
// and fails if the body loses its recovery.
func TestPanicGuard_runBatchChatLineFailsAloneAndTheBatchStillFinalizes(t *testing.T) {
	// A registered model with nothing loaded behind it: the line resolves it, then dereferences what a real model has.
	s := &server{models: map[string]*loadedModel{"m": {name: "m"}}}
	bs := newBatchStore(4)
	b := bs.create("openai_chat", "/v1/chat/completions", "24h", "file_x", []string{"a", "b"})
	good := chatReq{} // an empty request fails validation before it reaches the registry: line b is an ordinary 400
	bad := chatReq{Model: "m", Messages: []chatMessage{{Role: "user", Content: json.RawMessage(`"hi"`)}}}
	go runBatchChatLine(s, b, 0, bad)
	go runBatchChatLine(s, b, 1, good)
	b.wg.Wait()
	b.mu.Lock()
	defer b.mu.Unlock()
	if r := b.Results[0]; r.CustomID != "a" || r.StatusCode != 500 || r.ErrMsg != "internal error" {
		t.Errorf("the panicking line's result = %+v; want a 500 'internal error' under its own custom_id, set before the batch could finalize", r)
	}
	if r := b.Results[1]; r.CustomID != "b" || r.StatusCode != 400 {
		t.Errorf("the other line's result was disturbed: %+v", r)
	}
}

func TestPanicGuard_runJobFailsTheJobAndClosesItsStream(t *testing.T) {
	js, err := newJobStore("", 8)
	if err != nil {
		t.Fatal(err)
	}
	s := &server{jobs: js}
	js.createPending("job_p", "m", nil)
	released := false
	// lm is nil: tryEnter dereferences it, which is the panic.
	runJob(s, nil, func() { released = true }, genRequest{id: "job_p"}, admissionRecord{}, 1)
	if !released {
		t.Error("the model was not released after the panic: an /admin unload would wait on it forever")
	}
	j, ok := js.snapshot("job_p")
	if !ok || j.State != jobFailed || j.Error != "internal error" {
		t.Fatalf("job after a panic: ok=%v %+v; want failed with 'internal error'", ok, j)
	}
	js.mu.Lock()
	log := js.logs["job_p"]
	js.mu.Unlock()
	if log == nil {
		t.Fatal("no event log")
	}
	log.mu.Lock()
	defer log.mu.Unlock()
	if !log.done || len(log.events) == 0 {
		t.Errorf("the event stream: done=%v events=%d; want closed with the failure events written", log.done, len(log.events))
	}
}

func TestPanicGuard_finalizeBatchMarksTheBatchFailed(t *testing.T) {
	s := &server{batches: newBatchStore(4)} // no file store: assembling the output file dereferences it
	b := s.batches.create("openai_chat", "/v1/chat/completions", "24h", "f", []string{"a"})
	setResult(b, 0, lineError("a", 400, "invalid_request_error", "x"))
	b.wg.Done()
	s.finalizeBatch(b.ID)
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.Status != "failed" || b.CompletedAt == nil {
		t.Errorf("batch after a finalizer panic: status %q, completed %v; want failed and stamped", b.Status, b.CompletedAt)
	}
}

// A request may carry at most maxStopSequences stop strings and maxToolsPerRequest tools: each is matched or branched on at every
// generated token, so their count multiplies the cost of the whole generation. prepare is the one place every dialect's stop reaches.
func TestRequestAmplifiers_stopAndToolCountsAreCapped(t *testing.T) {
	lm := &loadedModel{name: "m"}
	stops := func(n int) json.RawMessage {
		a := make([]string, n)
		for i := range a {
			a[i] = fmt.Sprintf("s%d", i)
		}
		b, _ := json.Marshal(a)
		return b
	}
	if _, err := lm.prepare(sampling{Stop: stops(maxStopSequences + 1)}, []int{1}, false); err == nil || !strings.Contains(err.Error(), "stop accepts at most") {
		t.Errorf("%d stop strings: err = %v, want a refusal naming the cap", maxStopSequences+1, err)
	}
	if _, err := lm.prepare(sampling{Stop: stops(maxStopSequences)}, []int{1}, false); err != nil && strings.Contains(err.Error(), "stop accepts") {
		t.Errorf("%d stop strings must be accepted, got %v", maxStopSequences, err)
	}
	if err := checkToolCount(maxToolsPerRequest + 1); err == nil {
		t.Errorf("%d tools accepted", maxToolsPerRequest+1)
	}
	if err := checkToolCount(maxToolsPerRequest); err != nil {
		t.Errorf("%d tools refused: %v", maxToolsPerRequest, err)
	}
}
