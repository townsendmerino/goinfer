package stores

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"slices"
	"testing"

	"github.com/townsendmerino/goinfer/internal/app/serveapp/jobs"
)

func TestBatchStore_createAndGet(t *testing.T) {
	bs := NewBatchStore(0)
	b := bs.Create("openai_chat", "/v1/chat/completions", "24h", "file-1", []string{"a", "b"})
	if b.ID == "" {
		t.Fatal("Create returned an empty id")
	}
	if b.Status() != "in_progress" {
		t.Fatalf("initial status = %q, want in_progress", b.Status())
	}
	if len(b.results) != 2 || len(b.jobIDs) != 2 {
		t.Fatalf("results/jobIDs not sized to len(customIDs): %+v", b)
	}
	if got := bs.Get(b.ID); got != b {
		t.Fatal("Get returned a different record than Create")
	}
}

func TestBatchStore_getUnknownReturnsNil(t *testing.T) {
	bs := NewBatchStore(0)
	if bs.Get("does-not-exist") != nil {
		t.Fatal("Get for an unknown id returned non-nil")
	}
}

func TestBatchStore_evictionNeverDropsALiveBatch(t *testing.T) {
	bs := NewBatchStore(2)
	old := bs.Create("openai_chat", "/v1/chat/completions", "", "f", []string{"x"})

	a := bs.Create("openai_chat", "/v1/chat/completions", "", "f", []string{"x"})
	a.mu.Lock()
	a.status = "completed"
	a.mu.Unlock()
	b := bs.Create("openai_chat", "/v1/chat/completions", "", "f", []string{"x"})
	b.mu.Lock()
	b.status = "completed"
	b.mu.Unlock()
	bs.mu.Lock()
	bs.evictLocked()
	bs.mu.Unlock()

	if bs.Get(old.ID) == nil {
		t.Fatal("the still in_progress oldest batch was evicted — a live batch must never be dropped")
	}
	remaining := 0
	for _, id := range []string{old.ID, a.ID, b.ID} {
		if bs.Get(id) != nil {
			remaining++
		}
	}
	if remaining != 2 {
		t.Fatalf("batches remaining = %d, want 2", remaining)
	}
}

func TestBatchStore_unboundedWhenCapIsZero(t *testing.T) {
	bs := NewBatchStore(0)
	var ids []string
	for range 10 {
		b := bs.Create("openai_chat", "/v1/chat/completions", "", "f", []string{"x"})
		b.mu.Lock()
		b.status = "completed"
		b.mu.Unlock()
		ids = append(ids, b.ID)
	}
	for _, id := range ids {
		if bs.Get(id) == nil {
			t.Fatalf("batch %s evicted despite cap=0 (unbounded)", id)
		}
	}
}

func TestBatchRecord_requestCancel_marksCancellingAndCancelsJobs(t *testing.T) {
	js, err := jobs.NewStore("", 0)
	if err != nil {
		t.Fatalf("jobs.NewStore: %v", err)
	}
	js.CreatePending("j1", "m", nil)
	js.CreatePending("j2", "m", nil)
	var cancelled1, cancelled2 bool
	js.SetCancel("j1", func() { cancelled1 = true })
	js.SetCancel("j2", func() { cancelled2 = true })

	bs := NewBatchStore(0)
	b := bs.Create("openai_chat", "/v1/chat/completions", "", "f", []string{"a", "b"})
	b.RegisterJob(0, "j1", js, func() { cancelled1 = true })
	b.RegisterJob(1, "j2", js, func() { cancelled2 = true })

	b.RequestCancel(js)

	if b.Status() != "cancelling" {
		t.Fatalf("status = %q, want cancelling", b.Status())
	}
	if !cancelled1 || !cancelled2 {
		t.Fatalf("cancelled1=%v cancelled2=%v, want both true", cancelled1, cancelled2)
	}
}

func TestBatchRecord_requestCancel_noOpOnceTerminal(t *testing.T) {
	js, err := jobs.NewStore("", 0)
	if err != nil {
		t.Fatalf("jobs.NewStore: %v", err)
	}
	bs := NewBatchStore(0)
	b := bs.Create("openai_chat", "/v1/chat/completions", "", "f", []string{"a"})
	b.mu.Lock()
	b.status = "completed"
	b.mu.Unlock()

	b.RequestCancel(js)

	if b.Status() != "completed" {
		t.Fatalf("status = %q, want completed (a terminal batch must not be reopened)", b.Status())
	}
}

func TestBatchStore_activeLineBudgetIsReservedAndReleased(t *testing.T) {
	bs := NewBatchStoreWithMaxActive(0, 3)
	b1 := bs.Create("openai_chat", "/v1/chat/completions", "", "f", []string{"a", "b"})
	if b1 == nil {
		t.Fatal("a batch inside the budget was refused")
	}
	if b := bs.Create("openai_chat", "/v1/chat/completions", "", "f", []string{"c", "d"}); b != nil {
		t.Fatal("a batch past the server-wide line budget was created")
	}
	if got := bs.ActiveLineCount(); got != 2 {
		t.Fatalf("active lines = %d, want 2", got)
	}
	b1.LineFinished()
	b1.DoneLine()
	b1.LineFinished()
	b1.DoneLine()
	b1.LineFinished() // idempotent past N
	b1.Wait()
	if got := bs.ActiveLineCount(); got != 0 {
		t.Fatalf("active lines after release = %d, want 0", got)
	}
	ctx, cancel := context.WithCancel(context.Background())
	js, err := jobs.NewStore("", 4)
	if err != nil {
		t.Fatal(err)
	}
	b2 := bs.Create("openai_chat", "/v1/chat/completions", "", "f", []string{"x"})
	b2.RequestCancel(js)
	js.CreatePending("jx", "m", nil)
	if b2.RegisterJob(0, "jx", js, cancel) || ctx.Err() == nil {
		t.Fatal("RegisterJob on cancelling batch should return false and invoke cancel")
	}
}

func TestAssembleOpenAIOutput_splitsSuccessAndErrorPreservingOrder(t *testing.T) {
	b := &BatchRecord{
		CustomIDs: []string{"req-1", "req-2", "req-3"},
		results: []BatchLineResult{
			{CustomID: "req-1", StatusCode: 200, Body: map[string]any{"object": "chat.completion"}},
			{CustomID: "req-2", ErrType: "invalid_request_error", ErrMsg: "images are not supported"},
			{CustomID: "req-3", StatusCode: 200, Body: map[string]any{"object": "chat.completion"}},
		},
	}
	out, errs := b.assembleOpenAIOutputLocked()

	var outIDs []string
	sc := bufio.NewScanner(bytes.NewReader(out))
	for sc.Scan() {
		var line struct {
			CustomID string           `json:"custom_id"`
			Response *json.RawMessage `json:"response"`
			Error    *json.RawMessage `json:"error"`
		}
		if err := json.Unmarshal(sc.Bytes(), &line); err != nil {
			t.Fatalf("output line did not parse: %v", err)
		}
		if line.Response == nil {
			t.Errorf("output line %s has a nil response", line.CustomID)
		}
		if line.Error != nil && string(*line.Error) != "null" {
			t.Errorf("output line %s carries a non-null error: %s", line.CustomID, *line.Error)
		}
		outIDs = append(outIDs, line.CustomID)
	}
	if want := []string{"req-1", "req-3"}; !slices.Equal(outIDs, want) {
		t.Errorf("output custom_ids = %v, want %v", outIDs, want)
	}

	var errIDs []string
	sc = bufio.NewScanner(bytes.NewReader(errs))
	for sc.Scan() {
		var line struct {
			CustomID string           `json:"custom_id"`
			Response *json.RawMessage `json:"response"`
			Error    *json.RawMessage `json:"error"`
		}
		if err := json.Unmarshal(sc.Bytes(), &line); err != nil {
			t.Fatalf("error line did not parse: %v", err)
		}
		if line.Response != nil && string(*line.Response) != "null" {
			t.Errorf("error line %s carries a non-null response: %s", line.CustomID, *line.Response)
		}
		if line.Error == nil {
			t.Errorf("error line %s has a nil error", line.CustomID)
		}
		errIDs = append(errIDs, line.CustomID)
	}
	if want := []string{"req-2"}; !slices.Equal(errIDs, want) {
		t.Errorf("error custom_ids = %v, want %v", errIDs, want)
	}
}
