package serveapp

import "testing"

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
// DELETE /v1/jobs/{id} (jobStore.cancel), applied to every line at once.
func TestBatchRecord_requestCancel_marksCancellingAndCancelsJobs(t *testing.T) {
	jobs, err := newJobStore("", 0)
	if err != nil {
		t.Fatalf("newJobStore: %v", err)
	}
	j1 := jobs.createPending("j1", "m", nil)
	j2 := jobs.createPending("j2", "m", nil)
	_ = j1
	_ = j2
	var cancelled1, cancelled2 bool
	jobs.setCancel("j1", func() { cancelled1 = true })
	jobs.setCancel("j2", func() { cancelled2 = true })

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
// reopened into "cancelling" by a late cancel request — mirrors jobStore.cancel's own
// idempotence (TestJobStore_cancelUnknownIDIsNotAnError, jobs_test.go).
func TestBatchRecord_requestCancel_noOpOnceTerminal(t *testing.T) {
	jobs, err := newJobStore("", 0)
	if err != nil {
		t.Fatalf("newJobStore: %v", err)
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
