package serveapp

import "testing"

// TestJobStore_evictionNeverDropsALiveJob: FIFO eviction (J3, mirroring responseStore's own cap
// pattern) must skip over any pending/running job rather than evicting it — a job in flight must
// never disappear out from under a client polling it, even if it happens to be the oldest entry.
func TestJobStore_evictionNeverDropsALiveJob(t *testing.T) {
	s, err := newJobStore("", 2)
	if err != nil {
		t.Fatalf("newJobStore: %v", err)
	}

	// Oldest job stays PENDING (never finished) — must survive every later eviction.
	old := s.createPending("old", "m", nil)
	if old.State != jobPending {
		t.Fatalf("old.State = %q, want pending", old.State)
	}

	// Two more jobs, both finished immediately, push past cap=2.
	a := s.createPending("a", "m", nil)
	s.finish(a, jobDone, nil, "", nil)
	b := s.createPending("b", "m", nil)
	s.finish(b, jobDone, nil, "", nil)

	if s.get("old") == nil {
		t.Fatal("the still-pending oldest job was evicted — a live job must never be dropped")
	}
	// With "old" un-evictable, cap=2 should have dropped exactly one of the two finished jobs
	// (the store now holds 3: old, and whichever of a/b is left) rather than none.
	remaining := 0
	for _, id := range []string{"old", "a", "b"} {
		if s.get(id) != nil {
			remaining++
		}
	}
	if remaining != 2 {
		t.Fatalf("jobs remaining = %d, want 2 (old survives; exactly one finished job evicted)", remaining)
	}
}

// TestJobStore_evictionDropsOldestFinishedFirst: among evictable (finished) jobs, the OLDEST goes
// first — the FIFO half of "FIFO eviction of the oldest FINISHED job".
func TestJobStore_evictionDropsOldestFinishedFirst(t *testing.T) {
	s, err := newJobStore("", 1)
	if err != nil {
		t.Fatalf("newJobStore: %v", err)
	}
	a := s.createPending("a", "m", nil)
	s.finish(a, jobDone, nil, "", nil)
	b := s.createPending("b", "m", nil)
	s.finish(b, jobDone, nil, "", nil)

	if s.get("a") != nil {
		t.Fatal("expected the older finished job (a) to be evicted first")
	}
	if s.get("b") == nil {
		t.Fatal("the newer finished job (b) should still be present")
	}
}

// TestJobStore_unboundedWhenCapIsZero: cap<=0 means unbounded (existing J2 tests rely on this).
func TestJobStore_unboundedWhenCapIsZero(t *testing.T) {
	s, err := newJobStore("", 0)
	if err != nil {
		t.Fatalf("newJobStore: %v", err)
	}
	for i := range 10 {
		j := s.createPending(string(rune('a'+i)), "m", nil)
		s.finish(j, jobDone, nil, "", nil)
	}
	for i := range 10 {
		if s.get(string(rune('a'+i))) == nil {
			t.Fatalf("job %d evicted despite cap=0 (unbounded)", i)
		}
	}
}

// TestJobStore_getOrCreatePreCreatedJobTransitionsToRunning: the async path's own contract —
// createPending followed by getOrCreate (drive's chokepoint) must transition the SAME job to
// running, not silently create a second one.
func TestJobStore_getOrCreatePreCreatedJobTransitionsToRunning(t *testing.T) {
	s, err := newJobStore("", 0)
	if err != nil {
		t.Fatalf("newJobStore: %v", err)
	}
	pre := s.createPending("id-1", "m", []int{1, 2, 3})
	if pre.State != jobPending {
		t.Fatalf("state after createPending = %q, want pending", pre.State)
	}
	got := s.getOrCreate("id-1", "m", []int{1, 2, 3})
	if got != pre {
		t.Fatal("getOrCreate on a pre-created id returned a different *job — it created a second one")
	}
	if got.State != jobRunning {
		t.Fatalf("state after getOrCreate = %q, want running", got.State)
	}
}

// TestJobStore_cancelUnknownIDIsNotAnError mirrors K1's own cancel-by-id idempotence.
func TestJobStore_cancelUnknownIDIsNotAnError(t *testing.T) {
	s, err := newJobStore("", 0)
	if err != nil {
		t.Fatalf("newJobStore: %v", err)
	}
	if s.cancel("does-not-exist") {
		t.Fatal("cancel on an unknown id reported found=true")
	}
}

// TestJobStore_eventLogAttachAndFetch is the plumbing attach()/eventLog() rely on.
func TestJobStore_eventLogAttachAndFetch(t *testing.T) {
	s, err := newJobStore("", 0)
	if err != nil {
		t.Fatalf("newJobStore: %v", err)
	}
	if s.eventLog("id-1") != nil {
		t.Fatal("eventLog for an unattached id should be nil")
	}
	log := newJobEventLog()
	s.attach("id-1", log)
	if s.eventLog("id-1") != log {
		t.Fatal("eventLog did not return the attached log")
	}
}
