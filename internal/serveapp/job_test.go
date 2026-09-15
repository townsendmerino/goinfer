package serveapp

import (
	"os"
	"path/filepath"
	"testing"
)

// TestJobStore_inMemoryWithoutJobDir: -job-dir unset ⇒ every generation still gets a job (task
// doc J2: "Every generation gets one, whether or not anything durable is on"), but nothing
// touches disk.
func TestJobStore_inMemoryWithoutJobDir(t *testing.T) {
	s, err := newJobStore("", 0)
	if err != nil {
		t.Fatalf("newJobStore(\"\"): %v", err)
	}
	if s.journal != nil {
		t.Fatal("journal should be nil when -job-dir is unset")
	}
	j := s.create("id-1", "m", []int{1, 2, 3})
	if j.State != jobRunning {
		t.Fatalf("state after create = %q, want running", j.State)
	}
	if j.Started == nil {
		t.Fatal("Started not set after create")
	}
	s.finish(j, jobDone, &usage{PromptTokens: 3, CompletionTokens: 5, TotalTokens: 8}, "", nil)
	if j.State != jobDone || j.Finished == nil {
		t.Fatalf("state after finish = %+v, want done with Finished set", j)
	}
}

// TestJobJournal_roundTrip: transitions written to a fresh journal are recovered as the LAST
// state per id after a reload — the core "keep the last snapshot" reconstruction rule.
func TestJobJournal_roundTrip(t *testing.T) {
	dir := t.TempDir()
	s, err := newJobStore(dir, 0)
	if err != nil {
		t.Fatalf("newJobStore: %v", err)
	}
	if s.journal == nil {
		t.Fatal("journal should be non-nil when -job-dir is set")
	}
	j := s.create("id-1", "m", []int{1, 2, 3})
	s.finish(j, jobDone, &usage{PromptTokens: 3, CompletionTokens: 4, TotalTokens: 7}, "", nil)

	restored, err := loadJournal(filepath.Join(dir, "jobs.jsonl"))
	if err != nil {
		t.Fatalf("loadJournal: %v", err)
	}
	got, ok := restored["id-1"]
	if !ok {
		t.Fatal("id-1 not found after reload")
	}
	if got.State != jobDone {
		t.Fatalf("reloaded state = %q, want done", got.State)
	}
	if got.Usage == nil || got.Usage.TotalTokens != 7 {
		t.Fatalf("reloaded usage = %+v, want TotalTokens=7", got.Usage)
	}
	if len(got.PromptIDs) != 3 {
		t.Fatalf("reloaded prompt ids = %v, want length 3", got.PromptIDs)
	}
}

// TestJobJournal_crashedJobReconstructsAsInterrupted is J2's own gate: "A job that was running at
// exit is marked interrupted, never silently failed." Write only the pending→running transitions
// (simulating a crash mid-generation, no terminal line ever written), then reload.
func TestJobJournal_crashedJobReconstructsAsInterrupted(t *testing.T) {
	dir := t.TempDir()
	jj, restored, err := newJobJournal(dir)
	if err != nil {
		t.Fatalf("newJobJournal: %v", err)
	}
	if len(restored) != 0 {
		t.Fatalf("fresh journal should reconstruct nothing, got %d", len(restored))
	}
	j := job{ID: "id-crash", Model: "m", State: jobPending}
	if err := jj.record(j); err != nil {
		t.Fatalf("record pending: %v", err)
	}
	j.State = jobRunning
	if err := jj.record(j); err != nil {
		t.Fatalf("record running: %v", err)
	}
	// No terminal transition — this is the crash.

	restored2, err := loadJournal(filepath.Join(dir, "jobs.jsonl"))
	if err != nil {
		t.Fatalf("loadJournal after crash: %v", err)
	}
	got, ok := restored2["id-crash"]
	if !ok {
		t.Fatal("id-crash not found after reload")
	}
	if got.State != jobInterrupted {
		t.Fatalf("reconstructed state = %q, want interrupted (never failed, never missing)", got.State)
	}
	if got.Finished == nil {
		t.Fatal("interrupted job should have a Finished timestamp recorded")
	}
}

// TestJobJournal_reopenCompletesTheInterruptedRecord: newJobJournal (the constructor a real
// restart calls, not loadJournal directly) appends the interrupted transition back to the file
// itself, so the journal's own last line for that id stops silently saying "running".
func TestJobJournal_reopenCompletesTheInterruptedRecord(t *testing.T) {
	dir := t.TempDir()
	jj, _, err := newJobJournal(dir)
	if err != nil {
		t.Fatalf("newJobJournal (1st open): %v", err)
	}
	if err := jj.record(job{ID: "id-crash", Model: "m", State: jobPending}); err != nil {
		t.Fatal(err)
	}
	if err := jj.record(job{ID: "id-crash", Model: "m", State: jobRunning}); err != nil {
		t.Fatal(err)
	}

	// Reopen — simulating the process restarting after the crash.
	_, restored, err := newJobJournal(dir)
	if err != nil {
		t.Fatalf("newJobJournal (2nd open): %v", err)
	}
	if restored["id-crash"].State != jobInterrupted {
		t.Fatalf("state after reopen = %q, want interrupted", restored["id-crash"].State)
	}

	// The FILE's own last line for this id must now be the interrupted transition, not running —
	// a third open (no new crash) must reconstruct the SAME state, not flip anything again.
	_, restored3, err := newJobJournal(dir)
	if err != nil {
		t.Fatalf("newJobJournal (3rd open): %v", err)
	}
	if restored3["id-crash"].State != jobInterrupted {
		t.Fatalf("state after a clean 3rd open = %q, want interrupted (stable, not re-derived)", restored3["id-crash"].State)
	}
}

// TestJobJournal_permissions: dir at sessionDirPerm (0700), file at sessionFilePerm (0600) —
// sessions.go's own KV-snapshot rationale, reused verbatim (a job record replays the prompt).
func TestJobJournal_permissions(t *testing.T) {
	base := t.TempDir()
	dir := filepath.Join(base, "jobs")
	if _, _, err := newJobJournal(dir); err != nil {
		t.Fatalf("newJobJournal: %v", err)
	}
	di, err := os.Stat(dir)
	if err != nil {
		t.Fatalf("stat dir: %v", err)
	}
	if perm := di.Mode().Perm(); perm != sessionDirPerm {
		t.Errorf("job dir perm = %o, want %o", perm, sessionDirPerm)
	}
	fi, err := os.Stat(filepath.Join(dir, "jobs.jsonl"))
	if err != nil {
		t.Fatalf("stat file: %v", err)
	}
	if perm := fi.Mode().Perm(); perm != sessionFilePerm {
		t.Errorf("job file perm = %o, want %o", perm, sessionFilePerm)
	}
}

// TestJobJournal_missingFileIsNotAnError: the very first run ever has no jobs.jsonl yet.
func TestJobJournal_missingFileIsNotAnError(t *testing.T) {
	restored, err := loadJournal(filepath.Join(t.TempDir(), "does-not-exist.jsonl"))
	if err != nil {
		t.Fatalf("loadJournal on a missing file should not error, got %v", err)
	}
	if len(restored) != 0 {
		t.Fatalf("expected an empty map, got %d entries", len(restored))
	}
}
