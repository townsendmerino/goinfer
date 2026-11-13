package jobs

import (
	"encoding/json"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

// openJournal is NewJournal, closed when the test ends (Windows cannot remove a TempDir holding an open file).
func openJournal(t *testing.T, dir string) (*Journal, map[string]*Job, error) {
	t.Helper()
	jj, restored, err := NewJournal(dir)
	if err == nil {
		t.Cleanup(func() { jj.Close() })
	}
	return jj, restored, err
}

// TestJobStore_inMemoryWithoutJobDir: -job-dir unset ⇒ every generation still gets a job (task
// doc J2: "Every generation gets one, whether or not anything durable is on"), but nothing
// touches disk.
func TestJobStore_inMemoryWithoutJobDir(t *testing.T) {
	s, err := NewStore("", 0)
	if err != nil {
		t.Fatalf("NewStore(\"\"): %v", err)
	}
	if s.journal != nil {
		t.Fatal("journal should be nil when -job-dir is unset")
	}
	j := s.Create("id-1", "m", []int{1, 2, 3})
	if j.State != Running {
		t.Fatalf("state after create = %q, want running", j.State)
	}
	if j.Started == nil {
		t.Fatal("Started not set after create")
	}
	s.Finish(j, Done, &Usage{PromptTokens: 3, CompletionTokens: 5, TotalTokens: 8}, "", nil)
	if j.State != Done || j.Finished == nil {
		t.Fatalf("state after finish = %+v, want done with Finished set", j)
	}
}

// TestJobJournal_roundTrip: transitions written to a fresh journal are recovered as the LAST
// state per id after a reload — the core "keep the last snapshot" reconstruction rule.
func TestJobJournal_roundTrip(t *testing.T) {
	dir := t.TempDir()
	s, err := NewStore(dir, 0)
	if err != nil {
		t.Fatalf("NewStore: %v", err)
	}
	if s.journal == nil {
		t.Fatal("journal should be non-nil when -job-dir is set")
	}
	t.Cleanup(func() { s.journal.Close() })
	j := s.Create("id-1", "m", []int{1, 2, 3})
	s.Finish(j, Done, &Usage{PromptTokens: 3, CompletionTokens: 4, TotalTokens: 7}, "", nil)

	restored, err := LoadJournal(filepath.Join(dir, "jobs.jsonl"))
	if err != nil {
		t.Fatalf("LoadJournal: %v", err)
	}
	got, ok := restored["id-1"]
	if !ok {
		t.Fatal("id-1 not found after reload")
	}
	if got.State != Done {
		t.Fatalf("reloaded state = %q, want done", got.State)
	}
	if got.Usage == nil || got.Usage.TotalTokens != 7 {
		t.Fatalf("reloaded Usage = %+v, want TotalTokens=7", got.Usage)
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
	jj, restored, err := openJournal(t, dir)
	if err != nil {
		t.Fatalf("NewJournal: %v", err)
	}
	if len(restored) != 0 {
		t.Fatalf("fresh journal should reconstruct nothing, got %d", len(restored))
	}
	j := Job{ID: "id-crash", Model: "m", State: Pending}
	if err := jj.Record(j); err != nil {
		t.Fatalf("record pending: %v", err)
	}
	j.State = Running
	if err := jj.Record(j); err != nil {
		t.Fatalf("record running: %v", err)
	}
	// No terminal transition — this is the crash.

	restored2, err := LoadJournal(filepath.Join(dir, "jobs.jsonl"))
	if err != nil {
		t.Fatalf("LoadJournal after crash: %v", err)
	}
	got, ok := restored2["id-crash"]
	if !ok {
		t.Fatal("id-crash not found after reload")
	}
	if got.State != Interrupted {
		t.Fatalf("reconstructed state = %q, want interrupted (never failed, never missing)", got.State)
	}
	if got.Finished == nil {
		t.Fatal("interrupted job should have a Finished timestamp recorded")
	}
}

// TestJobJournal_reopenCompletesTheInterruptedRecord: NewJournal (the constructor a real
// restart calls, not LoadJournal directly) appends the interrupted transition back to the file
// itself, so the journal's own last line for that id stops silently saying "running".
func TestJobJournal_reopenCompletesTheInterruptedRecord(t *testing.T) {
	dir := t.TempDir()
	jj, _, err := openJournal(t, dir)
	if err != nil {
		t.Fatalf("NewJournal (1st open): %v", err)
	}
	if err := jj.Record(Job{ID: "id-crash", Model: "m", State: Pending}); err != nil {
		t.Fatal(err)
	}
	if err := jj.Record(Job{ID: "id-crash", Model: "m", State: Running}); err != nil {
		t.Fatal(err)
	}

	// Reopen — simulating the process restarting after the crash.
	_, restored, err := openJournal(t, dir)
	if err != nil {
		t.Fatalf("NewJournal (2nd open): %v", err)
	}
	if restored["id-crash"].State != Interrupted {
		t.Fatalf("state after reopen = %q, want interrupted", restored["id-crash"].State)
	}

	// The FILE's own last line for this id must now be the interrupted transition, not running —
	// a third open (no new crash) must reconstruct the SAME state, not flip anything again.
	_, restored3, err := openJournal(t, dir)
	if err != nil {
		t.Fatalf("NewJournal (3rd open): %v", err)
	}
	if restored3["id-crash"].State != Interrupted {
		t.Fatalf("state after a clean 3rd open = %q, want interrupted (stable, not re-derived)", restored3["id-crash"].State)
	}
}

// TestJobJournal_permissions: dir at dirPerm (0700), file at filePerm (0600) —
// sessions.go's own KV-snapshot rationale, reused verbatim (a job record replays the prompt).
func TestJobJournal_permissions(t *testing.T) {
	base := t.TempDir()
	dir := filepath.Join(base, "jobs")
	if _, _, err := openJournal(t, dir); err != nil {
		t.Fatalf("NewJournal: %v", err)
	}
	if runtime.GOOS == "windows" {
		// Go maps Windows file modes to the read-only bit alone, so a stat reports 0777 / 0666 whatever was asked
		// for; owner-only access there is the profile directory's ACL, not a mode bit this test can read.
		t.Skip("Unix permission bits do not exist on Windows")
	}
	di, err := os.Stat(dir)
	if err != nil {
		t.Fatalf("stat dir: %v", err)
	}
	if perm := di.Mode().Perm(); perm != 0o700 {
		t.Errorf("job dir perm = %o, want %o", perm, 0o700)
	}
	fi, err := os.Stat(filepath.Join(dir, "jobs.jsonl"))
	if err != nil {
		t.Fatalf("stat file: %v", err)
	}
	if perm := fi.Mode().Perm(); perm != 0o600 {
		t.Errorf("job file perm = %o, want %o", perm, 0o600)
	}
}

// TestJobJournal_missingFileIsNotAnError: the very first run ever has no jobs.jsonl yet.
func TestJobJournal_missingFileIsNotAnError(t *testing.T) {
	restored, err := LoadJournal(filepath.Join(t.TempDir(), "does-not-exist.jsonl"))
	if err != nil {
		t.Fatalf("LoadJournal on a missing file should not error, got %v", err)
	}
	if len(restored) != 0 {
		t.Fatalf("expected an empty map, got %d entries", len(restored))
	}
}

// A job keeps the prompt ids as int32: the same JSON, half the retained bytes of the []int the request carried. The bound is on the
// heap retained by a store full of 20K-token jobs once the request slices are gone.
func TestJobStore_promptIDsRetainFourBytesEach(t *testing.T) {
	const jobs, ids = 50, 20000
	js := &Store{jobs: map[string]*Job{}, cap: jobs}
	heap := func() uint64 {
		runtime.GC()
		var ms runtime.MemStats
		runtime.ReadMemStats(&ms)
		return ms.HeapAlloc
	}
	before := heap()
	for i := 0; i < jobs; i++ {
		req := make([]int, ids)
		for k := range req {
			req[k] = k % 150000
		}
		js.CreatePending(string(rune('a'+i%26))+strings.Repeat("x", i/26), "m", req)
	}
	retained := heap() - before
	runtime.KeepAlive(js)
	if want := uint64(jobs * ids * 4); retained > want+want/10 {
		t.Errorf("a store of %d jobs x %d ids retains %d bytes; int32 ids are %d", jobs, ids, retained, want)
	}
	b, err := json.Marshal(&Job{ID: "j", PromptIDs: []int32{1, 150000, 7}})
	if err != nil || !strings.Contains(string(b), `"prompt_ids":[1,150000,7]`) {
		t.Errorf("the job's JSON changed shape: %s (%v)", b, err)
	}
}
