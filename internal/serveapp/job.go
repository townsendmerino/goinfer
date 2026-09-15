package serveapp

import (
	"fmt"
	"os"
	"sync"
	"time"
)

// jobState is a job's lifecycle stage — J2, task-work-queue-2026-09.md. interrupted is written
// ONLY by restart recovery (jobjournal.go's loadJournal): a job never transitions into it live.
type jobState string

const (
	jobPending     jobState = "pending"
	jobRunning     jobState = "running"
	jobDone        jobState = "done"
	jobFailed      jobState = "failed"
	jobCancelled   jobState = "cancelled"
	jobInterrupted jobState = "interrupted"
)

// job is J2's record of one generation: "{id, model, request, class, state, created, started,
// finished, usage, error}" per the task doc, with `request` scoped to what drive/driveVL actually
// have at their chokepoint — the decoded prompt token ids, not the original HTTP body (retaining
// the raw body would need new plumbing from every one of the 8 handler call sites; the ids alone
// are enough to replay the prompt, which is the property the 0600 permission below exists for).
type job struct {
	ID        string     `json:"id"`
	Model     string     `json:"model"`
	PromptIDs []int      `json:"prompt_ids,omitempty"`
	Class     string     `json:"class,omitempty"` // reserved for J7; "" today
	State     jobState   `json:"state"`
	Created   time.Time  `json:"created"`
	Started   *time.Time `json:"started,omitempty"`
	Finished  *time.Time `json:"finished,omitempty"`
	Usage     *usage     `json:"usage,omitempty"`
	Error     string     `json:"error,omitempty"`
}

// jobStore is a process-wide, in-memory registry — every generation gets a job, whether or not
// anything durable is on (task doc: "Every generation gets one, whether or not anything durable
// is on"). journal is nil when -job-dir is unset; when set, every transition is also persisted.
//
// Deliberately NOT joined with generationRegistry (K1) in this pass — task-work-queue-2026-09.md's
// own J3 spec says that join is J3's job ("the two registries are joined, not parallel"), once
// /v1/jobs exists to need it. create/finish key on the SAME id K1 already mints (gr.id), so that
// join is a lookup away when J3 is built, not a retrofit.
type jobStore struct {
	mu      sync.Mutex
	jobs    map[string]*job
	journal *jobJournal
}

// newJobStore builds the in-memory registry and, when dir != "", opens/replays its journal. A
// corrupt journal is a hard error (same pattern as this file's other construction failures) —
// silently discarding a journal a restart depends on to mark interrupted jobs would defeat J2's
// whole point.
func newJobStore(dir string) (*jobStore, error) {
	s := &jobStore{jobs: map[string]*job{}}
	if dir == "" {
		return s, nil
	}
	jj, restored, err := newJobJournal(dir)
	if err != nil {
		return nil, err
	}
	s.journal = jj
	for id, j := range restored {
		s.jobs[id] = j
	}
	return s, nil
}

// create starts a job (pending → running, two recorded transitions) and returns it. Two
// transitions rather than one because the journal then shows exactly which line a crash reached:
// "pending" is the id minted; "running" is compute actually starting. Called from drive/driveVL's
// existing K1 chokepoint, gated the same way (gr.id != "").
func (s *jobStore) create(id, model string, promptIDs []int) *job {
	j := &job{ID: id, Model: model, PromptIDs: promptIDs, State: jobPending, Created: time.Now()}
	s.mu.Lock()
	s.jobs[id] = j
	s.mu.Unlock()
	s.record(j)

	now := time.Now()
	j.State = jobRunning
	j.Started = &now
	s.record(j)
	return j
}

// finish records the terminal transition. Called exactly once per job (drive/driveVL's own defer,
// after their named return values are set) — no concurrent or repeat access to the same *job to
// guard against, since one job is only ever touched by the single request that owns it.
func (s *jobStore) finish(j *job, state jobState, u *usage, errMsg string) {
	now := time.Now()
	j.State = state
	j.Finished = &now
	j.Usage = u
	j.Error = errMsg
	s.record(j)
}

// record persists a snapshot to the journal (no-op when unset), logging rather than failing the
// request on a write error — a generation that already ran must not be reported as failed because
// its AUDIT record couldn't be written.
func (s *jobStore) record(j *job) {
	if s.journal == nil {
		return
	}
	if err := s.journal.record(*j); err != nil {
		fmt.Fprintf(os.Stderr, "job journal: record %s: %v\n", j.ID, err)
	}
}
