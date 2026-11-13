// Package jobs holds the serving shell's generation records: the job store and its durable journal, the per-job event log, the
// turn clock that stamps a turn's timings and cost, and the usage block a job carries. It imports no other part of the shell.
package jobs

import (
	"context"
	"fmt"
	"os"
	"sync"
	"time"

	"github.com/townsendmerino/goinfer/internal/cost"
)

// State is a job's lifecycle stage (docs/tasks/task-work-queue-2026-09.md). interrupted is written ONLY by restart recovery
// (LoadJournal): a job never transitions into it live.
type State string

const (
	Pending     State = "pending"
	Running     State = "running"
	Done        State = "done"
	Failed      State = "failed"
	Cancelled   State = "cancelled"
	Interrupted State = "interrupted"
)

// job is the record of one generation: {id, model, request, class, state, created, started, finished, usage, error}, with
// `request` scoped to what drive/driveVL actually have at their chokepoint: the decoded prompt token ids, not the original
// HTTP body (retaining the raw body would need new plumbing from every handler call site; the ids alone are enough to replay
// the prompt, which is why the journal is owner-only 0600).
//
// Result is the assembled output, filled at the same point Usage/Error are (GET /v1/jobs/{id}'s "result"); nil until the job
// reaches a terminal state. Unexported and not journaled: it can be arbitrarily large free-form text, and nothing needs it to
// survive a restart (an interrupted job has no result to report either way).
type Job struct {
	ID        string     `json:"id"`
	Model     string     `json:"model"`
	PromptIDs []int32    `json:"prompt_ids,omitempty"` // int32: a vocabulary id fits, and a capped store of long prompts holds half the bytes
	Class     string     `json:"class,omitempty"`      // reserved for a future job class; "" today
	State     State      `json:"state"`
	Created   time.Time  `json:"created"`
	Started   *time.Time `json:"started,omitempty"`
	Finished  *time.Time `json:"finished,omitempty"`
	Usage     *Usage     `json:"usage,omitempty"`
	// Timings and Cost are the finished turn's "timings" and, when its request asked, "cost" (jobs/turn_timings.go), so a
	// re-attach reads what the live response carried.
	Timings *TimingsBlock `json:"timings,omitempty"`
	Cost    *cost.Report  `json:"cost,omitempty"`
	Error   string        `json:"error,omitempty"`
	Result  *Result       `json:"-"`
}

// Result is a job's assembled output. FinishReason mirrors the OpenAI chat completion field ("stop" | "length" |
// "cancelled"). StopSeq is drive's own stop-string match, when FinishReason came from one, so a caller in Anthropic's
// vocabulary can call anthropicStopReason(finish, stopSeq) and keep which stop sequence matched; "" for every non-stop-string
// Finish, and never read by the OpenAI surface.
type Result struct {
	Content      string `json:"content"`
	FinishReason string `json:"finish_reason"`
	StopSeq      string `json:"-"`
}

// Store is a process-wide, in-memory registry: every generation gets a job, whether or not anything durable is on. journal
// is nil when -job-dir is unset; when set, every transition is also persisted.
//
// Joined with generationRegistry by SHARING AN ID, not by a structural link: cancelling a job calls the SAME cancel this store
// holds for it (see cancel), and drive/driveVL separately register that same id in the generation registry once actually
// running, so /admin/generations/{id}/cancel and DELETE /v1/jobs/{id} both reach the same generation without either registry
// knowing about the other (docs/tasks/task-work-queue-2026-09.md).
//
// cap bounds jobs (FIFO eviction of the oldest FINISHED job, never one still pending/running), mirroring responseStore's cap
// pattern; 0 = unbounded (only used by tests and the zero-value case main.go never constructs).
type Store struct {
	mu      sync.Mutex
	jobs    map[string]*Job
	order   []string // insertion order, for FIFO eviction — parallels responseStore.order
	cap     int
	logs    map[string]*EventLog
	cancels map[string]context.CancelFunc
	journal *Journal
}

// NewStore builds the in-memory registry and, when dir != "", opens/replays its journal. A corrupt journal is a hard error:
// silently discarding a journal a restart depends on to mark interrupted jobs would defeat its purpose. cap <= 0 means
// unbounded.
func NewStore(dir string, cap int) (*Store, error) {
	s := &Store{
		jobs: map[string]*Job{}, cap: cap,
		logs: map[string]*EventLog{}, cancels: map[string]context.CancelFunc{},
	}
	if dir == "" {
		return s, nil
	}
	jj, restored, err := NewJournal(dir)
	if err != nil {
		return nil, err
	}
	s.journal = jj
	for id, j := range restored {
		s.jobs[id] = j
		s.order = append(s.order, id)
	}
	return s, nil
}

// Create starts a job (pending -> running, two recorded transitions) and returns it: the SYNCHRONOUS handlers' case, where
// admission has already completed by the time drive/driveVL reach their generation-registry chokepoint, so there is no
// observable "pending" window to represent. Two transitions rather than one because the journal then shows exactly which line
// a crash reached: "pending" is the id minted; "running" is compute actually starting.
func (s *Store) Create(id, model string, promptIDs []int) *Job {
	j := s.CreatePending(id, model, promptIDs)
	s.markRunning(j)
	return j
}

// CreatePending records just the first transition. The asynchronous path (POST /v1/jobs) calls this directly, BEFORE
// admission, so the id is immediately visible to GET /v1/jobs/{id} while still queued. Also applies the store's FIFO eviction
// (never evicting a pending/running job).
func (s *Store) CreatePending(id, model string, promptIDs []int) *Job {
	ids := make([]int32, len(promptIDs))
	for i, t := range promptIDs {
		ids[i] = int32(t)
	}
	j := &Job{ID: id, Model: model, PromptIDs: ids, State: Pending, Created: time.Now()}
	s.mu.Lock()
	s.jobs[id] = j
	s.order = append(s.order, id)
	s.evictLocked()
	s.mu.Unlock()
	s.Record(j)
	return j
}

// GetOrCreate is drive/driveVL's own chokepoint call (replacing a bare create()): if id was
// already pre-created (the async path), it is transitioned to running now; otherwise it is
// created fresh with both transitions immediately, identical to create()'s own behavior — so this
// changes nothing observable for every existing synchronous caller.
func (s *Store) GetOrCreate(id, model string, promptIDs []int) *Job {
	if j := s.Get(id); j != nil {
		s.markRunning(j)
		return j
	}
	return s.Create(id, model, promptIDs)
}

// markRunning records the pending -> running transition for an already-created job. A job has a genuine concurrent reader:
// handleGetJob runs on the HTTP handler's goroutine while runJob's goroutine calls this and finish, so every mutation is under
// s.mu; see snapshot for the matching read side.
func (s *Store) markRunning(j *Job) {
	s.mu.Lock()
	now := time.Now()
	j.State = Running
	j.Started = &now
	s.mu.Unlock()
	s.Record(j)
}

// Finish records the terminal transition, including the result (nil for the synchronous paths, which have nowhere to read it
// back from; only runJob's async path passes one). Called exactly once per job (drive/driveVL's own defer, after their named
// return values are set): one goroutine owns the mutation, but handleGetJob reads concurrently, see markRunning.
func (s *Store) Finish(j *Job, state State, u *Usage, errMsg string, res *Result) {
	s.FinishTurn(j, state, u, nil, errMsg, res)
}

// FinishTurn is finish for a job whose generation ran: it also records the turn's timings and cost (clock, nil for none).
func (s *Store) FinishTurn(j *Job, state State, u *Usage, clock *TurnClock, errMsg string, res *Result) {
	timings, rep := clock.Timings(), clock.CostReport()
	s.mu.Lock()
	now := time.Now()
	j.State = state
	j.Finished = &now
	j.Usage = u
	j.Timings, j.Cost = timings, rep
	j.Error = errMsg
	j.Result = res
	s.mu.Unlock()
	s.Record(j)
}

// Get returns the LIVE job pointer for id, or nil — for this file's own internal use only
// (getOrCreate, evictLocked), where the caller is about to mutate it itself (already under s.mu,
// or about to take it via markRunning/finish) or only needs to test existence. Never read a
// field off this pointer outside s.mu: use snapshot for that.
func (s *Store) Get(id string) *Job {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.jobs[id]
}

// Snapshot returns a COPY of id's job (ok=false if unknown), taken under s.mu — this is what any
// reader outside the owning goroutine (handleGetJob) must use instead of get(), so reading a
// job's fields never races its owner's markRunning/finish. A shallow copy is sufficient: every
// pointer field (Started, Finished, Usage, Result) is always assigned a freshly allocated value,
// never mutated in place after the fact, so sharing those pointers with the copy is safe.
func (s *Store) Snapshot(id string) (Job, bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	j, ok := s.jobs[id]
	if !ok {
		return Job{}, false
	}
	return *j, true
}

// evictLocked drops the oldest FINISHED jobs (and their event logs/cancels) once len(jobs) exceeds cap. Called with s.mu held.
// Mirrors responseStore.put's eviction loop but skips over any still-pending/running entry rather than evicting it: a job in
// flight must never disappear out from under a client polling it.
func (s *Store) evictLocked() {
	if s.cap <= 0 {
		return
	}
	for i := 0; len(s.order) > s.cap && i < len(s.order); {
		id := s.order[i]
		j := s.jobs[id]
		if j == nil || IsTerminal(j.State) {
			delete(s.jobs, id)
			delete(s.logs, id)
			delete(s.cancels, id)
			s.order = append(s.order[:i], s.order[i+1:]...)
			continue
		}
		i++ // skip a live job; try the next-oldest
	}
}

// IsTerminal reports whether st is a final state: the job will not change again.
func IsTerminal(st State) bool {
	switch st {
	case Done, Failed, Cancelled, Interrupted:
		return true
	default:
		return false
	}
}

// Attach registers id's event log — called once, by runJob, right after createPending.
func (s *Store) Attach(id string, log *EventLog) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.logs[id] = log
}

// EventLog returns id's event log, or nil (unknown id, or a job that predates this process —
// restored from the journal on restart, whose live output is gone with the process that ran it).
func (s *Store) EventLog(id string) *EventLog {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.logs[id]
}

// SetCancel records id's cancel func — called once, by runJob, before admission is even
// attempted, so DELETE /v1/jobs/{id} can stop a job whether it is still queued or already running
// (cancelling the outer context propagates to whatever drive derives from it once running, the
// same way any other context cancellation in this codebase does).
func (s *Store) SetCancel(id string, cancel context.CancelFunc) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.cancels[id] = cancel
}

// Cancel looks up id's cancel func and calls it, reporting whether one was found. A caller cancelling an id that already
// finished (or never existed) is not an error, mirroring generationRegistry.cancel.
func (s *Store) Cancel(id string) bool {
	s.mu.Lock()
	c := s.cancels[id]
	s.mu.Unlock()
	if c == nil {
		return false
	}
	c()
	return true
}

// Record persists a snapshot to the journal (no-op when unset), logging rather than failing the
// request on a write error — a generation that already ran must not be reported as failed because
// its AUDIT record couldn't be written.
func (s *Store) Record(j *Job) {
	if s.journal == nil {
		return
	}
	if err := s.journal.Record(*j); err != nil {
		fmt.Fprintf(os.Stderr, "job journal: record %s: %v\n", j.ID, err)
	}
}
