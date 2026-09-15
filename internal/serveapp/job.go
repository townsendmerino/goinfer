package serveapp

import (
	"context"
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
//
// Result is J3's addition (task-work-queue-2026-09.md): the assembled output, filled at the same
// point Usage/Error are — GET /v1/jobs/{id}'s "result". nil until the job reaches a terminal
// state. Unexported and not journaled: it can be arbitrarily large free-form text, unlike the
// small fixed fields above, and nothing needs it to survive a restart the way the state machine
// itself does (an interrupted job has no result to report either way).
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
	result    *jobResult
}

// jobResult is a job's assembled output — J3. FinishReason mirrors the OpenAI chat completion
// field ("stop" | "length" | "cancelled").
//
// StopSeq is J4's addition (task-work-queue-2026-09.md): drive's own stop-string match, when
// FinishReason came from one, so a caller in Anthropic's vocabulary can call the exact
// anthropicStopReason(finish, stopSeq) helper serveMessagesWith already uses instead of losing
// "which stop sequence matched" fidelity. "" for every non-stop-string finish, and a no-op for the
// OpenAI surface, which never reads it.
type jobResult struct {
	Content      string `json:"content"`
	FinishReason string `json:"finish_reason"`
	StopSeq      string `json:"-"`
}

// jobStore is a process-wide, in-memory registry — every generation gets a job, whether or not
// anything durable is on (task doc: "Every generation gets one, whether or not anything durable
// is on"). journal is nil when -job-dir is unset; when set, every transition is also persisted.
//
// Joined with generationRegistry (K1) by SHARING AN ID, not by a structural link — J3
// (task-work-queue-2026-09.md) cancels a job by calling the SAME cancel this store already holds
// for it (see cancel below), and drive/driveVL separately register that same id in K1's registry
// once actually running, so /admin/generations/{id}/cancel and DELETE /v1/jobs/{id} both reach the
// same generation without either registry knowing about the other.
//
// cap bounds jobs (FIFO eviction of the oldest FINISHED job — never one still pending/running),
// mirroring responseStore's own cap pattern (responses.go:49); 0 = unbounded (only used by tests
// and the zero-value case main.go never actually constructs).
type jobStore struct {
	mu      sync.Mutex
	jobs    map[string]*job
	order   []string // insertion order, for FIFO eviction — parallels responseStore.order
	cap     int
	logs    map[string]*jobEventLog
	cancels map[string]context.CancelFunc
	journal *jobJournal
}

// newJobStore builds the in-memory registry and, when dir != "", opens/replays its journal. A
// corrupt journal is a hard error (same pattern as this file's other construction failures) —
// silently discarding a journal a restart depends on to mark interrupted jobs would defeat J2's
// whole point. cap <= 0 means unbounded.
func newJobStore(dir string, cap int) (*jobStore, error) {
	s := &jobStore{
		jobs: map[string]*job{}, cap: cap,
		logs: map[string]*jobEventLog{}, cancels: map[string]context.CancelFunc{},
	}
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
		s.order = append(s.order, id)
	}
	return s, nil
}

// create starts a job (pending → running, two recorded transitions) and returns it — the
// SYNCHRONOUS handlers' case, where admission has already completed by the time drive/driveVL
// reach their K1 chokepoint, so there is no observable "pending" window to represent either way.
// Two transitions rather than one because the journal then shows exactly which line a crash
// reached: "pending" is the id minted; "running" is compute actually starting.
func (s *jobStore) create(id, model string, promptIDs []int) *job {
	j := s.createPending(id, model, promptIDs)
	s.markRunning(j)
	return j
}

// createPending records just the first transition — J3's asynchronous path (POST /v1/jobs) calls
// this directly, BEFORE admission, so the id is immediately visible to GET /v1/jobs/{id} while
// still queued. Also applies the store's FIFO eviction (never evicting a pending/running job).
func (s *jobStore) createPending(id, model string, promptIDs []int) *job {
	j := &job{ID: id, Model: model, PromptIDs: promptIDs, State: jobPending, Created: time.Now()}
	s.mu.Lock()
	s.jobs[id] = j
	s.order = append(s.order, id)
	s.evictLocked()
	s.mu.Unlock()
	s.record(j)
	return j
}

// getOrCreate is drive/driveVL's own chokepoint call (replacing a bare create()): if id was
// already pre-created (the async path), it is transitioned to running now; otherwise it is
// created fresh with both transitions immediately, identical to create()'s own behavior — so this
// changes nothing observable for every existing synchronous caller.
func (s *jobStore) getOrCreate(id, model string, promptIDs []int) *job {
	if j := s.get(id); j != nil {
		s.markRunning(j)
		return j
	}
	return s.create(id, model, promptIDs)
}

// markRunning records the pending -> running transition for an already-created job.
//
// J3 (task-work-queue-2026-09.md) added a genuine concurrent reader of *job's fields:
// handleGetJob, running on the HTTP handler's own goroutine while runJob's goroutine calls this
// and finish (below). J2 never needed s.mu held across a field mutation — nothing read a job's
// fields from outside the single goroutine that owned it. Every mutation here is under s.mu now;
// see snapshot's own comment for the matching read-side fix (found by -race, not by inspection).
func (s *jobStore) markRunning(j *job) {
	s.mu.Lock()
	now := time.Now()
	j.State = jobRunning
	j.Started = &now
	s.mu.Unlock()
	s.record(j)
}

// finish records the terminal transition, including J3's result (nil for the synchronous paths,
// which have nowhere to read it back from anyway — only runJob's async path passes one). Called
// exactly once per job (drive/driveVL's own defer, after their named return values are set) — no
// concurrent MUTATOR to guard against (one job is only ever finished by the single goroutine that
// owns it), but concurrent READERS exist since J3 (handleGetJob) — see markRunning's doc comment.
func (s *jobStore) finish(j *job, state jobState, u *usage, errMsg string, res *jobResult) {
	s.mu.Lock()
	now := time.Now()
	j.State = state
	j.Finished = &now
	j.Usage = u
	j.Error = errMsg
	j.result = res
	s.mu.Unlock()
	s.record(j)
}

// get returns the LIVE job pointer for id, or nil — for this file's own internal use only
// (getOrCreate, evictLocked), where the caller is about to mutate it itself (already under s.mu,
// or about to take it via markRunning/finish) or only needs to test existence. Never read a
// field off this pointer outside s.mu: use snapshot for that.
func (s *jobStore) get(id string) *job {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.jobs[id]
}

// snapshot returns a COPY of id's job (ok=false if unknown), taken under s.mu — this is what any
// reader outside the owning goroutine (handleGetJob) must use instead of get(), so reading a
// job's fields never races its owner's markRunning/finish. A shallow copy is sufficient: every
// pointer field (Started, Finished, Usage, result) is always assigned a freshly allocated value,
// never mutated in place after the fact, so sharing those pointers with the copy is safe.
func (s *jobStore) snapshot(id string) (job, bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	j, ok := s.jobs[id]
	if !ok {
		return job{}, false
	}
	return *j, true
}

// evictLocked drops the oldest FINISHED jobs (and their event logs/cancels) once len(jobs)
// exceeds cap. Called with s.mu held. Mirrors responseStore.put's own eviction loop
// (responses.go:66) but skips over any still-pending/running entry rather than evicting it —
// a job in flight must never disappear out from under a client polling it.
func (s *jobStore) evictLocked() {
	if s.cap <= 0 {
		return
	}
	for i := 0; len(s.order) > s.cap && i < len(s.order); {
		id := s.order[i]
		j := s.jobs[id]
		if j == nil || isTerminal(j.State) {
			delete(s.jobs, id)
			delete(s.logs, id)
			delete(s.cancels, id)
			s.order = append(s.order[:i], s.order[i+1:]...)
			continue
		}
		i++ // skip a live job; try the next-oldest
	}
}

func isTerminal(st jobState) bool {
	switch st {
	case jobDone, jobFailed, jobCancelled, jobInterrupted:
		return true
	default:
		return false
	}
}

// attach registers id's event log — called once, by runJob, right after createPending.
func (s *jobStore) attach(id string, log *jobEventLog) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.logs[id] = log
}

// eventLog returns id's event log, or nil (unknown id, or a job that predates this process —
// restored from the journal on restart, whose live output is gone with the process that ran it).
func (s *jobStore) eventLog(id string) *jobEventLog {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.logs[id]
}

// setCancel records id's cancel func — called once, by runJob, before admission is even
// attempted, so DELETE /v1/jobs/{id} can stop a job whether it is still queued or already running
// (cancelling the outer context propagates to whatever drive derives from it once running, the
// same way any other context cancellation in this codebase does).
func (s *jobStore) setCancel(id string, cancel context.CancelFunc) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.cancels[id] = cancel
}

// cancel looks up id's cancel func and calls it, reporting whether one was found — a caller
// cancelling an id that already finished (or never existed) is not an error, mirroring K1's own
// cancel-by-id semantics (generations.go's registry.cancel).
func (s *jobStore) cancel(id string) bool {
	s.mu.Lock()
	c := s.cancels[id]
	s.mu.Unlock()
	if c == nil {
		return false
	}
	c()
	return true
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
