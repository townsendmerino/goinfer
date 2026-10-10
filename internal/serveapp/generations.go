package serveapp

import (
	"sync"
	"sync/atomic"
	"time"
)

// generation is one in-flight generation in the process-wide registry (generationRegistry), keyed by the id each
// handler already mints (reqID(), via the chatcmpl-/msg_/resp_ prefixes), so an operator can cancel one generation,
// or every generation of one model, from outside the process that started it (docs/tasks/task-halt-2026-09.md).
//
// Registration happens inside drive/driveVL, which every generation path funnels through, not at each call site. A
// handler that wants cancellation opts in by setting genRequest.id before calling drive/driveVL; gr.id == "" (a
// caller that has not been wired, or a test) skips registration, nil-safe.
//
// There is no cancel-by-session. goinfer's session (sessionLRU/decoder.Session, sessions.go) is a content-addressed
// KV-reuse cache selected by longest-common-prefix match (bestExtend) and has no client-visible, stable identifier an
// operator could type into a cancel request, and none of the four request surfaces carries a session, user or thread
// id (embeddings' `User` field is accepted and ignored). The task doc's `POST /admin/sessions/{id}/cancel` is
// therefore not implemented (the task doc defers it to its K4 item); inventing an id would be a design decision to
// take, not to work around.
type generation struct {
	id      string
	model   string
	started time.Time
	cancel  func()

	// tokens counts onText callback firings, a proxy for tokens so far: drive's onText fires once per emitted
	// UTF-8-complete, stop-string-safe chunk, which is 1:1 with tokens on the common path and briefly lags it under
	// holdback. Close enough for an operator's live status view, not a billing count.
	tokens atomic.Int64

	// cancelReason is set once, by the first admin cancel to reach this generation (empty = not cancelled).
	// drive/driveVL read it after the stream ends to tell an admin cancel from a client disconnect or graceful
	// shutdown, which report "length" (see streamTokens).
	cancelMu     sync.Mutex
	cancelReason string
}

// cancelOnce sets the reason (first caller wins) and cancels the generation's context. Safe to call more than once (a
// global halt racing a by-id cancel of the same generation): only the first reason sticks, and cancel is idempotent.
func (g *generation) cancelOnce(reason string) {
	g.cancelMu.Lock()
	if g.cancelReason == "" {
		g.cancelReason = reason
	}
	g.cancelMu.Unlock()
	g.cancel()
}

// reason returns the cancel reason, or "" if this generation was never admin-cancelled.
func (g *generation) reason() string {
	g.cancelMu.Lock()
	defer g.cancelMu.Unlock()
	return g.cancelReason
}

// generationSnapshot is a generation's read-only view for GET /admin/generations: a plain struct, so a lister never
// holds a reference into the live registry entry past the lock that produced it.
type generationSnapshot struct {
	ID          string    `json:"id"`
	Model       string    `json:"model"`
	Started     time.Time `json:"started"`
	TokensSoFar int64     `json:"tokens_so_far"`
}

// generationRegistry is process-wide (one per server), guarding a plain map. Small and
// short-lived entries (one per in-flight generation), so a mutex is plenty — no need for
// sync.Map's write-once/read-many shape here.
type generationRegistry struct {
	mu   sync.Mutex
	gens map[string]*generation
}

func newGenerationRegistry() *generationRegistry {
	return &generationRegistry{gens: map[string]*generation{}}
}

// register adds an entry for a generation. Called by drive/driveVL, never by a handler directly; id is whatever the
// handler already minted via reqID().
func (r *generationRegistry) register(id, model string, cancel func()) *generation {
	g := &generation{id: id, model: model, started: time.Now(), cancel: cancel}
	r.mu.Lock()
	r.gens[id] = g
	r.mu.Unlock()
	return g
}

// remove drops the entry once the generation has ended (drive/driveVL's defer). Idempotent.
func (r *generationRegistry) remove(id string) {
	r.mu.Lock()
	delete(r.gens, id)
	r.mu.Unlock()
}

// cancel looks up id and cancels it with reason, reporting whether it found a live entry (a
// caller cancelling an id that already completed is not an error — the generation is already
// stopped, which is what the caller wanted).
func (r *generationRegistry) cancel(id, reason string) bool {
	r.mu.Lock()
	g, ok := r.gens[id]
	r.mu.Unlock()
	if !ok {
		return false
	}
	g.cancelOnce(reason)
	return true
}

// cancelAll cancels every currently registered generation with reason (the global halt) and returns how many it hit.
func (r *generationRegistry) cancelAll(reason string) int {
	r.mu.Lock()
	gens := make([]*generation, 0, len(r.gens))
	for _, g := range r.gens {
		gens = append(gens, g)
	}
	r.mu.Unlock()
	for _, g := range gens {
		g.cancelOnce(reason)
	}
	return len(gens)
}

// list returns a stable-ordered snapshot of every in-flight generation, for GET /admin/generations.
func (r *generationRegistry) list() []generationSnapshot {
	r.mu.Lock()
	defer r.mu.Unlock()
	out := make([]generationSnapshot, 0, len(r.gens))
	for _, g := range r.gens {
		out = append(out, generationSnapshot{ID: g.id, Model: g.model, Started: g.started, TokensSoFar: g.tokens.Load()})
	}
	return out
}

// count reports how many generations are currently registered.
func (r *generationRegistry) count() int {
	r.mu.Lock()
	defer r.mu.Unlock()
	return len(r.gens)
}

// waitEmpty polls until no generation is registered, or timeout elapses, and returns how long it took (the halt's
// time to quiescence, docs/tasks/task-halt-2026-09.md). Each registered generation stops at its next per-token ctx
// check, the fastest signal there is, so this is normally fast; the poll interval trades a little latency in the
// measurement for not spinning a goroutine per halt. The halt path calls it after cancelAll: the cancel calls return
// immediately (they only flip a context), but "halted" should mean the work actually stopped.
func (r *generationRegistry) waitEmpty(timeout time.Duration) time.Duration {
	start := time.Now()
	const poll = 10 * time.Millisecond
	deadline := start.Add(timeout)
	for {
		if r.count() == 0 {
			return time.Since(start)
		}
		if time.Now().After(deadline) {
			return time.Since(start)
		}
		time.Sleep(poll)
	}
}
