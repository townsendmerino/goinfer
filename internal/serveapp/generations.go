package serveapp

import (
	"sync"
	"sync/atomic"
	"time"
)

// K1 (docs/task-halt-2026-09.md): a process-wide registry of in-flight generations, keyed by the
// id each handler already mints (reqID(), via chatcmpl-/msg_/resp_ prefixes). It exists so an
// operator can cancel one generation, or every generation of one model, from outside the
// process that started it — the handle a cancel-by-id needs already existed (helpers.go's
// reqID); it was just never registered anywhere.
//
// Registration happens inside drive/driveVL (the two places EVERY generation path funnels
// through — openai.go's own doc comment on drive), not in each of the ~15 call sites, so the
// bookkeeping lives in one place. A handler that wants cancellation opts in by setting
// genRequest.id before calling drive/driveVL; gr.id == "" (a caller that hasn't been wired, or a
// test) skips registration entirely — nil-safe, not an error.
//
// NOTE ON "SESSION" (found while implementing, not assumed): docs/task-halt-2026-09.md's K1 also
// asks for `POST /admin/sessions/{id}/cancel` ("every generation of that session, since an agent
// loop is a session"). That endpoint is NOT implemented here. goinfer's own "session"
// (sessionLRU/decoder.Session, sessions.go) is a content-addressed KV-reuse cache selected by
// longest-common-prefix match (bestExtend) — it has no client-visible, stable identifier an
// operator could type into a cancel request. None of the four request surfaces (chat
// completions, completions, responses, messages) carry a session/user/thread id either
// (embeddings.go's `User` field is the only "user" field in the tree, and it is explicitly
// "accepted, ignored"). Inventing one here would be exactly the kind of undocumented design
// decision the task asked to be flagged instead of worked around — see the report accompanying
// this commit.
type generation struct {
	id      string
	model   string
	started time.Time
	cancel  func()

	// tokens counts onText callback firings, a proxy for "tokens so far" (drive's onText is
	// called once per emitted UTF-8-complete/stop-string-safe chunk, which is 1:1 with tokens
	// on the common path and briefly lags it under UTF-8/stop-string holdback — close enough
	// for an operator's live status view, not a billing count).
	tokens atomic.Int64

	// cancelReason is set ONCE, by the first admin cancel to reach this generation (empty =
	// not cancelled). Read by drive/driveVL after the stream ends to decide whether an
	// externally-cancelled generation (parent.Err() != nil) was an admin cancel specifically,
	// as opposed to a client disconnect or graceful shutdown, which report "length" as before
	// (M-23's existing distinction; see openai.go's streamTokens doc comment).
	cancelMu     sync.Mutex
	cancelReason string
}

// cancelOnce sets the reason (first caller wins) and cancels the generation's context. Safe to
// call more than once (e.g. a session-wide cancel racing a by-id cancel of the same generation);
// only the first reason sticks, and cancel() is idempotent (context.CancelFunc always is).
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

// snapshot is generation's read-only view for GET /admin/generations — a plain struct rather
// than *generation itself, so a lister never holds a reference into the live registry entry
// past the RLock that produced it.
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

// register adds a generation under construction's entry. Called by drive/driveVL, never by a
// handler directly — id is whatever the handler already minted via reqID().
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

// cancelAll cancels every currently-registered generation with reason (K2's global halt) and
// returns how many it hit.
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
