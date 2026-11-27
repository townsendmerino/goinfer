package stores

import (
	"sync"
	"sync/atomic"
	"time"
)

// Generation is one in-flight generation in the process-wide GenerationRegistry, keyed by the request ID minted by the
// surface handler, so an operator can cancel one generation or halt all in-flight generations cleanly.
type Generation struct {
	id      string
	model   string
	started time.Time
	cancel  func()

	tokens atomic.Int64

	cancelMu     sync.Mutex
	cancelReason string
}

// AddTokens increments the live emitted-chunk counter reported by GET /admin/generations.
func (g *Generation) AddTokens(delta int64) { g.tokens.Add(delta) }

// CancelOnce records reason (first caller wins) and invokes the generation's cancel func. Safe to call concurrently.
func (g *Generation) CancelOnce(reason string) {
	g.cancelMu.Lock()
	if g.cancelReason == "" {
		g.cancelReason = reason
	}
	g.cancelMu.Unlock()
	g.cancel()
}

// Reason returns the admin cancel reason, or "" if this generation was never admin-cancelled.
func (g *Generation) Reason() string {
	g.cancelMu.Lock()
	defer g.cancelMu.Unlock()
	return g.cancelReason
}

// GenerationSnapshot is a Generation's read-only view for GET /admin/generations.
type GenerationSnapshot struct {
	ID          string    `json:"id"`
	Model       string    `json:"model"`
	Started     time.Time `json:"started"`
	TokensSoFar int64     `json:"tokens_so_far"`
}

// GenerationRegistry tracks in-flight generations across a server instance.
type GenerationRegistry struct {
	mu   sync.Mutex
	gens map[string]*Generation
}

// NewGenerationRegistry constructs an empty GenerationRegistry.
func NewGenerationRegistry() *GenerationRegistry {
	return &GenerationRegistry{gens: map[string]*Generation{}}
}

// Register adds an entry for an in-flight generation and returns its handle.
func (r *GenerationRegistry) Register(id, model string, cancel func()) *Generation {
	g := &Generation{id: id, model: model, started: time.Now(), cancel: cancel}
	r.mu.Lock()
	r.gens[id] = g
	r.mu.Unlock()
	return g
}

// Remove deletes id once the generation has ended. Idempotent.
func (r *GenerationRegistry) Remove(id string) {
	r.mu.Lock()
	delete(r.gens, id)
	r.mu.Unlock()
}

// Cancel looks up id and cancels it with reason, reporting whether a live entry was found.
func (r *GenerationRegistry) Cancel(id, reason string) bool {
	r.mu.Lock()
	g, ok := r.gens[id]
	r.mu.Unlock()
	if !ok {
		return false
	}
	g.CancelOnce(reason)
	return true
}

// CancelAll cancels every currently registered generation with reason and returns how many were signalled.
func (r *GenerationRegistry) CancelAll(reason string) int {
	r.mu.Lock()
	gens := make([]*Generation, 0, len(r.gens))
	for _, g := range r.gens {
		gens = append(gens, g)
	}
	r.mu.Unlock()
	for _, g := range gens {
		g.CancelOnce(reason)
	}
	return len(gens)
}

// List returns snapshots of all currently registered generations.
func (r *GenerationRegistry) List() []GenerationSnapshot {
	r.mu.Lock()
	defer r.mu.Unlock()
	out := make([]GenerationSnapshot, 0, len(r.gens))
	for _, g := range r.gens {
		out = append(out, GenerationSnapshot{ID: g.id, Model: g.model, Started: g.started, TokensSoFar: g.tokens.Load()})
	}
	return out
}

// Count reports how many generations are currently registered.
func (r *GenerationRegistry) Count() int {
	r.mu.Lock()
	defer r.mu.Unlock()
	return len(r.gens)
}

// WaitEmpty polls until no generation is registered or timeout elapses, returning the elapsed wait duration.
func (r *GenerationRegistry) WaitEmpty(timeout time.Duration) time.Duration {
	start := time.Now()
	const poll = 10 * time.Millisecond
	deadline := start.Add(timeout)
	for {
		if r.Count() == 0 {
			return time.Since(start)
		}
		if time.Now().After(deadline) {
			return time.Since(start)
		}
		time.Sleep(poll)
	}
}
