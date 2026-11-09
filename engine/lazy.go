package engine

import (
	"context"
	"sync/atomic"

	"github.com/townsendmerino/goinfer/constrain"
	"github.com/townsendmerino/goinfer/decoder"
)

// lazyPhase starts one phase of a lazy-union turn over prompt with at most max tokens (0: the request's own unbounded budget) and
// sampling sp. err reports a phase that cannot start (a speculative path's sampler validation), before it touches any state.
type lazyPhase func(ctx context.Context, prompt []int, max int, sp decoder.SamplingParams) (<-chan int, *decoder.Generation, func(), error)

// lazyTurn is a tool_choice auto turn with a lazy tool-call union on a server that speculates (docs/tasks/task-tool-grammar-union-2026-09.md,
// row R2). Phase 1 runs free, the server's speculative path with no processor, and each token it emits is folded into lazy before it is
// forwarded. The token that arms lazy (its trigger, the call's opener, completes in the output) is forwarded too; phase 1 is then
// cancelled and drained, so nothing it drafted past the opener reaches the caller, and phase 2 runs the rest of the turn from prompt
// plus the output so far, gated by lazy (its Process and Gate, offset past phase 1's tokens): what a constrained turn takes on this
// server. Speculation does not resume after the call. A phase 1 that cannot start runs the whole turn as phase 2, today's lazy auto turn.
func lazyTurn(ctx context.Context, req Request, lazy *constrain.LazyMasker, free, constrained lazyPhase) *Stream {
	var cur, phase1 atomic.Pointer[decoder.Generation] // the phase now running, and phase 1 (whose speculation the report counts)
	var startErr atomic.Pointer[error]                 // a phase 2 that could not start
	var phase2At atomic.Int64                          // tokens phase 1 emitted before phase 2 began; -1 while phase 1 runs
	phase2At.Store(-1)
	out := make(chan int)
	gated := func(base []int) decoder.SamplingParams {
		sp := req.Sampling
		sp.LogitProcessor = func(generated []int, logits []float32) {
			lazy.Process(append(base[:len(base):len(base)], generated...), logits)
		}
		sp.LogitProcessorGate = func(generated []int) bool { return lazy.Gate(append(base[:len(base):len(base)], generated...)) }
		return sp
	}
	run2 := func(emitted []int) {
		budget := req.MaxTokens
		if budget > 0 {
			if budget -= len(emitted); budget <= 0 {
				return
			}
		}
		prompt := append(append([]int(nil), req.PromptIDs...), emitted...)
		s2, g2, d2, err := constrained(ctx, prompt, budget, gated(emitted))
		if err != nil {
			startErr.Store(&err)
			return
		}
		cur.Store(g2)
		phase2At.Store(int64(len(emitted)))
		for t := range s2 {
			out <- t
		}
		if d2 != nil {
			d2()
		}
	}
	go func() {
		defer close(out)
		ctx1, cancel1 := context.WithCancel(ctx)
		defer cancel1()
		s1, g1, d1, err := free(ctx1, req.PromptIDs, req.MaxTokens, req.Sampling)
		if err != nil {
			run2(nil)
			return
		}
		cur.Store(g1)
		phase1.Store(g1)
		var emitted []int
		armed := false
		for t := range s1 {
			emitted = append(emitted, t)
			out <- t
			if lazy.Gate(emitted) {
				armed = true
				break
			}
		}
		if armed {
			cancel1()
			for range s1 { // drain what phase 1 drafted past the opener; none of it is forwarded
			}
		}
		if d1 != nil {
			d1()
		}
		if armed {
			run2(emitted)
		}
	}()
	return NewStream(out, func() *Report {
		if e := startErr.Load(); e != nil {
			return &Report{Err: *e}
		}
		r := reportOf(cur.Load())
		if n := phase2At.Load(); r != nil && n >= 0 {
			r.LazyArmed = true
			if r.BudgetClamped {
				r.Budget += int(n) // phase 2's budget is the turn's less what phase 1 emitted
			}
			if p1 := reportOf(phase1.Load()); p1 != nil {
				r.SpecDrafted, r.SpecAccepted = p1.SpecDrafted, p1.SpecAccepted
			}
		}
		return r
	}, nil)
}
