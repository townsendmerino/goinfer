package decoder

import (
	"context"
	"time"
)

// decodeClock is plain decoding's per-phase timing under decodeTiming: t0 is the start of the phase being timed, shared
// by plainDecodeLoop and its target's hooks, which time their own phases.
type decodeClock struct {
	t0                       time.Time
	proc, sample, embed, fwd time.Duration
	n                        int
}

// decodeTarget is the model as plainDecodeLoop drives it (SB5, docs/tasks/task-subsystems-2026-10.md §8). generateInto
// builds one per generation from the path it chose; a test builds one over a scripted fake.
//   - pick returns the token the backend already chose for this step, with its draw's info where the backend has one
//     (drew true), so the host sampler does not run, or leaves the step's row for logits (drew false). It may consume
//     randomness only as the host sampler would.
//   - logits is the step's row for the host sampler; the processor masks it in place.
//   - advance forwards next, which has been emitted, and leaves the next step's state. needFull says the gated processor
//     wants the next row whole.
//   - send, when set, may emit next before the loop's own send and reports whether it did (MC3's hold grace).
//   - finish, when set, runs on a cancel and at the end of a completed decode, never after an error, with the tokens
//     emitted: it records what the KV holds.
type decodeTarget struct {
	pick    func(clk *decodeClock) (info SampleInfo, next int, drew bool, err error)
	logits  func() []float32
	advance func(next int, needFull bool, clk *decodeClock) error
	send    func(next int) bool
	finish  func(generated []int)
}

// plainDecodeLoop is plain decoding from the seed state tg holds: up to maxTokens tokens, each picked by the backend or
// drawn by sampler from the step's row (after sp's processor), stopped at a stop id (not emitted), streamed to out, then
// forwarded; commit, if set, sees each id once its forward has committed it. It returns true when the decode completed
// (maxTokens or a stop), false when it ended on a cancel or an error, which it records on g.
func plainDecodeLoop(ctx context.Context, tg decodeTarget, out chan<- int, g *Generation, sampler *Sampler, sp SamplingParams,
	maxTokens int, isStop func(int) bool, commit func(int), clk *decodeClock) bool {
	gated := sp.LogitProcessor != nil && sp.LogitProcessorGate != nil
	needFull := gated && sp.LogitProcessorGate(nil)
	var generated []int
	finish := func() {
		if tg.finish != nil {
			tg.finish(generated)
		}
	}
	for range maxTokens {
		select {
		case <-ctx.Done():
			g.err = ctx.Err()
			// A cancel that arrives during the previous step's forward is observed here; that step's token is in generated
			// and its forward has run, so the KV is as consistent as at the end of a completed decode.
			finish()
			return false
		default:
		}
		if decodeTiming {
			clk.t0 = time.Now()
		}
		info, next, drew, err := tg.pick(clk)
		if err != nil {
			g.err = err
			return false
		}
		if !drew {
			logits := tg.logits()
			// Constrained decoding: the processor masks this step's row (from what has been generated) before the draw
			// and the stop check.
			if sp.LogitProcessor != nil && (!gated || needFull) {
				sp.LogitProcessor(generated, logits)
			}
			if decodeTiming {
				clk.proc += time.Since(clk.t0)
				clk.t0 = time.Now()
			}
			info, err = sampler.SampleWithInfo(logits)
			if err != nil {
				g.err = err
				return false
			}
			if decodeTiming {
				clk.sample += time.Since(clk.t0)
			}
			next = info.ID
		}
		if isStop(next) {
			break
		}
		if sp.Logprobs {
			g.Logprobs = append(g.Logprobs, info)
		}
		// A bare send wedges this goroutine if the consumer stops ranging, holding the KV and, for a session, its
		// reconciliation; select on ctx like every speculative path.
		if tg.send == nil || !tg.send(next) {
			select {
			case <-ctx.Done():
				g.err = ctx.Err()
				// next was drawn but never forwarded: the KV holds exactly prompt+generated, which finish records.
				finish()
				return false
			case out <- next:
			}
		}
		generated = append(generated, next)
		if gated {
			needFull = sp.LogitProcessorGate(generated)
		}
		err = tg.advance(next, needFull, clk)
		if decodeTiming {
			clk.fwd += time.Since(clk.t0)
			clk.n++
		}
		if kvPoolFullEnds(g, err, len(generated)) {
			return false // this token's K/V row was not written, so nothing is committed: the next turn prefills its prompt
		}
		if err != nil {
			g.err = err
			return false
		}
		if commit != nil {
			commit(next)
		}
	}
	finish()
	return true
}
