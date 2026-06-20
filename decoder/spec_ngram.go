package decoder

import (
	"context"
	"fmt"
	"slices"
)

// Drafter is the cheap half of speculative decoding: it proposes continuation
// tokens for a context, which the target model then verifies in one batched pass.
// Because the target holds the true distribution and decides what commits, a
// Drafter can never change the output — only the speed. This is the seam the
// speculation spokes plug into (n-gram here; grammar and a feature head later);
// see docs/spec for the design.
type Drafter interface {
	// Draft proposes up to k continuation tokens for the token context ctx
	// (ctx[len-1] is the most recent confirmed token). It returns fewer than k
	// — possibly none — when it has no confident proposal. A miss is cheap: the
	// caller just falls back to one ordinary decode step that round.
	Draft(ctx []int, k int) []int
}

// NgramDrafter is a zero-model "prompt-lookup" drafter (02-cache-ngram): it finds
// the most recent earlier occurrence of the current token suffix within the
// running context (prompt + everything generated so far) and proposes the tokens
// that followed it. It costs no model compute and works on a single model — no
// separate draft model is needed. It earns its keep when the output echoes the
// input (code edits, RAG, agent loops with a fixed system prompt); on novel prose
// it simply misses, which is free.
//
// This is the deliberate "n-gram hash" baseline from the design doc, not yet the
// suffix automaton — start simple, measure the gap. Matching is exact-suffix and
// greedy-longest: try the longest pattern first, take the most recent earlier hit.
type NgramDrafter struct {
	MinMatch int // shortest suffix length trusted as a match (default 2)
	MaxMatch int // longest suffix length probed (default 16); caps the scan

	lastMatch int // suffix length of the most recent Draft hit (0 on miss); for SpecTrace
}

// DraftInfo is optional per-Draft metadata a Drafter may expose for SpecTrace.
type DraftInfo struct {
	MatchLen int // n-gram suffix-match length behind the last proposal (0 = miss)
}

// TracingDrafter is the optional capability a Drafter implements to feed
// instrumentation. LastDraftInfo is valid only immediately after Draft, and only
// under single-flight use (one in-flight Draft at a time) — which the decode loop
// guarantees per generation. Used by the trace path; never on the hot path.
type TracingDrafter interface {
	Drafter
	LastDraftInfo() DraftInfo
}

// LastDraftInfo reports the suffix-match length behind the most recent Draft.
func (d *NgramDrafter) LastDraftInfo() DraftInfo { return DraftInfo{MatchLen: d.lastMatch} }

// Draft implements Drafter via longest-suffix prompt lookup.
func (d *NgramDrafter) Draft(ctx []int, k int) []int {
	d.lastMatch = 0
	minM, maxM := d.MinMatch, d.MaxMatch
	if minM < 1 {
		minM = 2
	}
	if maxM < minM {
		maxM = 16
	}
	n := len(ctx)
	if n < minM+1 || k < 1 {
		return nil
	}
	hi := maxM
	if hi > n-1 { // need at least one earlier token to match against
		hi = n - 1
	}
	for L := hi; L >= minM; L-- {
		pat := ctx[n-L:]
		// Most recent earlier occurrence: the latest start s with s+L <= n-1.
		for s := n - L - 1; s >= 0; s-- {
			if !slices.Equal(ctx[s:s+L], pat) {
				continue
			}
			cont := ctx[s+L:] // tokens that followed this earlier occurrence
			m := min(k, len(cont))
			d.lastMatch = L
			return slices.Clone(cont[:m])
		}
	}
	return nil
}

// GenerateNgramSpeculative is single-model greedy speculative decoding driven by
// an n-gram (prompt-lookup) Drafter instead of a draft model. Each round the
// drafter proposes up to K tokens from the running context; the target verifies
// [cur, draft…] in one batched ForwardN, keeps the matching prefix, replaces the
// first mismatch with its own argmax, and rolls the cache back with TruncateTo.
//
// It is **lossless**: with Temperature==0 the output is token-identical to plain
// greedy (the target's argmax decides every position; TestNgramSpeculativeGreedyParity
// gates it); with Temperature>0 it reproduces the target sampler's distribution
// exactly via rejection sampling (point-mass q ⇒ accept x w.p. p(x), residual
// correction on reject — see spec_sample.go). On a drafter miss the round degenerates
// to one plain decode step. The win is amortizing the target's weight stream over
// (accepted+1) tokens whenever the context repeats.
//
// Sampled mode supports temperature + top-k/top-p/min-p; it rejects the
// history-dependent transforms (repetition/presence/frequency penalties, LogitBias,
// LogitProcessor), which the per-position verify does not yet thread. The returned
// Generation's Spec field carries acceptance telemetry.
func (target *Model) GenerateNgramSpeculative(ctx context.Context, prompt []int, maxTokens int, drafter Drafter, K int, sp SamplingParams) (<-chan int, *Generation, error) {
	return target.genNgram(ctx, prompt, maxTokens, drafter, K, sp, nil, nil)
}

// validateNgramSpec runs the synchronous precondition checks shared by the Model
// and Session speculative entry points.
func validateNgramSpec(drafter Drafter, sp SamplingParams) error {
	if drafter == nil {
		return fmt.Errorf("decoder.GenerateNgramSpeculative: nil drafter")
	}
	if sp.LogitProcessor != nil {
		return fmt.Errorf("decoder.GenerateNgramSpeculative: LogitProcessor (constrained/tool decoding) not supported yet; use Generate")
	}
	return nil
}

// genNgram is the async wrapper: it validates, launches genNgramInto in a
// goroutine over a fresh self-owned cache, and returns the stream. Session-backed
// callers use genNgramInto directly to thread their warm KV cache.
func (target *Model) genNgram(ctx context.Context, prompt []int, maxTokens int, drafter Drafter, K int, sp SamplingParams, tr specTracer, ad *AdaptiveDepth) (<-chan int, *Generation, error) {
	if err := validateNgramSpec(drafter, sp); err != nil {
		return nil, nil, err
	}
	if len(prompt) == 0 {
		return nil, nil, fmt.Errorf("decoder.GenerateNgramSpeculative: empty prompt")
	}
	out := make(chan int)
	stats := &SpecStats{}
	g := &Generation{Spec: stats}
	go func() {
		defer close(out)
		target.genNgramInto(ctx, out, g, stats, drafter, prompt, 0, maxTokens, K, sp, tr, ad, nil, nil)
	}()
	return out, g, nil
}

// genNgramInto is the synchronous speculative decode loop, mirroring generateInto:
// it assumes cache already holds prompt[:prefillFrom] (a Session's warm prefix; 0
// and a nil cache for a fresh run), prefills the rest, then runs the n-gram verify
// loop streaming each committed token to out. commit, if non-nil, is invoked with
// each token as its forward commits it to the cache (the Session seam — note the
// final pending token is emitted but not committed, exactly one behind the cache,
// so reconciliation matches the cache contents). It does NOT close out. When cache
// is non-nil the device-resident path is bypassed (sessions use the CPU cache, as
// generateInto does); tr enables §06 trace emission.
func (target *Model) genNgramInto(ctx context.Context, out chan<- int, g *Generation, stats *SpecStats, drafter Drafter, prompt []int, prefillFrom, maxTokens, K int, sp SamplingParams, tr specTracer, ad *AdaptiveDepth, cache *KVCache, commit func(int)) {
	if len(prompt) == 0 {
		g.err = fmt.Errorf("decoder.GenerateNgramSpeculative: empty prompt")
		return
	}
	if K < 1 {
		K = 4
	}
	sampled := sp.Temperature > 0
	var sampler *Sampler
	needHist := false
	if sampled {
		sampler = NewSampler(sp)
		needHist = sampler.needsHistory() // penalties / logit bias ⇒ thread per-position history
	}
	// dist is the target's next-token sampling distribution at one position. With a
	// history-dependent transform (penalties/bias) it is computed over `ph` (the
	// tokens before this position); otherwise the cheap history-free path. Returns
	// nil in greedy mode (callers use argmax there).
	dist := func(logits []float32, ph []int) []float64 {
		if needHist {
			return sampler.distVectorHist(logits, ph)
		}
		return sampler.distVector(logits)
	}

	// Resident verify on the device when the target's GPU-resident decode path is
	// built (webgpu + eligible arch) AND no session cache was handed in: prompt seeds
	// the resident KV via per-token Forward, the verify is a single batched ForwardN,
	// rollback is the resident no-op TruncateTo. There is no draft model to place —
	// the drafter is pure-Go — so this path adds no GPU memory. A session passes its
	// CPU cache, which forces the staged path (as generateInto does).
	resident := cache == nil && target.resident != nil && target.DecodeRunnerEligible()
	{
		tc := cache
		if tc == nil && !resident {
			tc = target.NewCache(len(prompt) + maxTokens + K + 8)
		}

		tpos := 0
		targetVerify := func(seq []int, base int) ([][]float32, error) {
			if resident {
				embs := make([][]float32, len(seq))
				for i, tok := range seq {
					embs[i] = target.embedResident(tok)
				}
				return target.resident.ForwardN(embs, base)
			}
			return target.forwardN(seq, tc)
		}
		targetTruncate := func(keep int) {
			tpos = keep
			if resident {
				target.resident.TruncateTo(keep)
			} else {
				tc.TruncateTo(keep)
			}
		}

		// Prefill the (divergent suffix of the) prompt; the last token's logits seed cur.
		var seedLogits []float32
		var err error
		if resident {
			for i, id := range prompt {
				if seedLogits, err = target.resident.Forward(target.embedResident(id), i); err != nil {
					g.err = err
					return
				}
			}
		} else {
			if seedLogits, err = target.prefillLogits(prompt[prefillFrom:], tc); err != nil {
				g.err = err
				return
			}
		}
		tpos = len(prompt)

		// hist is the confirmed context the drafter searches: the prompt plus every
		// committed token. cur is the next confirmed token (not yet in hist), exactly
		// like plain decode's pending-next-token invariant.
		hist := slices.Clone(prompt)
		// The seed token is the target's own first draw: argmax (greedy) or a sample
		// from the seed distribution (sampled) — exactly what plain decode emits first.
		cur := argmax(seedLogits)
		if sampled {
			cur = sampler.drawDist(dist(seedLogits, hist)) // seed history = prompt
		}

		emit := func(tok int) bool {
			if target.isStop(tok, sp) {
				return false
			}
			select {
			case <-ctx.Done():
				g.err = ctx.Err()
				return false
			case out <- tok:
			}
			stats.Emitted++
			return stats.Emitted < maxTokens
		}

		if !emit(cur) {
			return
		}
		for {
			// 1. Draft up to K tokens from the context ending at cur (zero on a miss).
			lookupCtx := append(slices.Clone(hist), cur)
			proposed := drafter.Draft(lookupCtx, K)
			// Fixed K verifies the whole proposal; the adaptive controller trims it to
			// the depth its running acceptance estimate still justifies (04).
			draftTok := proposed
			if ad != nil {
				draftTok = proposed[:ad.Depth(len(proposed))]
			}
			kEff := len(draftTok)
			matchLen := 0
			if tr != nil {
				if td, ok := drafter.(TracingDrafter); ok {
					matchLen = td.LastDraftInfo().MatchLen
				}
			}

			// 2. Verify: one target pass over [cur, draft…] gives the target's argmax
			// after each position in a single weight stream.
			base := tpos
			seq := append([]int{cur}, draftTok...)
			logitsN, err := targetVerify(seq, base)
			if err != nil {
				g.err = err
				return
			}

			// 3. Verify each draft position. Greedy: accept while draftTok[i] equals the
			// target's argmax, replacing the first mismatch with the target's token.
			// Sampled: accept draftTok[i] with probability p_i(draftTok[i]) (point-mass
			// q ⇒ accept rate p(x)); on reject, the correction is drawn from the residual
			// (p_i with draftTok[i] removed). Both are lossless; the first reject stops
			// the chain (positions past it are conditioned on a token we didn't take).
			stats.Rounds++
			stats.Drafted += kEff
			accepted := 0
			allAccept := true
			var nextTok int
			// ph is the penalty/bias history up to the current position: prompt +
			// committed + cur, growing by each accepted draft token (nil when no
			// history-dependent transform is active).
			var ph []int
			if needHist {
				ph = append(slices.Clone(hist), cur)
			}
			for i := 0; i < kEff; i++ {
				var acc bool
				if sampled {
					p := dist(logitsN[i], ph)
					var tok int
					tok, acc = sampler.specStep(p, draftTok[i])
					if tr != nil {
						tr(traceFromDist(stats.Rounds, i, matchLen, draftTok[i], p, acc))
					}
					if !acc {
						nextTok = tok
					}
					if needHist {
						ph = append(ph, draftTok[i]) // advance to the next position's history
					}
				} else {
					ti := argmax(logitsN[i])
					acc = draftTok[i] == ti
					if tr != nil {
						pTop1, pEnt, pTok := targetDist(logitsN[i], draftTok[i])
						tr(SpecTrace{
							Step: stats.Rounds, Pos: i, Source: "ngram", Token: draftTok[i],
							NgramMatch: matchLen, Streak: i, QTop1: 1,
							PTop1: pTop1, PEntropy: pEnt, TV: 1 - pTok, AcceptProb: pTok, Accepted: acc,
						})
					}
					if !acc {
						nextTok = ti
					}
				}
				if acc {
					accepted++
					continue
				}
				allAccept = false
				break
			}
			if allAccept {
				// Bonus token from the position after the last accepted draft — the
				// target's own draw (argmax / sample). Also covers the kEff==0 miss.
				// ph here is prompt+committed+cur+draft[:kEff] — the bonus position's
				// history.
				if sampled {
					nextTok = sampler.drawDist(dist(logitsN[kEff], ph))
				} else {
					nextTok = argmax(logitsN[kEff])
				}
			}
			stats.Accepted += accepted

			// Feed the realized outcome back to the depth controller: observed = the
			// positions actually checked (accepts plus the one rejection, if any).
			if ad != nil {
				observed := accepted
				if !allAccept {
					observed = accepted + 1
				}
				ad.Observe(accepted, observed)
			}

			// 4. Roll the cache back to the confirmed length: cur (at base) plus the
			// accepted draft tokens. nextTok stays pending for the next round.
			targetTruncate(base + 1 + accepted)

			// 5. Commit to history: cur, then the accepted draft tokens. nextTok is
			// the new cur and is committed at the top of the next round. The same
			// tokens (now resident in the cache at positions base..base+accepted) are
			// reported to the Session via commit, so its token list mirrors the cache.
			hist = append(hist, cur)
			if commit != nil {
				commit(cur)
			}
			for i := 0; i < accepted; i++ {
				hist = append(hist, draftTok[i])
				if commit != nil {
					commit(draftTok[i])
				}
			}

			// 6. Stream the accepted draft tokens, then the correction/bonus.
			for i := 0; i < accepted; i++ {
				if !emit(draftTok[i]) {
					return
				}
			}
			cur = nextTok
			if !emit(cur) {
				return
			}
		}
	}
}
