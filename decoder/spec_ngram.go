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
// Output is **token-identical to plain target greedy** (TestNgramSpeculativeGreedyParity
// is the gate) — losslessly so, because the target's argmax decides every position
// and the drafter only proposes. On a drafter miss the round degenerates to one
// plain decode step (no speculation, still correct). The win is amortizing the
// target's weight stream over (accepted+1) tokens whenever the context repeats.
//
// Greedy only for now (Temperature must be 0): sampled lossless speculation needs
// the rejection-sampling residual rule, which is the next foundation task. The
// returned Generation's Spec field carries acceptance telemetry.
func (target *Model) GenerateNgramSpeculative(ctx context.Context, prompt []int, maxTokens int, drafter Drafter, K int, sp SamplingParams) (<-chan int, *Generation, error) {
	return target.genNgram(ctx, prompt, maxTokens, drafter, K, sp, nil, nil)
}

// genNgram is the body behind GenerateNgramSpeculative, plus an optional per-
// position trace sink. When tr is non-nil it computes the target softmax at each
// verified draft position to emit the exact per-position acceptance
// (accept_prob = 1 − TV = p(token) for the n-gram's point-mass q) and diagnostic
// features — the §06 SpecTrace dataset. That softmax is extra work, so tr stays
// nil on the production path (the public wrapper) and is set only by the harness.
func (target *Model) genNgram(ctx context.Context, prompt []int, maxTokens int, drafter Drafter, K int, sp SamplingParams, tr specTracer, ad *AdaptiveDepth) (<-chan int, *Generation, error) {
	if drafter == nil {
		return nil, nil, fmt.Errorf("decoder.GenerateNgramSpeculative: nil drafter")
	}
	if sp.Temperature != 0 {
		return nil, nil, fmt.Errorf("decoder.GenerateNgramSpeculative: greedy only (Temperature must be 0); use Generate for sampling")
	}
	if sp.LogitProcessor != nil {
		return nil, nil, fmt.Errorf("decoder.GenerateNgramSpeculative: LogitProcessor (constrained decoding) not supported yet; use Generate")
	}
	if len(prompt) == 0 {
		return nil, nil, fmt.Errorf("decoder.GenerateNgramSpeculative: empty prompt")
	}
	if K < 1 {
		K = 4
	}

	// Resident verify on the device when the target's GPU-resident decode path is
	// built (webgpu + eligible arch): prompt seeds the resident KV via per-token
	// Forward, the verify is a single batched ForwardN, rollback is the resident
	// no-op TruncateTo. There is no draft model to place — the drafter is pure-Go —
	// so this path adds no GPU memory, unlike GenerateSpeculative's draft model.
	resident := target.resident != nil && target.DecodeRunnerEligible()

	out := make(chan int)
	stats := &SpecStats{}
	g := &Generation{Spec: stats}
	go func() {
		defer close(out)
		var tc *KVCache
		if !resident {
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

		// Prefill the prompt; the last token's logits seed cur.
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
			if seedLogits, err = target.prefillLogits(prompt, tc); err != nil {
				g.err = err
				return
			}
		}
		tpos = len(prompt)

		// hist is the confirmed context the drafter searches: the prompt plus every
		// committed token. cur is the next confirmed token (not yet in hist), exactly
		// like plain decode's pending-next-token invariant.
		hist := slices.Clone(prompt)
		cur := argmax(seedLogits)

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

			// 3. Greedy accept: keep draftTok[i] while it equals the target's argmax;
			// the first mismatch is replaced by the target's own token.
			stats.Rounds++
			stats.Drafted += kEff
			accepted := 0
			allAccept := true
			var nextTok int
			for i := 0; i < kEff; i++ {
				ti := argmax(logitsN[i])
				acc := draftTok[i] == ti
				if tr != nil {
					pTop1, pEnt, pTok := targetDist(logitsN[i], draftTok[i])
					tr(SpecTrace{
						Step: stats.Rounds, Pos: i, Source: "ngram", Token: draftTok[i],
						NgramMatch: matchLen, Streak: i, QTop1: 1,
						PTop1: pTop1, PEntropy: pEnt, TV: 1 - pTok, AcceptProb: pTok, Accepted: acc,
					})
				}
				if acc {
					accepted++
					continue
				}
				nextTok = ti
				allAccept = false
				break
			}
			if allAccept {
				nextTok = argmax(logitsN[kEff]) // bonus token (also covers the kEff==0 miss)
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
			// the new cur and is committed at the top of the next round.
			hist = append(hist, cur)
			hist = append(hist, draftTok[:accepted]...)

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
	}()
	return out, g, nil
}
