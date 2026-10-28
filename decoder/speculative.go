package decoder

import (
	"context"
	"fmt"
	"sync/atomic"
)

// SpecStats accumulates CPU speculative-decoding telemetry for one Generate run. The GPU backend's batched-verify
// counters are a distinct type, cuda.GPUSpecStats, because the fields do not map 1:1 (this Evaluated stops at the
// first reject; cuda.VerifyToks counts every position fed to the batched verify).
type SpecStats struct {
	Rounds    int // verification passes (one expensive target forwardN each)
	Drafted   int // draft tokens proposed (== Rounds*K)
	Evaluated int // draft positions actually verified (accepts + the one rejection/round); ≤ Drafted because the chain stops at the first reject, leaving the proposed tail untested
	Accepted  int // draft tokens the target confirmed (excludes the per-round correction/bonus)
	Emitted   int // tokens streamed to the caller
}

// AcceptanceRate is accepted draft tokens / proposed — the lever on the speedup.
func (s *SpecStats) AcceptanceRate() float64 {
	if s.Drafted == 0 {
		return 0
	}
	return float64(s.Accepted) / float64(s.Drafted)
}

// EvalAcceptanceRate is accepted / evaluated — the realized per-position accept
// rate over positions actually tested, which (unlike AcceptanceRate) is directly
// comparable to α̅ = mean accept_prob: it excludes the untested post-reject tail
// that AcceptanceRate's denominator includes. The gap between this and
// AcceptanceRate is exactly the chain-breaking waste (large when rejects come
// early — e.g. sampled decode breaking long copy runs).
func (s *SpecStats) EvalAcceptanceRate() float64 {
	if s.Evaluated == 0 {
		return 0
	}
	return float64(s.Accepted) / float64(s.Evaluated)
}

// TokensPerRound is the mean tokens confirmed per target pass (accepted + 1);
// the speedup over plain target decode is ≈ this / (1 + K·draft_cost_ratio).
func (s *SpecStats) TokensPerRound() float64 {
	if s.Rounds == 0 {
		return 0
	}
	return float64(s.Emitted) / float64(s.Rounds)
}

// GenerateSpeculative runs greedy speculative decoding: the small draft model proposes K tokens autoregressively, and
// this (target) model verifies them in one batched forward pass, keeping the matching prefix and replacing the first
// mismatch with its own token. The streamed output is token-identical to plain target greedy
// (TestDraftSpecLoop_exactOnTheFake; on a checkpoint, TestSpeculativeGreedyParity): a speedup, not an approximation.
//
// Greedy only: Temperature must be 0 (sampled speculation needs the rejection-sampling residual rule, which this path
// does not implement). LogitProcessor and penalties are refused. draft and target must share the exact vocab and
// tokenizer, which is asserted. The returned Generation's Spec field carries acceptance telemetry.
func (target *Model) GenerateSpeculative(ctx context.Context, prompt []int, maxTokens int, draft *Model, K int, sp SamplingParams) (<-chan int, *Generation, error) {
	if draft == nil {
		return nil, nil, fmt.Errorf("decoder.GenerateSpeculative: nil draft model")
	}
	if sp.Temperature != 0 {
		return nil, nil, fmt.Errorf("decoder.GenerateSpeculative: greedy only (Temperature must be 0); use Generate for sampling")
	}
	if sp.LogitProcessor != nil {
		return nil, nil, fmt.Errorf("decoder.GenerateSpeculative: LogitProcessor (constrained decoding) not supported yet; use Generate")
	}
	// Greedy verify argmaxes raw target logits; penalties or logit bias would be silently dropped, diverging from plain
	// greedy.
	if sp.HistoryDependent() {
		return nil, nil, fmt.Errorf("decoder.GenerateSpeculative: repetition penalties / logit bias are not supported in greedy speculative decoding; use Generate")
	}
	if dv, tv := draft.w.arch.VocabSize, target.w.arch.VocabSize; dv != tv {
		return nil, nil, fmt.Errorf("decoder.GenerateSpeculative: draft/target vocab mismatch (%d vs %d) — they must share a tokenizer", dv, tv)
	}
	if len(prompt) == 0 {
		return nil, nil, fmt.Errorf("decoder.GenerateSpeculative: empty prompt")
	}
	if K < 1 {
		K = 4
	}
	// Rollback safety: verify advances the target's KV by K per round and rolls back the rejected tail, which a recurrent
	// (Mamba-2 / Gated DeltaNet) or staged sliding-window cache cannot losslessly restore. Every speculative entry point
	// guards this.
	if !target.specRollbackSafe() {
		return nil, nil, fmt.Errorf("decoder.GenerateSpeculative: this model has recurrent state (Mamba-2 / Gated DeltaNet) or a staged sliding-window ring cache that speculative rollback cannot losslessly restore; use Generate")
	}
	// Verify runs on the resident's BATCHED path while the prompt is seeded through M=1 decode; if
	// the resident says those two can disagree, the lossless claim above is false. See SpecDecodeConflict.
	if err := target.SpecDecodeConflict(); err != nil {
		return nil, nil, fmt.Errorf("decoder.GenerateSpeculative: %w; use Generate", err)
	}

	// When the target's GPU-resident decode path is built, verify runs on the device: the prompt seeds the resident KV via
	// per-token Forward, the K+1-token verify is a single batched ForwardN, and rollback is the resident no-op TruncateTo.
	// The output is token-identical to plain greedy (TestSpeculativeResident_parity).
	resident := target.resident != nil && target.DecodeRunnerEligible()
	// The draft runs on the GPU too when it was loaded with a GPU backend and is eligible. A CPU draft is slower per token
	// than the GPU target it feeds, so K CPU draft tokens per round cost more than they save and speculation is a net loss;
	// a resident draft is the only regime where it can pay. Draft and target are separate Contexts (one device and queue
	// each), driven sequentially here. The draft is only a proposer and the target decides, so the draft's backend cannot
	// change the output.
	draftResident := draft.resident != nil && draft.DecodeRunnerEligible()

	out := make(chan int)
	stats := &SpecStats{}
	g := &Generation{Spec: stats}
	// Held so decode and verify cannot use different attention trees. Not load-bearing for THIS loop today (it seeds and verifies in batches and takes no M=1
	// step, so its rows are exact whether or not the scope is held; TestFlashDecodeTwoModelSpecLane's mutant survives) — kept as defense in depth. The n-gram loop is
	// the one that needs it (TestFlashDecodeSpeculativeScope kills its mutant).
	leaveExact := target.enterExactAttention() // decode and verify must share one attention tree
	target.runGen(ctx, out, g, leaveExact, func(ctx context.Context) {
		defer leaveExact()
		// Claim the single shared resident KV before any device write: a second concurrent Generate on the same *Model would
		// otherwise prefill into the same positional device KV, interleaving writes and corrupting both streams, against a
		// Model doc that promises concurrent distinct sequences. On loss, fall back to the staged CPU cache. The draft is a
		// separate Model with its own claim. tryClaimResident, not a bare CAS on resBusy: MC3's batched holders never set
		// resBusy, so only the claim that also requires no holder keeps this off a resident mid-batch
		// (docs/completed/task-audit-followups-2026-10-06.md, A1).
		if resident {
			if target.tryClaimResident() {
				defer atomic.StoreInt32(&target.resBusy, 0)
				// Forget first: from here until this generation completes, the resident KV is mid-write, so the next turn must
				// cold-prefill rather than trust a half-written cache (resident_reuse.go). This path always seeds from position 0, so
				// there is nothing to preserve by deferring it.
				target.residentForgetIDs()
			} else {
				resident = false
			}
		}
		if draftResident {
			if draft.tryClaimResident() {
				defer atomic.StoreInt32(&draft.resBusy, 0)
				// Same as the target's claim above, for the draft model: until this generation completes its resident KV is
				// mid-write, so a later generation on the same draft Model must not trust a stale resIDs.
				draft.residentForgetIDs()
			} else {
				draftResident = false
			}
		}
		room := len(prompt) + maxTokens + K + 8
		var tf, df Forward
		if resident {
			tf = &residentForward{m: target}
		} else {
			tf = cpuForward{target, target.NewCache(room)}
		}
		if draftResident {
			df = &residentForward{m: draft}
		} else {
			df = cpuForward{draft, draft.NewCache(room)}
		}
		isStop := func(tok int) bool { return target.isStop(tok, sp) }
		draftSpecLoop(ctx, tf, df, isStop, target.ResidentContextCap, out, g, stats, prompt, maxTokens, K)
	})
	return out, g, nil
}

// draftSpecLoop is greedy speculative decoding with a draft model: each round the draft (df) proposes up to K tokens one
// Step at a time, the target (tf) verifies them in one Verify, the longest prefix matching the target's argmax is kept
// with the correction or bonus token, and both are truncated back to it. It emits exactly the target's greedy stream
// (TestDraftSpecLoop_exactOnTheFake). ctxCap is the target's context cap, read each round as specRoundDraftWidth needs.
func draftSpecLoop(ctx context.Context, tf, df Forward, isStop func(int) bool, ctxCap func() int, out chan<- int, g *Generation, stats *SpecStats, prompt []int, maxTokens, K int) {
	seedLogits, err := tf.Prefill(ctx, prompt)
	if err != nil {
		g.err = err
		return
	}
	tpos := len(prompt)
	if _, err := df.Prefill(ctx, prompt); err != nil {
		g.err = err
		return
	}
	// cur is the next confirmed token to feed; it is NOT yet in either cache
	// (matching plain decode's "pending next token" invariant).
	cur := argmax(seedLogits)

	// emit streams a confirmed token; returns false when generation should end
	// (EOS/stop, maxTokens, or ctx cancel — the latter sets g.err).
	emit := func(tok int) bool {
		if isStop(tok) {
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
		// Clamp this round's fixed K to what is left under the resident context cap, fresh each round (specRoundDraftWidth).
		// A -1 means no room even for cur: stop cleanly rather than attempt a verify round the backend's checkCap would refuse
		// with a hard error.
		kRound := specRoundDraftWidth(K, tpos, ctxCap())
		if kRound < 0 {
			return
		}

		// 1. Draft kRound tokens greedily on the draft model (kRound may be 0 at the cap
		// boundary: no draft, just a plain verified step on cur).
		draftTok := make([]int, kRound)
		d := cur
		for i := range kRound {
			dl, err := df.Step(ctx, d)
			if err != nil {
				g.err = err
				return
			}
			draftTok[i] = argmax(dl)
			d = draftTok[i]
		}

		// 2. Verify: one target pass over [cur, draftTok...] gives the target's
		// argmax after each, in one weight stream (the speculative win).
		base := tpos // cur will be appended at this position
		seq := make([]int, 0, kRound+1)
		seq = append(seq, cur)
		seq = append(seq, draftTok...)
		logitsN, err := tf.Verify(ctx, seq)
		if err != nil {
			g.err = err
			return
		}

		// 3. Greedy accept: keep draftTok[i] while it equals the target's argmax
		// at position i; the first mismatch is replaced by the target's token.
		stats.Rounds++
		stats.Drafted += kRound
		accepted := 0
		allAccept := true
		var nextTok int
		for i := range kRound {
			ti := argmax(logitsN[i])
			if draftTok[i] == ti {
				accepted++
				continue
			}
			nextTok = ti // correction (free, from the same pass)
			allAccept = false
			break
		}
		stats.Evaluated += accepted // positions actually verified: the accepts...
		if !allAccept {
			stats.Evaluated++ // ...plus the one rejection this round (EvalAcceptanceRate)
		}
		if allAccept {
			nextTok = argmax(logitsN[kRound]) // bonus token (kRound==0: cur's own verified next)
		}
		stats.Accepted += accepted

		// 4. Roll the caches back to the confirmed length: cur (at base) plus the
		// accepted draft tokens. nextTok stays pending (fed next round), exactly
		// as plain decode would carry it.
		keep := base + 1 + accepted
		tpos = keep
		tf.TruncateTo(keep)
		switch {
		case allAccept && kRound > 0:
			// The draft only cached cur..draftTok[kRound-2]; on a full accept,
			// draftTok[kRound-1] is confirmed too, so feed it once to sync the draft
			// cache before trimming (a no-op trim here).
			if _, err := df.Step(ctx, draftTok[kRound-1]); err != nil {
				g.err = err
				return
			}
		case allAccept:
			// kRound == 0: no draft ran this round, so cur itself was never fed to the draft's cache (the loop above feeds from
			// cur only when kRound >= 1). Feed it now so the draft cache stays in sync with the target's confirmed position before
			// next round's draft resumes from nextTok.
			if _, err := df.Step(ctx, cur); err != nil {
				g.err = err
				return
			}
		}
		df.TruncateTo(keep)

		// 5. Stream the accepted draft tokens, then the correction/bonus, which
		// becomes the next cur. Stop mid-stream on EOS/stop/maxTokens/cancel.
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
