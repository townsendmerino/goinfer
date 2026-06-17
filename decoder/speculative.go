package decoder

import (
	"context"
	"fmt"
)

// SpecStats accumulates speculative-decoding telemetry for one Generate run.
type SpecStats struct {
	Rounds   int // verification passes (one expensive target forwardN each)
	Drafted  int // draft tokens proposed (== Rounds*K)
	Accepted int // draft tokens the target confirmed (excludes the per-round correction/bonus)
	Emitted  int // tokens streamed to the caller
}

// AcceptanceRate is accepted draft tokens / proposed — the lever on the speedup.
func (s *SpecStats) AcceptanceRate() float64 {
	if s.Drafted == 0 {
		return 0
	}
	return float64(s.Accepted) / float64(s.Drafted)
}

// TokensPerRound is the mean tokens confirmed per target pass (accepted + 1);
// the speedup over plain target decode is ≈ this / (1 + K·draft_cost_ratio).
func (s *SpecStats) TokensPerRound() float64 {
	if s.Rounds == 0 {
		return 0
	}
	return float64(s.Emitted) / float64(s.Rounds)
}

// GenerateSpeculative runs greedy speculative decoding: the small draft model
// proposes K tokens autoregressively, and this (target) model verifies them in a
// single batched forward pass, keeping the matching prefix and replacing the
// first mismatch with its own token. The streamed output is **token-identical to
// plain target greedy** (TestSpeculativeGreedyParity is the gate) — a pure
// speedup, not an approximation. The win comes from amortizing the target's
// expensive weight stream over (accepted+1) tokens per pass instead of 1.
//
// Greedy only: Temperature must be 0 (sampled speculative needs the
// rejection-sampling residual rule — a follow-up). draft and target must share
// the exact vocab/tokenizer; this is asserted. The returned Generation's Spec
// field carries acceptance telemetry.
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
	if dv, tv := draft.w.arch.VocabSize, target.w.arch.VocabSize; dv != tv {
		return nil, nil, fmt.Errorf("decoder.GenerateSpeculative: draft/target vocab mismatch (%d vs %d) — they must share a tokenizer", dv, tv)
	}
	if len(prompt) == 0 {
		return nil, nil, fmt.Errorf("decoder.GenerateSpeculative: empty prompt")
	}
	if K < 1 {
		K = 4
	}

	// When the target's GPU-resident decode path is built (webgpu + eligible arch),
	// run its verify on the device: the prompt seeds the resident KV via per-token
	// Forward, the K+1-token verify is a single batched ForwardN (one Submit/Poll —
	// the speculative win amortizes the cgo-encode glue over K), and the rollback is
	// the resident no-op TruncateTo. The draft stays CPU (small). Output is still
	// token-identical to plain greedy (TestSpeculativeResident_parity).
	resident := target.resident != nil && target.DecodeRunnerEligible()
	// The draft runs on the GPU too when it was loaded --backend webgpu and is
	// eligible. This is the decisive lever: a CPU draft is slower PER TOKEN than the
	// GPU target it feeds (measured ~4.9× on the 2070S), so K CPU draft tokens/round
	// cost far more than they save and speculation is a net loss; a resident draft is
	// ~0.5× the target's per-token cost, which is the only regime where it can pay.
	// Draft + target are separate Contexts (one device/queue each), driven
	// sequentially here. Output stays token-identical to plain target greedy — the
	// draft is only a proposer; the target decides (so its backend can't change it).
	draftResident := draft.resident != nil && draft.DecodeRunnerEligible()

	out := make(chan int)
	stats := &SpecStats{}
	g := &Generation{Spec: stats}
	go func() {
		defer close(out)
		room := len(prompt) + maxTokens + K + 8
		var tc *KVCache // target CPU cache (nil on the resident path)
		if !resident {
			tc = target.NewCache(room)
		}
		var dc *KVCache // draft CPU cache (nil on the resident-draft path)
		if !draftResident {
			dc = draft.NewCache(room)
		}

		// dpos tracks the resident draft's next write position (mirrors dc.Pos() on the
		// CPU path). draftForward/draftPrefill/draftTruncate route to the resident or CPU
		// draft uniformly; the resident draft is positional (TruncateTo is a no-op, stale
		// speculative KV past `keep` is overwritten next round), exactly like the target.
		dpos := 0
		draftForward := func(tok int) ([]float32, error) {
			if draftResident {
				l, err := draft.resident.Forward(draft.embedResident(tok), dpos)
				dpos++
				return l, err
			}
			return draft.forward(tok, dc)
		}
		draftPrefill := func() error {
			if draftResident {
				for i, id := range prompt {
					if _, err := draft.resident.Forward(draft.embedResident(id), i); err != nil {
						return err
					}
				}
				dpos = len(prompt)
				return nil
			}
			_, err := draft.prefillLogits(prompt, dc)
			return err
		}
		draftTruncate := func(keep int) {
			if draftResident {
				dpos = keep
				draft.resident.TruncateTo(keep)
			} else {
				dc.TruncateTo(keep)
			}
		}

		// targetVerify runs one target pass over seq (= [cur, drafts...]) at absolute
		// position base, returning a logit row after each. tpos tracks the target's
		// confirmed position (== tc.Pos() on the CPU path). targetTruncate rolls the
		// confirmed length back after an accept.
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

		// Prefill the prompt. The target's last-token logits seed cur; the draft's are
		// discarded (it just needs its KV filled). Resident: seed the GPU KV with
		// O(prompt) Forwards (mirrors resident Generate). CPU: batched prefillLogits.
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
			seedLogits, err = target.prefillLogits(prompt, tc)
			if err != nil {
				g.err = err
				return
			}
		}
		tpos = len(prompt)
		if err := draftPrefill(); err != nil {
			g.err = err
			return
		}
		// cur is the next confirmed token to feed; it is NOT yet in either cache
		// (matching plain decode's "pending next token" invariant).
		cur := argmax(seedLogits)

		// emit streams a confirmed token; returns false when generation should end
		// (EOS/stop, maxTokens, or ctx cancel — the latter sets g.err).
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
			// 1. Draft K tokens greedily on the draft model.
			draftTok := make([]int, K)
			d := cur
			for i := 0; i < K; i++ {
				dl, err := draftForward(d)
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
			seq := make([]int, 0, K+1)
			seq = append(seq, cur)
			seq = append(seq, draftTok...)
			logitsN, err := targetVerify(seq, base)
			if err != nil {
				g.err = err
				return
			}

			// 3. Greedy accept: keep draftTok[i] while it equals the target's argmax
			// at position i; the first mismatch is replaced by the target's token.
			stats.Rounds++
			stats.Drafted += K
			accepted := 0
			allAccept := true
			var nextTok int
			for i := 0; i < K; i++ {
				ti := argmax(logitsN[i])
				if draftTok[i] == ti {
					accepted++
					continue
				}
				nextTok = ti // correction (free, from the same pass)
				allAccept = false
				break
			}
			if allAccept {
				nextTok = argmax(logitsN[K]) // bonus token
			}
			stats.Accepted += accepted

			// 4. Roll the caches back to the confirmed length: cur (at base) plus the
			// accepted draft tokens. nextTok stays pending (fed next round), exactly
			// as plain decode would carry it.
			keep := base + 1 + accepted
			targetTruncate(keep)
			if allAccept {
				// The draft only cached cur..draftTok[K-2]; on a full accept,
				// draftTok[K-1] is confirmed too, so feed it once to sync the draft
				// cache before trimming (a no-op trim here).
				if _, err := draftForward(draftTok[K-1]); err != nil {
					g.err = err
					return
				}
			}
			draftTruncate(keep)

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
	}()
	return out, g, nil
}
