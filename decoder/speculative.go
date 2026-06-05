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

	out := make(chan int)
	stats := &SpecStats{}
	g := &Generation{Spec: stats}
	go func() {
		defer close(out)
		room := len(prompt) + maxTokens + K + 8
		tc := target.NewCache(room)
		dc := draft.NewCache(room)

		// Prefill the prompt into both caches. Only the target's last-token logits
		// are needed (to seed the first generated token, cur); the draft just needs
		// the KV, so runLayers (skips its LM head).
		for _, id := range prompt[:len(prompt)-1] {
			if _, err := target.runLayers(id, tc); err != nil {
				g.err = err
				return
			}
			if _, err := draft.runLayers(id, dc); err != nil {
				g.err = err
				return
			}
		}
		last := prompt[len(prompt)-1]
		seedLogits, err := target.forward(last, tc)
		if err != nil {
			g.err = err
			return
		}
		if _, err := draft.runLayers(last, dc); err != nil {
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
				dl, err := draft.forward(d, dc)
				if err != nil {
					g.err = err
					return
				}
				draftTok[i] = argmax(dl)
				d = draftTok[i]
			}

			// 2. Verify: one target pass over [cur, draftTok...] gives the target's
			// argmax after each, in one weight stream (the speculative win).
			base := tc.Pos() // cur will be appended at this position
			seq := make([]int, 0, K+1)
			seq = append(seq, cur)
			seq = append(seq, draftTok...)
			logitsN, err := target.forwardN(seq, tc)
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
			tc.TruncateTo(keep)
			if allAccept {
				// The draft only cached cur..draftTok[K-2]; on a full accept,
				// draftTok[K-1] is confirmed too, so feed it once to sync the draft
				// cache before trimming (a no-op trim here).
				if _, err := draft.forward(draftTok[K-1], dc); err != nil {
					g.err = err
					return
				}
			}
			dc.TruncateTo(keep)

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
