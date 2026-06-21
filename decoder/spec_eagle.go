package decoder

import (
	"context"
	"fmt"
	"slices"
)

// GenerateEagleSpeculative is end-to-end lossless speculative decode with an EAGLE-3
// draft head (05): the head drafts K tokens from the target's fused hidden, the target
// verifies them in one batched pass, and the matching prefix (plus the target's own
// correction) is committed — token-identical to plain greedy (the target's argmax
// decides every position; TestEagleSpecParity gates it). The head re-seeds each round
// from the target hidden captured during the verify, and its KV is rebuilt over the
// confirmed context so its attention has full history.
//
// First cut: greedy + CPU, generic-arch target (qwen3 dense), head.Hidden() ==
// target HiddenDim. The win is on novel text (where n-gram/grammar can't draft);
// acceptance is the head's (~1.6 tok/verify measured), so the speedup is modest and
// backend-dependent. capLayers selects the 3 fused target layers (e.g. {2,L/2,L-3}).
func (m *Model) GenerateEagleSpeculative(ctx context.Context, prompt []int, maxTokens int, head *EagleHead, capLayers []int, K int, sp SamplingParams) (<-chan int, *Generation, error) {
	if head == nil {
		return nil, nil, fmt.Errorf("decoder.GenerateEagleSpeculative: nil head")
	}
	if sp.Temperature != 0 || sp.LogitProcessor != nil {
		return nil, nil, fmt.Errorf("decoder.GenerateEagleSpeculative: greedy only (no temperature/LogitProcessor)")
	}
	if !m.specRollbackSafe() {
		return nil, nil, fmt.Errorf("decoder.GenerateEagleSpeculative: recurrent family unsupported")
	}
	if head.Hidden() != m.w.arch.HiddenDim {
		return nil, nil, fmt.Errorf("decoder.GenerateEagleSpeculative: head hidden %d != target %d", head.Hidden(), m.w.arch.HiddenDim)
	}
	if len(prompt) < 2 {
		return nil, nil, fmt.Errorf("decoder.GenerateEagleSpeculative: prompt too short")
	}
	if K < 1 {
		K = 5
	}
	hidden := m.w.arch.HiddenDim
	embedOf := func(tok int, dst []float32) { m.embedToken(tok, dst) }

	out := make(chan int)
	stats := &SpecStats{}
	g := &Generation{Spec: stats}
	go func() {
		defer close(out)
		tc := m.NewCache(len(prompt) + maxTokens + K + 8)

		// fuseAt fuses the 3 captured target hidden states at batch row i into a feature.
		fuseAt := func(i int) []float32 {
			h3 := make([]float32, 0, 3*hidden)
			for ci := range capLayers {
				row := tc.captured[ci][i*hidden : (i+1)*hidden]
				h3 = append(h3, row...)
			}
			return head.Fuse(m.be, h3)
		}
		// captureN runs forwardN over ids with the seam on, returning per-position logits
		// and per-position fused features.
		captureN := func(ids []int) (logits [][]float32, feats [][]float32, err error) {
			tc.captureLayers = capLayers
			tc.captured = make([][]float32, len(capLayers))
			defer func() { tc.captureLayers, tc.captured = nil, nil }()
			lg, e := m.forwardN(ids, tc)
			if e != nil {
				return nil, nil, e
			}
			feats = make([][]float32, len(ids))
			for i := range ids {
				feats[i] = fuseAt(i)
			}
			return lg, feats, nil
		}

		// Prefill: one batched pass over the prompt, capturing every position's feature.
		logitsN, feats, err := captureN(prompt)
		if err != nil {
			g.err = err
			return
		}
		confirmed := slices.Clone(prompt)
		curLogits := slices.Clone(logitsN[len(prompt)-1]) // base dist after the last confirmed token

		emit := func(tok int) bool {
			if m.isStop(tok, sp) {
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

		for {
			C := len(confirmed)
			// Head drafts K tokens: rebuild its KV over confirmed[:C-1], then draft from
			// the last confirmed token + its feature. (O(C) head steps — head is tiny.)
			hst := head.Prefill(m.be, embedOf, confirmed[:C-1], feats[:C-1], 0)
			draft := head.DraftFrom(m.be, hst, embedOf, confirmed[C-1], feats[C-1], C-1, K)

			// Verify the draft in one batched target pass at positions C..C+K-1.
			dlogits, dfeats, err := captureN(draft)
			if err != nil {
				g.err = err
				return
			}
			stats.Rounds++
			stats.Drafted += K
			acc := 0
			prev := curLogits
			for i := 0; i < K; i++ {
				if draft[i] != argmax(prev) {
					break
				}
				acc++
				prev = dlogits[i]
			}
			stats.Accepted += acc
			stats.Evaluated += acc
			if acc < K {
				stats.Evaluated++
			}
			correction := argmax(prev) // base's own token at position C+acc (greedy)

			// Commit: keep the accepted prefix in the cache; append accepted feats.
			tc.TruncateTo(C + acc)
			for i := 0; i < acc; i++ {
				confirmed = append(confirmed, draft[i])
				feats = append(feats, dfeats[i])
			}
			for i := 0; i < acc; i++ {
				if !emit(draft[i]) {
					return
				}
			}
			// The correction's hidden wasn't captured (it differs from the rejected
			// draft), so forward it once to get its feature + next-token logits.
			cl, cfeat, err := captureN([]int{correction})
			if err != nil {
				g.err = err
				return
			}
			confirmed = append(confirmed, correction)
			feats = append(feats, cfeat[0])
			curLogits = slices.Clone(cl[0])
			if !emit(correction) {
				return
			}
		}
	}()
	return out, g, nil
}
