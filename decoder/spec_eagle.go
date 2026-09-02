package decoder

import (
	"context"
	"fmt"
	"slices"
)

// GenerateEagleSpeculativeTree is the tree-drafting variant of GenerateEagleSpeculative
// (05): the head drafts a full B-ary tree — every node expands its top-B children at each
// of the D depths, so a round has Σ_{i=1}^{D} B^i nodes (e.g. B=2,D=4 ⇒ 30, not B*D=8) —
// the target verifies them all in ONE batched pass under tree attention, and the longest
// base-greedy-matching root-to-leaf path is committed. Still lossless (the base's argmax
// decides every token); the win over the linear chain is recovering positions where the
// correct token is in the head's top-B but not top-1, and continuing from the TRUE token.
// Wider/deeper trees cover more but cost B^D-ish verify work — tune B, D to the workload.
func (m *Model) GenerateEagleSpeculativeTree(ctx context.Context, prompt []int, maxTokens int, head *EagleHead, capLayers []int, B, D int, sp SamplingParams) (<-chan int, *Generation, error) {
	// Not generateInto's resident path: this entry point may drive the resident KV on its
	// own schedule, so the recorded id list stops being true the moment it does. Forget it —
	// the next turn cold-prefills, which is slow, not wrong (resident_reuse.go).
	m.residentForgetIDs()
	if head == nil {
		return nil, nil, fmt.Errorf("decoder.GenerateEagleSpeculativeTree: nil head")
	}
	if sp.Temperature != 0 || sp.LogitProcessor != nil {
		return nil, nil, fmt.Errorf("decoder.GenerateEagleSpeculativeTree: greedy only")
	}
	if sp.HistoryDependent() { // argmax verify can't apply penalties / logit bias (M13)
		return nil, nil, fmt.Errorf("decoder.GenerateEagleSpeculativeTree: repetition penalties / logit bias not supported in greedy speculative decoding; use Generate")
	}
	if !m.specRollbackSafe() {
		return nil, nil, fmt.Errorf("decoder.GenerateEagleSpeculativeTree: recurrent family or staged sliding-window unsupported (rollback cannot restore)")
	}
	if head.Hidden() != m.w.arch.HiddenDim {
		return nil, nil, fmt.Errorf("decoder.GenerateEagleSpeculativeTree: head hidden %d != target %d", head.Hidden(), m.w.arch.HiddenDim)
	}
	if len(prompt) < 2 {
		return nil, nil, fmt.Errorf("decoder.GenerateEagleSpeculativeTree: prompt too short")
	}
	if B < 1 {
		B = 2
	}
	if D < 1 {
		D = 4
	}
	// The EAGLE loop batches the target forward (forwardN) with the hidden-state capture seam
	// on. specRollbackSafe alone is insufficient: an own-runLayers family (gemma4/qwen35/
	// granite/nemotron/mla/llama4/gptoss) can pass it yet never populate cache.captured, so
	// fuseAt would slice a nil buffer — and even where the sequential path captures, forwardN
	// can't batch it. canBatchN gates exactly the batched-capture-safe arches (a superset of
	// ForwardCapture's allow-list); refuse the rest so the caller falls back to plain decode (M-07).
	if !m.canBatchN(eagleTreeNodes(B, D)) {
		return nil, nil, fmt.Errorf("decoder.GenerateEagleSpeculativeTree: batched hidden-state capture not supported for arch %q (own runLayers)", m.w.arch.Name)
	}
	hidden := m.w.arch.HiddenDim
	embedOf := func(tok int, dst []float32) { m.embedToken(tok, dst) }

	out := make(chan int)
	stats := &SpecStats{}
	g := &Generation{Spec: stats}
	go func() {
		defer close(out)
		tc := m.NewCache(len(prompt) + maxTokens + eagleTreeNodes(B, D) + 8) // a round writes the whole tree before rollback (M15)
		fuseAt := func(i int) []float32 {
			h3 := make([]float32, 0, 3*hidden)
			for ci := range capLayers {
				row := tc.captured[ci][i*hidden : (i+1)*hidden]
				h3 = append(h3, row...)
			}
			return head.Fuse(m.be, h3)
		}
		captureN := func(ids []int, fastAttn bool) (logits [][]float32, feats [][]float32, err error) {
			tc.captureLayers = capLayers
			tc.captured = make([][]float32, len(capLayers))
			defer func() { tc.captureLayers, tc.captured = nil, nil }()
			lg, e := m.forwardNAttn(ctx, ids, tc, fastAttn)
			if e != nil {
				return nil, nil, e
			}
			feats = make([][]float32, len(ids))
			for i := range ids {
				feats[i] = fuseAt(i)
			}
			return lg, feats, nil
		}
		logitsN, feats, err := captureN(prompt, cpuFastAttention())
		if err != nil {
			g.err = err
			return
		}
		confirmed := slices.Clone(prompt)
		curLogits := slices.Clone(logitsN[len(prompt)-1])

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

		hkvLen := len(prompt) - 1
		hkv := head.Prefill(m.be, embedOf, confirmed[:hkvLen], feats[:hkvLen], 0)

		for {
			C := len(confirmed)
			td := head.DraftTree(m.be, hkv.clone(), embedOf, confirmed[hkvLen], feats[hkvLen], hkvLen, B, D)

			// Verify the whole tree in one batched pass under tree attention.
			tc.treeRowPos, tc.treeMask = td.RowPos, td.Mask
			dlogits, _, err := captureN(td.Tokens, false)
			tc.treeRowPos, tc.treeMask = nil, nil
			if err != nil {
				g.err = err
				return
			}
			stats.Rounds++
			stats.Drafted += len(td.Tokens) // full b-ary tree, not B*D — M15

			// Best-path accept: walk the tree from the root, at each level following the
			// child whose token is the base's greedy argmax, as far as it keeps matching.
			var accepted []int
			last := curLogits
			parent := -1
			for {
				want := argmax(last)
				hit := -1
				for _, ch := range td.Children(parent) {
					if td.Tokens[ch] == want {
						hit = ch
						break
					}
				}
				if hit < 0 {
					break
				}
				accepted = append(accepted, td.Tokens[hit])
				last = dlogits[hit]
				parent = hit
			}
			correction := argmax(last)
			stats.Accepted += len(accepted)
			stats.Evaluated += len(accepted) + 1

			// Commit: drop the tree, re-forward the accepted path + correction cleanly so
			// their KV lands at the right positions and we get fresh features + logits.
			tc.TruncateTo(C)
			commit := append(append([]int{}, accepted...), correction)
			cl, cf, err := captureN(commit, false)
			if err != nil {
				g.err = err
				return
			}
			for i, tk := range accepted {
				confirmed = append(confirmed, tk)
				feats = append(feats, cf[i])
				if !emit(tk) {
					return
				}
			}
			confirmed = append(confirmed, correction)
			feats = append(feats, cf[len(accepted)])
			curLogits = slices.Clone(cl[len(accepted)])
			if !emit(correction) {
				return
			}
			newRoot := len(confirmed) - 1
			head.Extend(m.be, hkv, embedOf, confirmed[hkvLen:newRoot], feats[hkvLen:newRoot], hkvLen)
			hkvLen = newRoot
		}
	}()
	return out, g, nil
}

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
	// Not generateInto's resident path: this entry point may drive the resident KV on its
	// own schedule, so the recorded id list stops being true the moment it does. Forget it —
	// the next turn cold-prefills, which is slow, not wrong (resident_reuse.go).
	m.residentForgetIDs()
	if head == nil {
		return nil, nil, fmt.Errorf("decoder.GenerateEagleSpeculative: nil head")
	}
	if sp.Temperature != 0 || sp.LogitProcessor != nil {
		return nil, nil, fmt.Errorf("decoder.GenerateEagleSpeculative: greedy only (no temperature/LogitProcessor)")
	}
	if sp.HistoryDependent() { // argmax verify can't apply penalties / logit bias (M13)
		return nil, nil, fmt.Errorf("decoder.GenerateEagleSpeculative: repetition penalties / logit bias not supported in greedy speculative decoding; use Generate")
	}
	if !m.specRollbackSafe() {
		return nil, nil, fmt.Errorf("decoder.GenerateEagleSpeculative: recurrent family or staged sliding-window unsupported (rollback cannot restore)")
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
	// See GenerateEagleSpeculativeTree: canBatchN gates the batched hidden-state capture the
	// verify relies on; an own-runLayers family passes specRollbackSafe but never fills
	// cache.captured, panicking fuseAt. Refuse → the caller falls back to plain decode (M-07).
	if !m.canBatchN(K + 1) {
		return nil, nil, fmt.Errorf("decoder.GenerateEagleSpeculative: batched hidden-state capture not supported for arch %q (own runLayers)", m.w.arch.Name)
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
		captureN := func(ids []int, fastAttn bool) (logits [][]float32, feats [][]float32, err error) {
			tc.captureLayers = capLayers
			tc.captured = make([][]float32, len(capLayers))
			defer func() { tc.captureLayers, tc.captured = nil, nil }()
			lg, e := m.forwardNAttn(ctx, ids, tc, fastAttn)
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
		logitsN, feats, err := captureN(prompt, cpuFastAttention())
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

		// hkv holds the head's KV over the confirmed prefix confirmed[:hkvLen]; it grows
		// incrementally each round (cloned for drafting) instead of being rebuilt — O(C²)
		// rebuild was the dominant wall-clock cost.
		hkvLen := len(prompt) - 1
		hkv := head.Prefill(m.be, embedOf, confirmed[:hkvLen], feats[:hkvLen], 0)

		for {
			// Head drafts K tokens from the last confirmed token (position hkvLen).
			draft := head.DraftFrom(m.be, hkv.clone(), embedOf, confirmed[hkvLen], feats[hkvLen], hkvLen, K)
			C := len(confirmed)

			// Verify the draft in one batched target pass at positions C..C+K-1.
			dlogits, dfeats, err := captureN(draft, false)
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
			cl, cfeat, err := captureN([]int{correction}, false)
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
			// Grow the head KV over the newly-confirmed prefix tokens (up to the new root).
			newRoot := len(confirmed) - 1
			head.Extend(m.be, hkv, embedOf, confirmed[hkvLen:newRoot], feats[hkvLen:newRoot], hkvLen)
			hkvLen = newRoot
		}
	}()
	return out, g, nil
}
