package decoder

import (
	"context"
	"fmt"
)

// Shared state, many questions (docs/tasks/task-constrained-confidence.md, D8). A decision request asks several
// questions about one state, and on the hybrid Qwen3.5 family each question would prefill the whole prompt again. The
// state is the same in every question's prompt, so its prefill is done once and its CACHE (the KV rows of the attention
// layers and the Gated DeltaNet's recurrent state, which has no per-position history and so cannot be rewound, only
// copied) is resumed once per question.
//
// What this covers, and what it deliberately does not:
//   - The CPU path only. A model with a resident backend keeps answering from the device (PromptHidden and Generate prefer it), whose own prefill is
//     faster than a CPU one; copying the DEVICE's recurrent state is a per-backend change that does not exist, so on a resident model every method here
//     falls back to the one-prompt-at-a-time path and says nothing different.
//   - The Qwen3.5 batched family (qwen35BatchN) with the plain f32 or int8 KV: no sliding-window ring, no m-RoPE, no capture hooks, no layer pager. Anything
//     else takes the fallback.
//   - A shared prefix is found, not assumed: the longest common token prefix of the prompts actually built (the ids, after tokenization), so a template that
//     puts the question before the state (bare-v1 puts "[kind]" first) shares only what it really shares.

// prefixShareHook, when non-nil, is called once per checkpoint PromptHiddenMany takes, with the prefix length and the number of prompts that resume from it. It
// exists so a test can prove that sharing happened (an answer that matches the unshared one cannot tell sharing from its absence).
var prefixShareHook func(prefixLen, members int)

// minSharedPrefix is the shortest common prefix worth checkpointing: below it the copy costs more than the prefill it saves.
const minSharedPrefix = 16

// prefixCheckpoint is an immutable snapshot of a prompt prefix's cache. It is never run forward: every resume copies it into a fresh cache first.
type prefixCheckpoint struct {
	cache *KVCache
	n     int
}

// CanSharePrefix reports whether this model takes the CPU prefix-checkpoint path (see above): no resident backend, the batched Qwen3.5 forward, and a cache
// that is only KV rows plus Gated DeltaNet state.
func (m *Model) CanSharePrefix() bool {
	if m.resident != nil || m.w == nil {
		return false
	}
	c := m.NewCache(2)
	return m.qwen35BatchN(2, c) && c.delta != nil && c.conv == nil && c.mamba == nil && c.kda == nil && c.mlaLatent == nil && !c.localAny
}

// copyStateFrom copies src's per-sequence state into c, a fresh cache from the same model: the position, the global layers' KV rows (f32 or int8) and the
// Gated DeltaNet states, deep, so neither cache can reach the other's memory. It refuses what it does not know how to copy.
func (c *KVCache) copyStateFrom(src *KVCache) error {
	switch {
	case src.conv != nil || src.mamba != nil || src.kda != nil || src.mlaLatent != nil:
		return fmt.Errorf("decoder: a prefix checkpoint of a conv, Mamba, KDA or MLA cache is not implemented")
	case src.localAny || src.imgBlocks != nil || src.mropePos != nil || src.captureLayers != nil || src.treeMask != nil:
		return fmt.Errorf("decoder: a prefix checkpoint of a windowed, multimodal or capturing cache is not implemented")
	case c.numLayers != src.numLayers || c.quant != src.quant || (c.delta == nil) != (src.delta == nil):
		return fmt.Errorf("decoder: the checkpoint's cache and the target's differ in shape")
	}
	c.pos, c.mropeDelta, c.manualPos = src.pos, src.mropeDelta, src.manualPos
	for l := 0; l < src.numLayers; l++ {
		c.stride[l] = src.stride[l]
		if src.quant == kvI8 {
			c.keysQ[l] = append(c.keysQ[l][:0], src.keysQ[l]...)
			c.valsQ[l] = append(c.valsQ[l][:0], src.valsQ[l]...)
			c.keyScale[l] = append(c.keyScale[l][:0], src.keyScale[l]...)
			c.valScale[l] = append(c.valScale[l][:0], src.valScale[l]...)
			continue
		}
		c.keys[l] = append(c.keys[l][:0], src.keys[l]...)
		c.vals[l] = append(c.vals[l][:0], src.vals[l]...)
	}
	for l, st := range src.delta {
		if st == nil {
			continue
		}
		dst := c.delta[l]
		if dst == nil {
			return fmt.Errorf("decoder: layer %d has a recurrent state in the checkpoint and none in the target", l)
		}
		dst.convWin = make([][]float32, len(st.convWin))
		for i, row := range st.convWin {
			dst.convWin[i] = append([]float32(nil), row...)
		}
		dst.s = append(dst.s[:0], st.s...)
	}
	return nil
}

// checkpointPrefix prefills prefix on the CPU in a fresh cache and keeps the cache as a checkpoint.
func (m *Model) checkpointPrefix(ctx context.Context, prefix []int) (*prefixCheckpoint, error) {
	cache := m.NewCache(len(prefix))
	if len(prefix) < 2 || !m.qwen35BatchN(len(prefix), cache) {
		return nil, fmt.Errorf("decoder: the prefix cannot take the batched Qwen3.5 path")
	}
	if _, err := m.runLayersQwen35N(ctx, m.embedN(prefix), cache); err != nil {
		return nil, err
	}
	return &prefixCheckpoint{cache: cache, n: len(prefix)}, nil
}

// resumeHidden copies the checkpoint into a fresh cache and prefills suffix, returning the final-norm hidden state at its last position. The checkpoint is
// not touched, so one checkpoint serves any number of suffixes.
func (m *Model) resumeHidden(ctx context.Context, cp *prefixCheckpoint, suffix []int) ([]float32, error) {
	if len(suffix) == 0 {
		return nil, fmt.Errorf("decoder: an empty suffix (the prompt would end at the shared prefix)")
	}
	a := m.w.arch
	cache := m.NewCache(cp.n + len(suffix))
	if err := cache.copyStateFrom(cp.cache); err != nil {
		return nil, err
	}
	if len(suffix) > 1 && m.qwen35BatchN(len(suffix), cache) {
		hN, err := m.runLayersQwen35N(ctx, m.embedN(suffix), cache)
		if err != nil {
			return nil, err
		}
		return append([]float32(nil), hN[(len(suffix)-1)*a.HiddenDim:]...), nil
	}
	var h []float32
	for _, id := range suffix {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		var err error
		if h, err = m.runLayers(id, cache); err != nil {
			return nil, err
		}
	}
	out := append([]float32(nil), h[:a.HiddenDim]...)
	normalize(a, out, m.w.FinalNorm, m.w.FinalNormBias, a.HiddenDim)
	return out, nil
}

// shareGroups partitions the prompts into groups that share a prefix of at least minShared tokens, each group greedily anchored on its first member, and
// returns for every group its member indices and the longest prefix common to ALL of them (cut so every member keeps at least one token of its own: the
// last position's hidden state must come from a suffix run). A prompt that shares nothing is a group of one with prefix 0.
func shareGroups(prompts [][]int, minShared int) (members [][]int, prefix []int) {
	assigned := make([]bool, len(prompts))
	for i := range prompts {
		if assigned[i] {
			continue
		}
		g, p := []int{i}, len(prompts[i])-1
		assigned[i] = true
		for j := i + 1; j < len(prompts); j++ {
			if assigned[j] {
				continue
			}
			l := min(commonPrefixLen(prompts[i], prompts[j]), len(prompts[j])-1)
			if l >= minShared && min(p, l) >= minShared {
				p = min(p, l)
				g = append(g, j)
				assigned[j] = true
			}
		}
		if len(g) == 1 {
			p = 0
		}
		members, prefix = append(members, g), append(prefix, p)
	}
	return members, prefix
}

// PromptHiddenMany returns PromptHidden for every prompt, prefilling the prefix that prompts share ONCE. It is the many-question form of PromptHidden (a Route B
// decision head reads the last position's final-norm hidden state): prompts that share a prefix of at least minSharedPrefix tokens are grouped, the group's common
// prefix is prefilled and checkpointed, and each member resumes from a copy and prefills only its own suffix. A prompt that shares nothing, and every prompt on a
// model that cannot take the checkpoint path (CanSharePrefix), is answered by PromptHidden exactly as before.
//
// Not bit-identical to PromptHidden: splitting one batched prefill in two changes the matmuls' row blocking, and the recurrence is sequential either way. The
// difference is bounded by TestPromptHiddenMany_matchesSeparate. ctx is checked per layer, as PromptHidden's batched path checks it.
func (m *Model) PromptHiddenMany(ctx context.Context, prompts [][]int) ([][]float32, error) {
	out := make([][]float32, len(prompts))
	for i, p := range prompts {
		if len(p) == 0 {
			return nil, fmt.Errorf("decoder.PromptHiddenMany: prompt %d is empty", i)
		}
		if err := m.checkHiddenIDs("decoder.PromptHiddenMany", p); err != nil {
			return nil, err
		}
	}
	alone := func(i int) error {
		h, err := m.PromptHidden(ctx, prompts[i])
		out[i] = h
		return err
	}
	if len(prompts) < 2 || !m.CanSharePrefix() {
		for i := range prompts {
			if err := alone(i); err != nil {
				return nil, err
			}
		}
		return out, nil
	}
	groups, prefixLen := shareGroups(prompts, minSharedPrefix)
	for gi, g := range groups {
		var cp *prefixCheckpoint
		if prefixLen[gi] > 0 {
			var err error
			if cp, err = m.checkpointPrefix(ctx, prompts[g[0]][:prefixLen[gi]]); err == nil {
				if prefixShareHook != nil {
					prefixShareHook(prefixLen[gi], len(g))
				}
			} else {
				if ctx.Err() != nil {
					return nil, err
				}
				cp = nil // unsupported or failed: each member goes alone
			}
		}
		for _, i := range g {
			if cp == nil {
				if err := alone(i); err != nil {
					return nil, err
				}
				continue
			}
			h, err := m.resumeHidden(ctx, cp, prompts[i][prefixLen[gi]:])
			if err != nil {
				return nil, err
			}
			out[i] = h
		}
	}
	return out, nil
}

// PromptLogitsMany is PromptHiddenMany followed by the LM head: the next-token logits after each prompt, in the model's vocabulary. A label-scoring decision
// (Route A) reads these. It is only meaningful where CanSharePrefix is true; elsewhere a caller keeps its own prefill (a resident model's, which this
// method does not use).
func (m *Model) PromptLogitsMany(ctx context.Context, prompts [][]int) ([][]float32, error) {
	hs, err := m.PromptHiddenMany(ctx, prompts)
	if err != nil {
		return nil, err
	}
	out := make([][]float32, len(hs))
	for i, h := range hs {
		out[i] = m.lmHeadN(h, 1)
	}
	return out, nil
}
