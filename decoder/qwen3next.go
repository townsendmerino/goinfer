package decoder

// splitQwen3NextQKVZ and splitQwen3NextBA undo qwen3_next's checkpoint-level
// fusion of the Gated DeltaNet input projections. qwen3_5_moe's checkpoint
// stores four separate tensors (in_proj_qkv, in_proj_z, in_proj_b, in_proj_a);
// qwen3_next fuses them into in_proj_qkvz and in_proj_ba instead — same math,
// different packing (verified against modular_qwen3_next.py's
// Qwen3NextGatedDeltaNet.torch_forward: mixed_qkvz/mixed_ba are reshaped to
// [..., num_k_heads, per_group_dim] and torch.split along that per-group axis).
//
// Per key-head group g, the fused row ranges are contiguous:
//
//	in_proj_qkvz group g: [ q_g(head_k_dim) | k_g(head_k_dim) | v_g(rep·head_v_dim) | z_g(rep·head_v_dim) ]
//	in_proj_ba   group g: [ b_g(rep)                          | a_g(rep) ]
//
// where rep = num_v_heads/num_k_heads. Because a Linear layer's output-feature
// index maps 1:1 to its weight's output (row) index, splitting the *activation*
// per HF's reshape is equivalent to splitting the *weight matrix* by row — and
// because group g's v/z (or b/a) sub-ranges already cover value-head indices
// [g·rep, (g+1)·rep), gathering them group-by-group directly produces the flat,
// value-head-ordered layout loadQwen35Attn's four separate tensors already use
// (the same order untileVHeads produces from GGUF's differently-tiled layout).
// So each destination range is one contiguous row-slice copy — no per-element
// interleaving needed.
func splitQwen3NextQKVZ(qkvz []float32, g *qwen35Params, hidden int) (qkv, z []float32) {
	numK, numV := g.NumKeyHeads, g.NumValueHeads
	rep := numV / numK
	hk, hv := g.KeyHeadDim, g.ValueHeadDim
	keyDim, valueDim := hk*numK, hv*numV
	groupRows := 2*hk + 2*rep*hv // q_g + k_g + v_g + z_g

	qkv = make([]float32, (2*keyDim+valueDim)*hidden)
	z = make([]float32, valueDim*hidden)
	for gi := range numK {
		src := qkvz[gi*groupRows*hidden:]
		// q_g -> qkv[g*hk : ...]
		copy(qkv[gi*hk*hidden:], src[:hk*hidden])
		src = src[hk*hidden:]
		// k_g -> qkv[keyDim + g*hk : ...]
		copy(qkv[(keyDim+gi*hk)*hidden:], src[:hk*hidden])
		src = src[hk*hidden:]
		// v_g -> qkv[2*keyDim + g*rep*hv : ...] (group g owns value heads [g*rep,(g+1)*rep))
		copy(qkv[(2*keyDim+gi*rep*hv)*hidden:], src[:rep*hv*hidden])
		src = src[rep*hv*hidden:]
		// z_g -> z[g*rep*hv : ...]
		copy(z[gi*rep*hv*hidden:], src[:rep*hv*hidden])
	}
	return qkv, z
}

func splitQwen3NextBA(ba []float32, g *qwen35Params, hidden int) (b, a []float32) {
	numK, numV := g.NumKeyHeads, g.NumValueHeads
	rep := numV / numK
	groupRows := 2 * rep // b_g + a_g

	b = make([]float32, numV*hidden)
	a = make([]float32, numV*hidden)
	for gi := range numK {
		src := ba[gi*groupRows*hidden:]
		copy(b[gi*rep*hidden:], src[:rep*hidden])
		src = src[rep*hidden:]
		copy(a[gi*rep*hidden:], src[:rep*hidden])
	}
	return b, a
}
