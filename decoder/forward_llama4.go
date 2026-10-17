package decoder

import (
	"math"
	"unsafe"
)

// Llama 4 (llama4_text): the iRoPE text decoder. Its program is Architecture.llama4LayerOps: a pre-norm layer whose attention
// varies per layer (RoPE layers rotate, normalise q and k without weights, and attend within a chunk of attention_chunk_size;
// NoPE layers scale the query by the attention temperature instead) and whose FFN is dense or the MoE below.
//
// Chunked attention: below the chunk size it equals full causal, which is all the parity gates exercise, so they cannot catch a
// dropped chunk bound (TestLlama4_attendQueryHonoursTheChunk pins the bound).

// llama4MoE is Llama 4's mixture-of-experts FFN. Unlike the standard moeMLP (which scales
// each expert's OUTPUT by its router weight), Llama 4 scales the expert's INPUT by the
// router score — routed_in = score·h, then expert(routed_in) — and since the expert is
// SwiGLU (nonlinear), input-scaling ≠ output-scaling, so it gets its own path. The score is
// sigmoid(logit) of the top-k experts (top-1 by default); the always-on shared expert runs
// on the unscaled h and is added ungated. out = shared(h) + Σ_topk expert_e(sigmoid(logit_e)·h).
func (m *Model) llama4MoE(h, out []float32, lw *LayerWeights, arch *Architecture) {
	m.llama4MoEWith(h, out, lw, arch.MoE, arch.HiddenDim)
}

// llama4MoEWith is llama4MoE with the routing and width given: the program walker's form.
func (m *Model) llama4MoEWith(h, out []float32, lw *LayerWeights, moe *MoEConfig, hidden int) {
	logits := make([]float32, moe.NumExperts)
	matmul(m.be, &lw.Router, h, logits, 1)
	idx, _ := topK(logits, moe.TopK) // top-k by raw logit (== top-k by sigmoid, monotonic)

	// One gate/up pair for the token, shared by the shared expert and every routed one: they run sequentially, so k pairs are never
	// simultaneously live.
	sc := max(moe.SharedIntermediateDim, moe.IntermediateDim)
	l4gate, l4up := make([]float32, sc), make([]float32, sc)

	// Shared expert on the unscaled input, ungated.
	swiGLUExpert(&lw.SharedExpert, h, out, moe.SharedIntermediateDim, m.be, l4gate, l4up, nil)

	// Touch every routed expert before evaluating it, as moeMLP does: without it, m.pager's budget and SpanCache LRU are built but
	// never consulted for Llama 4, so -stream-weights enforces no RAM bound on this family.
	if m.pager != nil {
		for _, e := range idx {
			m.pager.touch(unsafe.Pointer(&lw.Experts[e]))
		}
	}

	scaled := make([]float32, hidden)
	expOut := make([]float32, hidden)
	for _, e := range idx {
		w := sigmoidf(logits[e]) // router weight = sigmoid of the (unbiased) logit
		for i := range h {
			scaled[i] = w * h[i]
		}
		swiGLUExpert(&lw.Experts[e], scaled, expOut, moe.IntermediateDim, m.be, l4gate, l4up, nil) // each expert's input is h scaled by its own weight
		addResidual(out, expOut)
	}
}

// l2NormHeads applies Llama 4's parameter-free QK-norm — a per-head RMS over head_dim with
// no learned weight: x ← x · rsqrt(mean(x²) + eps). vec is [heads, headDim] flattened.
func l2NormHeads(vec []float32, heads, headDim int, eps float64) {
	for hh := range heads {
		seg := vec[hh*headDim : (hh+1)*headDim]
		var ss float64
		for _, x := range seg {
			ss += float64(x) * float64(x)
		}
		scale := 1.0 / math.Sqrt(ss/float64(headDim)+eps)
		for i := range seg {
			seg[i] = float32(float64(seg[i]) * scale)
		}
	}
}
