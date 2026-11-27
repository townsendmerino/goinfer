package cpuengine

import (
	"unsafe"

	darch "github.com/townsendmerino/goinfer/decoder/arch"
	"github.com/townsendmerino/goinfer/internal/runtime/archparts"
	"github.com/townsendmerino/goinfer/internal/runtime/normact"
)

// Nemotron-H (nemotron_h): a SINGLE-OP-per-block hybrid. Each layer is exactly one of {Mamba-2 mixer, NoPE GQA attention,
// non-gated relu² MLP, non-gated relu² MoE FFN}, applied pre-norm with a residual add, not the mixer+FFN block every other
// family uses (Architecture.nemotronLayerOps). The moe layers (Nemotron 3 Nano only; plain Nemotron-H never sets arch.MoE) are
// DeepSeek-V3-style sigmoid + e_score_correction_bias + group-limited top-k routing (routeExperts, the primitive DeepSeek, GLM,
// Mellum and Qwen3.5 share) over non-gated relu² experts, not moeMLP, which requires gated SwiGLU experts.

// NemotronMoE is Nemotron 3 Nano's MoE FFN: route with routeExperts (sigmoid +
// e_score_correction_bias + group-limited top-k — the same primitive DeepSeek/GLM
// share), evaluate the selected experts as non-gated relu² FFNs (NOT moeMLP's gated
// SwiGLU — this family's experts have only up_proj/down_proj, no gate_proj,
// confirmed against the real safetensors index), weighted-sum them, then add the
// always-on shared expert UNGATED (routing.SharedUngated is unconditionally true for
// this family — verified against NemotronHTopkRouter's forward, which does a plain
// `hidden_states = hidden_states + self.shared_experts(residuals)`, no sigmoid gate).
func (m *Engine) NemotronMoE(n []float32, lw *darch.LayerWeights, arch *darch.Architecture, hidden int) []float32 {
	return m.nemotronMoEWith(n, lw, arch.MoE, hidden)
}

// nemotronMoEWith is nemotronMoE with the routing given: the program walker's form.
func (m *Engine) nemotronMoEWith(n []float32, lw *darch.LayerWeights, moe *darch.MoEConfig, hidden int) []float32 {
	logits := make([]float32, moe.NumExperts)
	Matmul(m.Be, &lw.Router, n, logits, 1)
	idx, wts := RouteExperts(logits, lw.RouterBias, moe.TopK, moe.RouterSigmoid, moe.NormTopKProb, moe.RoutedScale, moe.NGroup, moe.TopkGroup)

	// Touch every routed expert before evaluating it, as moeMLP does: without it, m.Pager's budget and SpanCache LRU are built but
	// never consulted for Nemotron 3 Nano, so -stream-weights enforces no RAM bound on this family.
	if m.Pager != nil {
		for _, e := range idx {
			m.Pager.Touch(unsafe.Pointer(&lw.Experts[e]))
		}
	}

	out := make([]float32, hidden)
	for j, e := range idx {
		expOut := m.nemotronExpertFFN(&lw.Experts[e], n, moe.IntermediateDim, hidden)
		AddScaled(out, expOut, wts[j])
	}
	if moe.SharedIntermediateDim > 0 {
		shOut := m.nemotronExpertFFN(&lw.SharedExpert, n, moe.SharedIntermediateDim, hidden)
		AddResidual(out, shOut)
	}
	return out
}

// nemotronExpertFFN is one MoE expert's non-gated relu² FFN: down(relu²(up(x))) —
// the same shape as an mlp block's non-gated FFN, just parameterized by width
// (routed experts and the shared expert differ) and evaluated per-expert. Only
// expertWeights.Up/.Down are populated for this family; .Gate is left unset.
func (m *Engine) nemotronExpertFFN(ex *archparts.ExpertWeights, n []float32, inter, hidden int) []float32 {
	up := make([]float32, inter)
	Matmul(m.Be, &ex.Up, n, up, 1)
	for i := range up {
		up[i] = normact.ReLU2(up[i])
	}
	out := make([]float32, hidden)
	Matmul(m.Be, &ex.Down, up, out, 1)
	return out
}
