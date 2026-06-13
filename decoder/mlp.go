package decoder

import (
	"fmt"
	"math"

	"github.com/townsendmerino/aikit/linalg"
)

// moeSelTrace — when non-nil, records the top-k expert indices of every moeMLP call
// in forward order. SPIKE instrumentation for the #2 (MoE expert demand-paging)
// viability measurement (docs/ideas-weight-memory.md): de-interleave by NumLayers to
// get per-(layer, token) selections, then simulate LRU hit rate. Off (nil) in
// production — a single nil-check per MoE FFN, zero allocation. Set by the spike test.
var moeSelTrace [][]int

// gatedMLP runs one block's gated MLP for the current position and returns the
// output (caller applies the post-MLP norm + residual add). The gate/up/down
// structure is shared by GeGLU (Gemma) and SwiGLU (Llama/Mistral/Qwen); only
// the gate activation differs (Architecture.Act).
//
//	gate = GateProj·h            // [IntermediateDim]
//	up   = UpProj·h              // [IntermediateDim]
//	mid  = act(gate) ⊙ up        // [IntermediateDim]
//	out  = DownProj·mid          // [HiddenDim]
//
// mlp runs the block's feed-forward network, dispatching on the descriptor:
// a sparse mixture of experts (Mixtral), GPT-2's non-gated up→act→down with
// biases, or the gated GeGLU/SwiGLU shared by Gemma/Llama/Qwen.
// mlp writes the FFN output for input h into out ([hidden]); the caller applies
// any post-MLP norm + residual. The hot dense path (gatedMLP) reuses scratch and
// writes straight into out; the rarer MoE / non-gated paths still allocate
// internally and are copied into out.
func mlp(h, out []float32, lw *LayerWeights, arch *Architecture, be Backend, scr *decodeScratch) error {
	switch {
	case arch.MoE != nil:
		g, err := moeMLP(h, lw, arch, be)
		if err != nil {
			return err
		}
		copy(out, g)
		return nil
	case arch.NonGatedMLP:
		g, err := nonGatedMLP(h, lw, arch, be)
		if err != nil {
			return err
		}
		copy(out, g)
		return nil
	default:
		return gatedMLP(h, out, lw, arch, be, scr)
	}
}

// moeMLP runs a sparse mixture-of-experts FFN (Mixtral). The router scores all
// experts; the top-k by softmax probability run as gated SwiGLU MLPs and their
// outputs combine weighted by the (optionally renormalized) router weights:
//
//	probs   = softmax(Router·h)              // over all NumExperts
//	(w, e)  = topk(probs, TopK)              // weights + expert indices
//	if NormTopKProb { w /= sum(w) }          // Mixtral renormalizes
//	out     = Σ_j w[j] · expert_{e[j]}(h)    // expert = down(silu(gate(h)) ⊙ up(h))
//
// Only the chosen experts are evaluated — the point of MoE.
func moeMLP(h []float32, lw *LayerWeights, arch *Architecture, be Backend) ([]float32, error) {
	moe := arch.MoE
	nE, k := moe.NumExperts, moe.TopK
	if arch.Act != ActSiLU {
		return nil, fmt.Errorf("decoder: MoE expert activation %d unsupported (SwiGLU only)", arch.Act)
	}

	// Router logits → full softmax → top-k probabilities.
	logits := make([]float32, nE)
	matmul(be, &lw.Router, h, logits, 1)
	probs := softmaxF32(logits)
	idx, wts := topK(probs, k)
	if moeSelTrace != nil { // SPIKE: record this call's expert selection (forward order)
		moeSelTrace = append(moeSelTrace, append([]int(nil), idx...))
	}
	if moe.NormTopKProb {
		var s float32
		for _, w := range wts {
			s += w
		}
		if s > 0 {
			for j := range wts {
				wts[j] /= s
			}
		}
	}

	// Weighted sum of the chosen experts (each a SwiGLU MLP). Experts use the
	// MoE expert width (Mellum's moe_intermediate_size), not the dense one.
	hidden := arch.HiddenDim
	out := make([]float32, hidden)
	expOut := make([]float32, hidden)
	for j, e := range idx {
		ex := &lw.Experts[e]
		swiGLUExpert(ex, h, expOut, moe.IntermediateDim, be)
		w := wts[j]
		for i := range out {
			out[i] += w * expOut[i]
		}
	}

	// Shared expert (Qwen2-MoE): an always-on gated MLP whose output is scaled by
	// a per-token sigmoid gate and added to the routed sum.
	if moe.SharedIntermediateDim > 0 {
		swiGLUExpert(&lw.SharedExpert, h, expOut, moe.SharedIntermediateDim, be)
		var gl [1]float32
		matmul(be, &lw.SharedGate, h, gl[:], 1)
		g := float32(1.0 / (1.0 + math.Exp(-float64(gl[0])))) // sigmoid
		for i := range out {
			out[i] += g * expOut[i]
		}
	}
	return out, nil
}

// swiGLUExpert evaluates one gated (SwiGLU) expert MLP of the given intermediate
// width into dst[:hidden]: dst = Down·(silu(Gate·h) ⊙ Up·h).
func swiGLUExpert(ex *expertWeights, h, dst []float32, inter int, be Backend) {
	gate := make([]float32, inter)
	up := make([]float32, inter)
	matmul(be, &ex.Gate, h, gate, 1)
	matmul(be, &ex.Up, h, up, 1)
	for i := range gate {
		gate[i] = silu(gate[i]) * up[i]
	}
	matmul(be, &ex.Down, gate, dst, 1)
}

// softmaxF32 returns the softmax of xs (float64 accumulation, max-shifted for
// stability). Small (NumExperts) so allocation is cheap.
func softmaxF32(xs []float32) []float32 {
	maxv := xs[0]
	for _, v := range xs {
		if v > maxv {
			maxv = v
		}
	}
	out := make([]float32, len(xs))
	var sum float64
	for i, v := range xs {
		e := math.Exp(float64(v - maxv))
		out[i] = float32(e)
		sum += e
	}
	inv := float32(1.0 / sum)
	for i := range out {
		out[i] *= inv
	}
	return out
}

// topK returns the indices and values of the k largest entries of xs, in
// descending order. O(k·n) selection — k and n (NumExperts) are tiny.
func topK(xs []float32, k int) ([]int, []float32) {
	idx := make([]int, 0, k)
	val := make([]float32, 0, k)
	used := make([]bool, len(xs))
	for ; k > 0; k-- {
		best, bi := float32(math.Inf(-1)), -1
		for i, v := range xs {
			if !used[i] && v > best {
				best, bi = v, i
			}
		}
		if bi < 0 {
			break
		}
		used[bi] = true
		idx = append(idx, bi)
		val = append(val, best)
	}
	return idx, val
}

// nonGatedMLP runs GPT-2's feed-forward block: a single up projection, an
// activation, and a down projection, each with an additive bias.
//
//	mid = act(UpProj·h + UpBias)      // [IntermediateDim]
//	out = DownProj·mid + DownBias     // [HiddenDim]
func nonGatedMLP(h []float32, lw *LayerWeights, arch *Architecture, be Backend) ([]float32, error) {
	inter, hidden := arch.IntermediateDim, arch.HiddenDim
	mid := make([]float32, inter)
	matmul(be, &lw.UpProj, h, mid, 1)
	if lw.UpBias != nil {
		addBias(mid, lw.UpBias)
	}
	switch arch.Act {
	case ActGeluTanh: // GPT-2's "gelu_new" is the tanh approximation
		for i := range mid {
			mid[i] = geluTanh(mid[i])
		}
	default:
		return nil, fmt.Errorf("decoder: unsupported non-gated activation %d (have gelu-tanh)", arch.Act)
	}
	out := make([]float32, hidden)
	matmul(be, &lw.DownProj, mid, out, 1)
	if lw.DownBias != nil {
		addBias(out, lw.DownBias)
	}
	return out, nil
}

// No biases on any projection.
func gatedMLP(h, out []float32, lw *LayerWeights, arch *Architecture, be Backend, scr *decodeScratch) error {
	gate, up := scr.gate, scr.up // [inter] scratch; matmul fully overwrites each
	if isW8A8(&lw.GateProj) && isW8A8(&lw.UpProj) {
		scr.gateUpOps[0] = linalg.W8A8Op{BQ: wmInt8(&lw.GateProj), Scales: wmScales(&lw.GateProj), Dst: gate, N: lw.GateProj.Rows()}
		scr.gateUpOps[1] = linalg.W8A8Op{BQ: wmInt8(&lw.UpProj), Scales: wmScales(&lw.UpProj), Dst: up, N: lw.UpProj.Rows()}
		matmulW8A8Batch(be, scr.ws, h, 1, lw.GateProj.Cols(), scr.gateUpOps[:]) // gate/up in one dispatch (GPU: one submit)
	} else {
		matmulInto(scr.ws, be, &lw.GateProj, h, gate, 1)
		matmulInto(scr.ws, be, &lw.UpProj, h, up, 1)
	}
	switch arch.Act {
	case ActGeluTanh:
		for i := range gate {
			gate[i] = geluTanh(gate[i]) * up[i]
		}
	case ActSiLU:
		for i := range gate {
			gate[i] = silu(gate[i]) * up[i]
		}
	default:
		return fmt.Errorf("decoder: unsupported activation %d (have GeGLU/SwiGLU)", arch.Act)
	}
	matmulInto(scr.ws, be, &lw.DownProj, gate, out, 1) // [1,hidden] = mid · DownProjᵀ
	return nil
}
