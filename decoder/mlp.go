package decoder

import (
	"fmt"
	"math"
	"unsafe"

	"github.com/townsendmerino/aikit/linalg"
)

// moeSelTrace — when non-nil, records the top-k expert indices of every moeMLP call
// in forward order. SPIKE instrumentation for the #2 (MoE expert demand-paging)
// viability measurement (docs/ideas-weight-memory.md): de-interleave by NumLayers to
// get per-(layer, token) selections, then simulate LRU hit rate. Off (nil) in
// production — a single nil-check per MoE FFN, zero allocation. Set by the spike test.
var moeSelTrace [][]int

// moeWtsTrace mirrors moeSelTrace for the per-call routing WEIGHTS; together they
// capture a forward's full routing decision. moeSelOverride/moeWtsOverride — when
// non-nil — force each moeMLP call to REPLAY a recorded routing (idx+wts from a
// higher-precision reference run) in forward order instead of routing on its own
// hidden. The precision-localization experiment (E2, decoder/ssm_precision_localize_test.go):
// does feeding the f32-SSM forward the f64 reference's routing recover quality
// (→ a cheap router-selection island suffices) or not (→ the whole SSM needs precision)?
var (
	moeWtsTrace    [][]float32
	moeSelOverride [][]int
	moeWtsOverride [][]float32
	moeOverridePos int
)

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
func mlp(h, out []float32, lw *LayerWeights, arch *Architecture, be Backend, scr *decodeScratch, pager *expertPager, lora *loraLayerDelta) error {
	switch {
	case arch.MoE != nil && lw.Experts != nil:
		// Per-layer: GLM's first_k_dense_replace dense layers carry no experts and
		// fall through to gatedMLP; Mixtral/Qwen-MoE have experts on every layer.
		g, err := moeMLP(h, lw, arch, be, pager) // compute-time LoRA on experts not wired — LoadAdapter rejects MoE
		if err != nil {
			return err
		}
		copy(out, g)
		return nil
	case arch.NonGatedMLP:
		g, err := nonGatedMLP(h, lw, arch, be) // LoadAdapter rejects non-gated archs
		if err != nil {
			return err
		}
		copy(out, g)
		return nil
	default:
		return gatedMLP(h, out, lw, arch, be, scr, lora)
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
func moeMLP(h []float32, lw *LayerWeights, arch *Architecture, be Backend, pager *expertPager) ([]float32, error) {
	moe := arch.MoE
	nE, k := moe.NumExperts, moe.TopK
	if arch.Act != ActSiLU {
		return nil, fmt.Errorf("decoder: MoE expert activation %d unsupported (SwiGLU only)", arch.Act)
	}

	// Router logits → top-k experts + weights. softmax-topk (Mixtral/Qwen2-MoE) or
	// the DeepSeek/GLM sigmoid-score + selection-bias path — routeExperts unifies both.
	logits := make([]float32, nE)
	matmul(be, &lw.Router, h, logits, 1)
	var idx []int
	var wts []float32
	if moeSelOverride != nil { // E2: replay the higher-precision reference's routing
		idx, wts = moeSelOverride[moeOverridePos], moeWtsOverride[moeOverridePos]
		moeOverridePos++
	} else {
		idx, wts = routeExperts(logits, lw.RouterBias, k, moe.RouterSigmoid, moe.NormTopKProb, moe.RoutedScale, moe.NGroup, moe.TopkGroup)
	}
	if moeSelTrace != nil { // record this call's routing (forward order)
		moeSelTrace = append(moeSelTrace, append([]int(nil), idx...))
		moeWtsTrace = append(moeWtsTrace, append([]float32(nil), wts...))
	}
	// Weight residency (idea #2): the router selection is the demand signal. Touch
	// every chosen expert before the matmuls so the pager faults them in and keeps
	// resident RAM within budget (releasing the LRU tail). Bit-exact — released
	// experts re-fault from the read-only mapping.
	if pager != nil {
		for _, e := range idx {
			pager.touch(unsafe.Pointer(&lw.Experts[e]))
		}
	}

	// Weighted sum of the chosen experts (each a SwiGLU MLP). Experts use the
	// MoE expert width (Mellum's moe_intermediate_size), not the dense one.
	hidden := arch.HiddenDim
	out := make([]float32, hidden)
	expOut := make([]float32, hidden)
	// One gate/up pair for the whole token. The experts run sequentially, so k pairs were never
	// simultaneously live — this was 2*k allocations per token where 2 suffice, and at top-k 8 with
	// a large moe_intermediate that is the bulk of moeMLP's per-token allocation.
	sc := moe.IntermediateDim
	if moe.SharedIntermediateDim > sc {
		sc = moe.SharedIntermediateDim
	}
	egate, eup := make([]float32, sc), make([]float32, sc)
	for j, e := range idx {
		ex := &lw.Experts[e]
		swiGLUExpert(ex, h, expOut, moe.IntermediateDim, be, egate, eup)
		w := wts[j]
		for i := range out {
			out[i] += w * expOut[i]
		}
	}

	// Shared always-on expert (Qwen2-MoE / GLM). Qwen2 scales it by a per-token
	// sigmoid(SharedGate·h) gate; GLM/DeepSeek add it ungated.
	if moe.SharedIntermediateDim > 0 {
		swiGLUExpert(&lw.SharedExpert, h, expOut, moe.SharedIntermediateDim, be, egate, eup)
		if moe.SharedUngated {
			for i := range out {
				out[i] += expOut[i]
			}
		} else {
			var gl [1]float32
			matmul(be, &lw.SharedGate, h, gl[:], 1)
			g := float32(1.0 / (1.0 + math.Exp(-float64(gl[0])))) // sigmoid
			for i := range out {
				out[i] += g * expOut[i]
			}
		}
	}
	return out, nil
}

// routeExperts selects the top-k experts and their weights from router logits.
// Mixtral/Qwen2-MoE: softmax over experts, top-k by probability, weights = those
// probabilities. DeepSeek/GLM (sigmoid=true): per-expert sigmoid scores; if bias
// (e_score_correction_bias) is present it shifts the top-k SELECTION only, while the
// weights are the chosen experts' UN-biased sigmoid scores. norm renormalizes the
// weights to sum 1; scale (routed_scaling_factor) multiplies them (0/1 = no-op).
// With sigmoid=false, bias=nil, scale∈{0,1}, nGroup≤1 this is the exact prior Mixtral path.
//
// nGroup>1 (DeepSeek-V3 noaux_tc) adds group-limited selection: experts are partitioned
// into nGroup contiguous groups, each group scored by the sum of its top-2 selection
// scores; only experts in the top topkGroup groups are eligible for the per-token top-k.
func routeExperts(logits, bias []float32, k int, sigmoid, norm bool, scale float64, nGroup, topkGroup int) (idx []int, wts []float32) {
	var scores []float32
	if sigmoid {
		scores = make([]float32, len(logits))
		for i, l := range logits {
			scores[i] = float32(1.0 / (1.0 + math.Exp(-float64(l))))
		}
	} else {
		scores = softmaxF32(logits)
	}
	sel := scores
	if bias != nil {
		sel = make([]float32, len(scores))
		for i := range scores {
			sel[i] = scores[i] + bias[i]
		}
	}
	if nGroup > 1 {
		sel = groupLimit(sel, nGroup, topkGroup) // mask experts outside the top groups to -inf
	}
	idx, _ = topK(sel, k) // top-k by selection score
	wts = make([]float32, len(idx))
	for j, e := range idx {
		wts[j] = scores[e] // weights are the un-biased scores
	}
	if norm {
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
	if scale != 0 && scale != 1 {
		for j := range wts {
			wts[j] *= float32(scale)
		}
	}
	return idx, wts
}

// groupLimit implements DeepSeek-V3's group-limited expert selection. The experts'
// selection scores are partitioned into nGroup contiguous equal-size groups; each
// group is scored by the sum of its top-2 scores; the top topkGroup groups are kept
// and every expert outside them is masked to -inf. Returns a fresh slice (sel is not
// mutated). Mirrors DeepseekV3MoE.route_tokens_to_experts.
func groupLimit(sel []float32, nGroup, topkGroup int) []float32 {
	gsz := len(sel) / nGroup
	negInf := float32(math.Inf(-1))
	// Per-group score = sum of its two largest selection scores.
	gscore := make([]float32, nGroup)
	for g := range nGroup {
		var top1, top2 float32 = negInf, negInf
		for _, v := range sel[g*gsz : (g+1)*gsz] {
			if v > top1 {
				top1, top2 = v, top1
			} else if v > top2 {
				top2 = v
			}
		}
		gscore[g] = top1 + top2
	}
	keepIdx, _ := topK(gscore, topkGroup)
	keep := make([]bool, nGroup)
	for _, g := range keepIdx {
		keep[g] = true
	}
	out := make([]float32, len(sel))
	for g := range nGroup {
		for i := g * gsz; i < (g+1)*gsz; i++ {
			if keep[g] {
				out[i] = sel[i]
			} else {
				out[i] = negInf
			}
		}
	}
	// Trailing experts beyond nGroup*gsz (when NumExperts % nGroup != 0) belong to no group
	// and must be masked, not left at 0.0 — else they can outscore a legitimately -inf'd expert
	// (N-02; latent, real DeepSeek divides evenly, but a future/odd config would mis-route).
	for i := nGroup * gsz; i < len(sel); i++ {
		out[i] = negInf
	}
	return out
}

// swiGLUExpert evaluates one gated (SwiGLU) expert MLP of the given intermediate
// width into dst[:hidden]: dst = Down·(silu(Gate·h) ⊙ Up·h).
// P6: gate/up come from the CALLER so a token's k experts share one pair instead of allocating a
// pair each. They are fully overwritten by the two matmuls below before anything reads them, so
// reuse cannot carry state between experts and the result is bit-identical by construction — same
// operands, same order, same accumulation.
//
// expertScratch sizes them; a nil or short buffer allocates, so callers that have no scratch (the
// llama4 path) keep working unchanged.
func swiGLUExpert(ex *expertWeights, h, dst []float32, inter int, be Backend, gate, up []float32) {
	if cap(gate) < inter || cap(up) < inter {
		gate, up = make([]float32, inter), make([]float32, inter)
	}
	gate, up = gate[:inter], up[:inter]
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
func gatedMLP(h, out []float32, lw *LayerWeights, arch *Architecture, be Backend, scr *decodeScratch, lora *loraLayerDelta) error {
	gate, up := scr.gate, scr.up // [inter] scratch; matmul fully overwrites each
	if isW8A8(&lw.GateProj) && isW8A8(&lw.UpProj) {
		scr.gateUpOps[0] = linalg.W8A8Op{BQ: wmInt8(&lw.GateProj), Scales: wmScales(&lw.GateProj), Dst: gate, N: lw.GateProj.Rows()}
		scr.gateUpOps[1] = linalg.W8A8Op{BQ: wmInt8(&lw.UpProj), Scales: wmScales(&lw.UpProj), Dst: up, N: lw.UpProj.Rows()}
		matmulW8A8Batch(be, scr.ws, h, 1, lw.GateProj.Cols(), scr.gateUpOps[:]) // gate/up in one dispatch (GPU: one submit)
	} else {
		matmulInto(scr.ws, be, &lw.GateProj, h, gate, 1)
		matmulInto(scr.ws, be, &lw.UpProj, h, up, 1)
	}
	if lora != nil { // compute-time LoRA (#7): delta into gate/up before the activation
		applyLoRA(lora.gate, h, gate, scr)
		applyLoRA(lora.up, h, up, scr)
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
	matmulInto(scr.ws, be, &lw.DownProj, gate, out, 1) // [1,hidden] = mid · DownProjᵀ (gate now holds the activated mid)
	if lora != nil {
		applyLoRA(lora.down, gate, out, scr)
	}
	return nil
}
