package decoder

import (
	"fmt"
	"math"
	"sync"
	"sync/atomic"
	"time"
	"unsafe"

	"github.com/townsendmerino/aikit/linalg"
)

// moeSelTrace, when non-nil, records the top-k expert indices of every moeMLP call in forward order (de-interleave by NumLayers
// for per-(layer, token) selections). Test instrumentation set by the demand-paging spike test; nil in production, costing one
// nil check per MoE FFN.
var moeSelTrace [][]int

// moeWtsTrace mirrors moeSelTrace for the routing weights. moeSelOverride/moeWtsOverride, when non-nil, make each moeMLP call
// replay a recorded routing (from a higher-precision reference run) in forward order instead of routing on its own hidden; used
// by the precision-localization experiment (ssm_precision_localize_test.go).
var (
	moeWtsTrace    [][]float32
	moeSelOverride [][]int
	moeWtsOverride [][]float32
	moeOverridePos int
)

// mlp writes the FFN output for input h into out ([hidden]); the caller applies any post-MLP norm and residual. It dispatches on
// the descriptor: a sparse mixture of experts (moeMLP), GPT-2's non-gated up→act→down with biases (nonGatedMLP), or the gated
// GeGLU/SwiGLU shared by Gemma/Llama/Qwen (gatedMLP). The hot dense path reuses scratch and writes straight into out; the rarer
// MoE and non-gated paths allocate internally and are copied into out.
func mlp(h, out []float32, lw *LayerWeights, arch *Architecture, be Backend, scr *decodeScratch, pager *expertPager, lora *loraLayerDelta) error {
	switch {
	case arch.MoE != nil && lw.Experts != nil:
		// Per-layer: GLM's first_k_dense_replace dense layers carry no experts and
		// fall through to gatedMLP; Mixtral/Qwen-MoE have experts on every layer.
		g, err := moeMLP(h, lw, arch, be, scr, pager) // compute-time LoRA on experts not wired — LoadAdapter rejects MoE
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

// moeMLP runs a sparse mixture-of-experts FFN. The router scores all experts; the top-k run as gated SwiGLU MLPs and their
// outputs combine weighted by the (optionally renormalized) router weights:
// out = Σ_j w[j] · expert_{e[j]}(h), with expert = down(silu(gate(h)) ⊙ up(h)). Only the chosen experts are evaluated.
//
// scr, when non-nil, backs the router-logits/accumulator/expert-gate-up buffers with per-stream scratch instead of allocating
// them per call. The single-token decode call sites always pass their cache's scr; the batched-prefill call site (forwardN) has
// none in scope and passes nil, allocating per call, amortized over the K-token batch.
func moeMLP(h []float32, lw *LayerWeights, arch *Architecture, be Backend, scr *decodeScratch, pager *expertPager) ([]float32, error) {
	moe := arch.MoE
	nE, k := moe.NumExperts, moe.TopK
	if arch.Act != ActSiLU {
		return nil, fmt.Errorf("decoder: MoE expert activation %d unsupported (SwiGLU only)", arch.Act)
	}

	// Router logits → top-k experts + weights. softmax-topk (Mixtral/Qwen2-MoE) or
	// the DeepSeek/GLM sigmoid-score + selection-bias path — routeExperts unifies both.
	var logits []float32
	if scr != nil {
		logits = scr.moeLogits
	} else {
		logits = make([]float32, nE)
	}
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
	// Weight residency: the router selection is the demand signal. Touch every chosen expert before the matmuls so the pager faults
	// them in and keeps resident RAM within budget (releasing the LRU tail). Bit-exact: released experts re-fault from the read-only
	// mapping (mmap mode) or are re-pread (pool mode). Lock/Unlock spans the touch and the matmul reads below it (swiGLUExpert); see
	// expertPager's doc for why pool mode needs it held that long.
	if pager != nil {
		pager.Lock()
		defer pager.Unlock()
		for _, e := range idx {
			pager.touch(unsafe.Pointer(&lw.Experts[e]))
		}
	}

	// Weighted sum of the chosen experts (each a SwiGLU MLP). Experts use the
	// MoE expert width (Mellum's moe_intermediate_size), not the dense one.
	hidden := arch.HiddenDim
	// One gate/up pair for the whole token: the experts run sequentially, so k pairs are never simultaneously live.
	sc := max(moe.SharedIntermediateDim, moe.IntermediateDim)
	var out, expOut, egate, eup []float32
	if scr != nil {
		out, expOut = scr.moeOut, scr.moeExpOut
		egate, eup = scr.moeGate[:sc], scr.moeUp[:sc]
	} else {
		out, expOut = make([]float32, hidden), make([]float32, hidden)
		egate, eup = make([]float32, sc), make([]float32, sc)
	}
	clear(out)
	// h is every routed and shared expert's gate and up input: quantize it once for the layer, not once in each of the 2k+2 matmuls.
	var hq *w4a8Act
	if scr != nil {
		hq = &scr.moeHQ
	} else {
		hq = new(w4a8Act)
	}
	if len(idx) > 0 {
		hq.prepare(be, &lw.Experts[idx[0]].Gate, h, 1)
	} else {
		hq.prepare(be, &lw.SharedExpert.Gate, h, 1)
	}
	for j, e := range idx {
		ex := &lw.Experts[e]
		swiGLUExpert(ex, h, expOut, moe.IntermediateDim, be, egate, eup, hq)
		addScaled(out, expOut, wts[j])
	}

	// Shared always-on expert (Qwen2-MoE / GLM). Qwen2 scales it by a per-token
	// sigmoid(SharedGate·h) gate; GLM/DeepSeek add it ungated.
	if moe.SharedIntermediateDim > 0 {
		swiGLUExpert(&lw.SharedExpert, h, expOut, moe.SharedIntermediateDim, be, egate, eup, hq)
		if moe.SharedUngated {
			addResidual(out, expOut)
		} else {
			var gl [1]float32
			matmul(be, &lw.SharedGate, h, gl[:], 1)
			g := float32(1.0 / (1.0 + math.Exp(-float64(gl[0])))) // sigmoid
			addScaled(out, expOut, g)
		}
	}
	return out, nil
}

// routeExperts selects the top-k experts and their weights from router logits. Mixtral/Qwen2-MoE: softmax over experts, top-k by
// probability, weights = those probabilities. DeepSeek/GLM (sigmoid=true): per-expert sigmoid scores; a bias
// (e_score_correction_bias), when present, shifts the top-k selection only, while the weights are the chosen experts' unbiased
// sigmoid scores. norm renormalizes the weights to sum 1; scale (routed_scaling_factor) multiplies them (0/1 = no-op). With
// sigmoid=false, bias=nil, scale in {0,1} and nGroup<=1 this is plain Mixtral routing.
//
// nGroup>1 (DeepSeek-V3 noaux_tc) adds group-limited selection: experts are partitioned into nGroup contiguous groups, each group
// scored by the sum of its top-2 selection scores; only experts in the top topkGroup groups are eligible for the per-token top-k.
func routeExperts(logits, bias []float32, k int, sigmoid, norm bool, scale float64, nGroup, topkGroup int) (idx []int, wts []float32) {
	var scoreBuf [128]float32
	var scores []float32
	if sigmoid {
		if len(logits) <= len(scoreBuf) {
			scores = scoreBuf[:len(logits)]
		} else {
			scores = make([]float32, len(logits))
		}
		for i, l := range logits {
			scores[i] = float32(1.0 / (1.0 + math.Exp(-float64(l))))
		}
	} else {
		scores = softmaxF32(logits)
	}
	var selBuf [128]float32
	var sel []float32
	if bias != nil {
		if len(scores) <= len(selBuf) {
			sel = selBuf[:len(scores)]
		} else {
			sel = make([]float32, len(scores))
		}
		for i := range scores {
			sel[i] = scores[i] + bias[i]
		}
	} else {
		sel = scores
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
	// Trailing experts beyond nGroup*gsz (when NumExperts % nGroup != 0) belong to no group and must be masked, not left at 0.0, or
	// they can outscore a legitimately -inf'd expert. Real DeepSeek divides evenly; this guards an odd config from mis-routing.
	for i := nGroup * gsz; i < len(sel); i++ {
		out[i] = negInf
	}
	return out
}

// activationFanoutThreshold gates parallelElementwise's fan-out: below this element count the fork/join (goroutine-wake
// stagger) costs more than a short shard's work, so the range runs serially. 8192 sits safely past the ambiguous band of the
// silubench microbenchmark rather than riding its edge. Re-measure with silubench before changing it, and do not derive a new
// value from the attention pool's stagger: this fan-out's per-worker body is a bare closure over a slice range, with a cheaper
// stagger.
const activationFanoutThreshold = 8192

// activationFanoutWorkers caps the fan-out at the P-core count, same reasoning as
// maxAttnWorkers (scratch.go) — reused directly rather than redefined, so a future change to the
// core count only has one place to update.
const activationFanoutWorkers = maxAttnWorkers

// parallelElementwise splits [0,n) into up to activationFanoutWorkers contiguous ranges and runs fn(lo,hi) on each in its own
// goroutine when n crosses activationFanoutThreshold; otherwise it runs fn(0,n) serially. Bit-identical either way by
// construction: fn must depend only on its own index range (no cross-range read, no shared accumulator). Every call site is a
// pure elementwise activation (silu/geluTanh × gate); a reduction (a softmax's `sum += e`) would reorder under this split and must
// not use it.
func parallelElementwise(n int, fn func(lo, hi int)) {
	if !activationFanoutEnabled || n < activationFanoutThreshold {
		fn(0, n)
		return
	}
	workers := min(activationFanoutWorkers, n)
	per := (n + workers - 1) / workers
	var wg sync.WaitGroup
	for w := range workers {
		lo, hi := w*per, min((w+1)*per, n)
		if lo >= hi {
			continue
		}
		wg.Add(1)
		go func(lo, hi int) {
			defer wg.Done()
			fn(lo, hi)
		}(lo, hi)
	}
	wg.Wait()
}

func swiglu(gate, up []float32) {
	n := len(gate)
	if n == 0 {
		return
	}
	u := up[:n]
	_ = gate[n-1]
	_ = u[n-1]
	i := 0
	for ; i+3 < n; i += 4 {
		_ = gate[i+3]
		_ = u[i+3]
		g0, g1, g2, g3 := gate[i], gate[i+1], gate[i+2], gate[i+3]
		gate[i] = silu(g0) * u[i]
		gate[i+1] = silu(g1) * u[i+1]
		gate[i+2] = silu(g2) * u[i+2]
		gate[i+3] = silu(g3) * u[i+3]
	}
	for ; i < n; i++ {
		gate[i] = silu(gate[i]) * u[i]
	}
}

func geglu(gate, up []float32) {
	n := len(gate)
	if n == 0 {
		return
	}
	u := up[:n]
	_ = gate[n-1]
	_ = u[n-1]
	i := 0
	for ; i+3 < n; i += 4 {
		_ = gate[i+3]
		_ = u[i+3]
		g0, g1, g2, g3 := gate[i], gate[i+1], gate[i+2], gate[i+3]
		gate[i] = geluTanh(g0) * u[i]
		gate[i+1] = geluTanh(g1) * u[i+1]
		gate[i+2] = geluTanh(g2) * u[i+2]
		gate[i+3] = geluTanh(g3) * u[i+3]
	}
	for ; i < n; i++ {
		gate[i] = geluTanh(gate[i]) * u[i]
	}
}

// gegluExact is geglu with the exact erf GELU (geluErf) instead of the tanh approximation: Spark-X2.5's gated MLP,
// down(gelu(gate(x)) * up(x)), whose vendor code requires hidden_act == "gelu" (HF's exact "gelu", not "gelu_new" or
// "gelu_pytorch_tanh"). ActGelu also reaches the non-gated path (nonGatedMLP, Nemotron-H).
func gegluExact(gate, up []float32) {
	n := len(gate)
	if n == 0 {
		return
	}
	u := up[:n]
	_ = gate[n-1]
	_ = u[n-1]
	i := 0
	for ; i+3 < n; i += 4 {
		_ = gate[i+3]
		_ = u[i+3]
		g0, g1, g2, g3 := gate[i], gate[i+1], gate[i+2], gate[i+3]
		gate[i] = geluErf(g0) * u[i]
		gate[i+1] = geluErf(g1) * u[i+1]
		gate[i+2] = geluErf(g2) * u[i+2]
		gate[i+3] = geluErf(g3) * u[i+3]
	}
	for ; i < n; i++ {
		gate[i] = geluErf(gate[i]) * u[i]
	}
}

// swiGLUExpert evaluates one gated (SwiGLU) expert MLP of the given intermediate width into dst[:hidden]:
// dst = Down·(silu(Gate·h) ⊙ Up·h). gate/up come from the caller so a token's k experts share one pair; the two matmuls fully
// overwrite them before anything reads them, so reuse carries no state between experts and the result is bit-identical. A nil or
// short buffer allocates, so callers with no scratch (the llama4 path) work unchanged. hq, when non-nil, is h already quantized
// for the CPU W4A8 path (moeMLP quantizes it once for every expert of the layer); with hq nil, gate and up still share one
// quantization of h.
func swiGLUExpert(ex *expertWeights, h, dst []float32, inter int, be Backend, gate, up []float32, hq *w4a8Act) {
	if cap(gate) < inter || cap(up) < inter {
		gate, up = make([]float32, inter), make([]float32, inter)
	}
	gate, up = gate[:inter], up[:inter]
	if hq == nil {
		hq = new(w4a8Act)
		hq.prepare(be, &ex.Gate, h, 1)
	}
	matmulPre(be, &ex.Gate, hq, h, gate, 1)
	matmulPre(be, &ex.Up, hq, h, up, 1)
	if len(gate) < activationFanoutThreshold {
		swiglu(gate, up)
	} else {
		parallelElementwise(len(gate), func(lo, hi int) {
			swiglu(gate[lo:hi], up[lo:hi])
		})
	}
	matmul(be, &ex.Down, gate, dst, 1)
}

// softmaxF32 returns the softmax of xs (float64 accumulation, max-shifted for
// stability). Small (NumExperts) so allocation is cheap.
func softmaxF32(xs []float32) []float32 {
	if len(xs) == 0 {
		return nil
	}
	maxv := xs[0]
	i := 1
	_ = xs[len(xs)-1]
	for ; i+3 < len(xs); i += 4 {
		v0, v1, v2, v3 := xs[i], xs[i+1], xs[i+2], xs[i+3]
		m0 := max(v0, v1)
		m1 := max(v2, v3)
		m := max(m0, m1)
		if m > maxv {
			maxv = m
		}
	}
	for ; i < len(xs); i++ {
		if xs[i] > maxv {
			maxv = xs[i]
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
	_ = out[len(out)-1]
	j := 0
	for ; j+3 < len(out); j += 4 {
		out[j] *= inv
		out[j+1] *= inv
		out[j+2] *= inv
		out[j+3] *= inv
	}
	for ; j < len(out); j++ {
		out[j] *= inv
	}
	return out
}

// topK returns the indices and values of the k largest entries of xs, in
// descending order. O(k·n) selection — k and n (NumExperts) are tiny.
func topK(xs []float32, k int) ([]int, []float32) {
	idx := make([]int, 0, k)
	val := make([]float32, 0, k)
	var usedBuf [128]bool
	var used []bool
	if len(xs) <= len(usedBuf) {
		used = usedBuf[:len(xs)]
	} else {
		used = make([]bool, len(xs))
	}
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
	case ActGelu: // HF's "gelu" — the exact erf function, NOT a spelling of gelu_new
		for i := range mid {
			mid[i] = geluErf(mid[i])
		}
	default:
		return nil, fmt.Errorf("decoder: unsupported non-gated activation %d (have gelu-tanh, gelu)", arch.Act)
	}
	out := make([]float32, hidden)
	matmul(be, &lw.DownProj, mid, out, 1)
	if lw.DownBias != nil {
		addBias(out, lw.DownBias)
	}
	return out, nil
}

// gatedMLP runs one block's gated MLP for the current position into out (the caller applies the post-MLP norm and residual).
// GeGLU (Gemma) and SwiGLU (Llama/Mistral/Qwen) share the structure and differ only in the gate activation (Architecture.Act).
// No biases on any projection.
//
//	gate = GateProj·h            // [IntermediateDim]
//	up   = UpProj·h              // [IntermediateDim]
//	mid  = act(gate) ⊙ up        // [IntermediateDim]
//	out  = DownProj·mid          // [HiddenDim]
func gatedMLP(h, out []float32, lw *LayerWeights, arch *Architecture, be Backend, scr *decodeScratch, lora *loraLayerDelta) error {
	gate, up := scr.gate, scr.up // [inter] scratch; matmul fully overwrites each
	fusedAct := false            // set when the fused path already applied the activation
	if cpuFusedGateUp && lora == nil {
		var dt0 time.Time
		if decodeTiming {
			dt0 = time.Now()
		}
		if gatedMLPFusedGateUp(h, lw, arch, scr) {
			fusedAct = true
			if decodeTiming {
				atomic.AddInt64(&dtGU, int64(time.Since(dt0)))
			}
		}
	}
	if fusedAct {
		// gate/up + activation already done in one fork/join
	} else if isW8A8(&lw.GateProj) && isW8A8(&lw.UpProj) {
		scr.gateUpOps[0] = linalg.W8A8Op{BQ: wmInt8(&lw.GateProj), Scales: wmScales(&lw.GateProj), Dst: gate, N: lw.GateProj.Rows()}
		scr.gateUpOps[1] = linalg.W8A8Op{BQ: wmInt8(&lw.UpProj), Scales: wmScales(&lw.UpProj), Dst: up, N: lw.UpProj.Rows()}
		matmulW8A8Batch(be, scr.ws, h, 1, lw.GateProj.Cols(), scr.gateUpOps[:], lw.GateProj.ActQuantGroup()) // gate/up in one dispatch (GPU: one submit)
	} else if w4a8BatchEnabled && isW4A8(&lw.GateProj) && isW4A8(&lw.UpProj) {
		// Fused gate/up W4A8, mirroring the W8A8 batch above.
		var group int
		scr.guOpsW4[0], group = wmW4A8Op(&lw.GateProj, gate)
		scr.guOpsW4[1], _ = wmW4A8Op(&lw.UpProj, up)
		scr.ws.SetThreshold(int4ParThreshold)
		matmulW4A8Batch(be, scr.ws, h, 1, lw.GateProj.Cols(), group, scr.guOpsW4[:], lw.GateProj.ActQuantGroup())
	} else {
		var dt0 time.Time
		if decodeTiming {
			dt0 = time.Now()
		}
		scr.hq.prepare(be, &lw.GateProj, h, 1) // one quantization of h for gate and up (R-13)
		matmulIntoPre(scr.ws, be, &lw.GateProj, &scr.hq, h, gate, 1)
		matmulIntoPre(scr.ws, be, &lw.UpProj, &scr.hq, h, up, 1)
		if decodeTiming {
			atomic.AddInt64(&dtGU, int64(time.Since(dt0)))
		}
	}
	var dt1 time.Time
	if decodeTiming {
		dt1 = time.Now()
		defer func() { atomic.AddInt64(&dtActDown, int64(time.Since(dt1))) }()
	}
	if fusedAct {
		matmulInto(scr.ws, be, &lw.DownProj, gate, out, 1)
		return nil
	}
	if lora != nil { // compute-time LoRA (#7): delta into gate/up before the activation
		applyLoRA(lora.gate, h, gate, scr)
		applyLoRA(lora.up, h, up, scr)
	}
	switch arch.Act {
	case ActGeluTanh:
		if len(gate) < activationFanoutThreshold {
			geglu(gate, up)
		} else {
			parallelElementwise(len(gate), func(lo, hi int) {
				geglu(gate[lo:hi], up[lo:hi])
			})
		}
	case ActSiLU:
		if len(gate) < activationFanoutThreshold {
			swiglu(gate, up)
		} else {
			parallelElementwise(len(gate), func(lo, hi int) {
				swiglu(gate[lo:hi], up[lo:hi])
			})
		}
	case ActGelu:
		if len(gate) < activationFanoutThreshold {
			gegluExact(gate, up)
		} else {
			parallelElementwise(len(gate), func(lo, hi int) {
				gegluExact(gate[lo:hi], up[lo:hi])
			})
		}
	default:
		return fmt.Errorf("decoder: unsupported activation %d (have GeGLU/SwiGLU/exact-GELU-GLU)", arch.Act)
	}
	if decodeTiming {
		atomic.AddInt64(&dtAct, int64(time.Since(dt1)))
	}
	matmulInto(scr.ws, be, &lw.DownProj, gate, out, 1) // [1,hidden] = mid · DownProjᵀ (gate now holds the activated mid)
	if lora != nil {
		applyLoRA(lora.down, gate, out, scr)
	}
	return nil
}

// moeExpertMajorChunk is the row count per expert-major chunk. The expert-major path (moeMLPBatch) turns the M=1 matmuls moeMLP
// issues per row into M=N per expert, so an expert's weights are read once per chunk rather than once per token routed to it.
//
// Bit-identity is the constraint, and it holds on two counts. The matmuls are M-invariant (linalg.MatmulBT's contract: a row
// computed alone equals the same row inside a batch; weightmat.go says the same of the int4 W4A8 kernel). And the per-row
// accumulation order is preserved: moeMLP folds the k experts in routing-rank order and float addition is not associative, so
// moeMLPBatch computes every (row, rank) expert output first and then folds each row in rank order. That is also why it holds a
// [rows][k][hidden] buffer and runs in chunks rather than over the whole prompt (604 MB at K=8192, k=8, hidden=2304).
const moeExpertMajorChunk = 512

// knobSet.moeExpertMajor (knobs.go) reports whether the expert-major prefill path is on; GOINFER_MOE_EXPERT_MAJOR=0 restores the
// per-row path, as an escape hatch and A/B handle rather than a user setting. It is bit-identical, so unlike
// --cpu-fast-attention it changes speed and nothing else. TestMoEExpertMajor_bitIdentical asserts equality on every logit through
// the real forward, and asserts non-vacuity via the chunk counter (moeExpertMajorRuns) so a silent refusal cannot pass as green.
// Measurements: docs/code-notes/decoder.md#knobSet.moeExpertMajor.

// moeExpertMajorRuns counts chunks that took the expert-major path, so the bit-identity gate can prove it is not vacuous:
// moeMLPBatch refuses for several legitimate reasons (shared expert, live pager, test seams), and a refusal makes both arms take
// the identical per-row path, so the test would pass while proving nothing.
var moeExpertMajorRuns int64

// moeMLPBatch runs the MoE FFN over rows ([n, hidden]) expert-major, writing n*hidden results into dst; bit-identical to calling
// moeMLP per row. It refuses (returns false) for the cases whose observable behaviour is order-dependent rather than
// value-dependent, which preserving the accumulation order cannot make identical:
//
//   - moeSelOverride / moeSelTrace: test seams keyed on per-call forward order.
//   - a live pager: touch order is the demand signal that drives eviction, so reordering it changes which experts are resident.
//     Expert-major is likely better for paging, but different is not better until measured.
func moeMLPBatch(rows []float32, n int, lw *LayerWeights, arch *Architecture, be Backend, pager *expertPager, dst []float32) (bool, error) {
	moe := arch.MoE
	if moe == nil || len(lw.Experts) == 0 || moeSelOverride != nil || moeSelTrace != nil || pager != nil {
		return false, nil
	}
	if arch.Act != ActSiLU {
		return false, nil
	}
	// The shared expert is added after the routed fold in moeMLP. Refuse up front so a fall-back costs nothing: a partial
	// implementation that dropped it would change results, not speed.
	if moe.SharedIntermediateDim > 0 {
		return false, nil
	}
	hidden, nE, k := arch.HiddenDim, moe.NumExperts, moe.TopK
	inter := moe.IntermediateDim

	// Router for the whole chunk in one matmul (M=n), instead of a matmul and an nE-sized allocation per row.
	logits := make([]float32, n*nE)
	matmul(be, &lw.Router, rows, logits, n)

	idxs := make([][]int, n)
	wtss := make([][]float32, n)
	// expert -> the (row, rank) pairs that selected it
	type slot struct{ row, rank int }
	byExpert := make([][]slot, nE)
	for r := range n {
		idx, wts := routeExperts(logits[r*nE:(r+1)*nE], lw.RouterBias, k,
			moe.RouterSigmoid, moe.NormTopKProb, moe.RoutedScale, moe.NGroup, moe.TopkGroup)
		idxs[r], wtss[r] = idx, wts
		for j, e := range idx {
			byExpert[e] = append(byExpert[e], slot{r, j})
		}
	}

	// Every (row, rank) expert output, so the fold below can run in rank order.
	perRank := make([]float32, n*k*hidden)
	maxRows := 0
	for _, s := range byExpert {
		if len(s) > maxRows {
			maxRows = len(s)
		}
	}
	if maxRows == 0 {
		return false, nil
	}
	gathered := make([]float32, maxRows*hidden)
	gate := make([]float32, maxRows*inter)
	up := make([]float32, maxRows*inter)
	outBuf := make([]float32, maxRows*hidden)

	for e, slots := range byExpert {
		if len(slots) == 0 {
			continue
		}
		m := len(slots)
		for i, s := range slots { // GATHER this expert's rows
			copy(gathered[i*hidden:(i+1)*hidden], rows[s.row*hidden:(s.row+1)*hidden])
		}
		ex := &lw.Experts[e]
		matmul(be, &ex.Gate, gathered[:m*hidden], gate[:m*inter], m)
		matmul(be, &ex.Up, gathered[:m*hidden], up[:m*inter], m)
		if m*inter < activationFanoutThreshold {
			swiglu(gate[:m*inter], up[:m*inter])
		} else {
			parallelElementwise(m*inter, func(lo, hi int) {
				swiglu(gate[lo:hi], up[lo:hi])
			})
		}
		matmul(be, &ex.Down, gate[:m*inter], outBuf[:m*hidden], m)
		for i, s := range slots { // SCATTER into (row, rank)
			copy(perRank[(s.row*k+s.rank)*hidden:(s.row*k+s.rank+1)*hidden], outBuf[i*hidden:(i+1)*hidden])
		}
	}

	atomic.AddInt64(&moeExpertMajorRuns, 1)
	// Fold each row in ROUTING RANK order -- the sequence moeMLP uses, which is
	// what makes this bit-identical rather than merely close.
	for r := range n {
		o := dst[r*hidden : (r+1)*hidden]
		clear(o)
		for j := range idxs[r] {
			w := wtss[r][j]
			p := perRank[(r*k+j)*hidden : (r*k+j+1)*hidden]
			addScaled(o, p, w)
		}
	}
	return true, nil
}

// addScaled computes dst[i] += scale * src[i] over len(dst) elements.
func addScaled(dst, src []float32, scale float32) {
	if len(dst) == 0 {
		return
	}
	_ = dst[len(dst)-1]
	_ = src[len(dst)-1]
	i := 0
	for ; i+3 < len(dst); i += 4 {
		dst[i] += scale * src[i]
		dst[i+1] += scale * src[i+1]
		dst[i+2] += scale * src[i+2]
		dst[i+3] += scale * src[i+3]
	}
	for ; i < len(dst); i++ {
		dst[i] += scale * src[i]
	}
}
