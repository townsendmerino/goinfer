package decoder

import (
	"math"
	"sync"
	"unsafe"

	"github.com/townsendmerino/aikit/linalg"
)

// Gemma 4 26B-A4B parallel dense-MLP + MoE FFN sub-block (enable_moe_block). The
// two branches run on the SAME post-attention residual h but through DIFFERENT
// normalizations, and their outputs are joint-normed then residual-added; a
// per-layer scalar multiplies the whole layer output. Pinned against transformers
// 5.12.0's Gemma4TextDecoderLayer.forward — see docs/task-gemma4-moe.md Phase 1a.
//
// The router and experts are Gemma-4-specific (a weightless-norm + learned-scale
// pre-projection, an unconditional renorm, a per-expert scale, and gelu-tanh GeGLU
// experts), so this does NOT reuse the SiLU moeMLP/swiGLUExpert.

// gemma4MoEWeights is one layer's FFN sub-block, as WeightMats (matmul path) + f32
// norm/scale vectors. Built by the loader (production) or a golden (the op-test).
type gemma4MoEWeights struct {
	preFFNNorm   []float32 // pre_feedforward_layernorm   (dense branch pre-norm)
	postFFNNorm1 []float32 // post_feedforward_layernorm_1 (dense branch post-norm)
	preFFNNorm2  []float32 // pre_feedforward_layernorm_2  (moe branch pre-norm)
	postFFNNorm2 []float32 // post_feedforward_layernorm_2 (moe branch post-norm)
	postFFNNorm  []float32 // post_feedforward_layernorm   (joint post-norm on x1+x2)

	mlpGate, mlpUp, mlpDown linalg.WeightMat // dense MLP at denseInter

	routerProj     linalg.WeightMat // [nE, hidden]
	routerScale    []float32        // [hidden] learned pre-projection scale
	perExpertScale []float32        // [nE] learned scale on the renormalized top-k weights

	expertsGateUp []linalg.WeightMat // per expert [2*moeInter, hidden] (gate‖up, contiguous)
	expertsDown   []linalg.WeightMat // per expert [hidden, moeInter]

	layerScalar                    float32
	denseInter, moeInter, nE, topK int
}

// gemma4MoEFFN applies the sub-block to one token's post-attention residual h
// ([hidden]) and returns the layer output ([hidden]). Position-independent, so the
// decode loop calls it per token.
//
// Lever 3 (task-moe-streaming.md, "overlap routed reads with the resident branch"):
// the router only needs h (not the dense branch's output), and the dense branch only
// needs h (not the router's chosen experts) — the two are independent until the join.
// So the router runs FIRST (to learn which experts to fetch), the expert fills are
// ISSUED, and the dense branch's matmuls run on the calling goroutine WHILE a second
// goroutine drives the fills — hiding a cold miss's fault/pread latency behind the
// dense branch's own compute time instead of paying both serially. Bit-identical
// either way (dense and moe branches touch disjoint memory and are summed
// order-independently at the join; TestGemma4MoEFFN_overlapBitIdentical), backend
// calls untouched (the fill goroutine never touches `be`, only the pager).
func gemma4MoEFFN(be Backend, arch *Architecture, h []float32, w *gemma4MoEWeights, pager *expertPager) []float32 {
	hidden := arch.HiddenDim
	eps := arch.NormEps
	// The weighted norms follow arch.RMSAddOne (the (1+w) offset) exactly like the
	// dense forward's normalize(arch, …), so if that flag is ever flipped for gemma4
	// these five track it rather than silently diverging. false today (Gemma4RMSNorm
	// is plain x*w); the router's norm at rmsNormNoWeight below is genuinely weightless.
	addOne := arch.RMSAddOne

	// moe branch — router on the RAW residual h (its own weightless RMSNorm + learned
	// scale + hidden^-0.5), softmax over all experts → top-k → UNCONDITIONAL renorm →
	// per-expert scale. Computed BEFORE the dense branch so idx is known in time to
	// issue the fills the dense branch's compute below will overlap.
	rn := append([]float32(nil), h...)
	rmsNormNoWeight(rn, 1, hidden, eps)
	root := float32(math.Pow(float64(hidden), -0.5))
	for i := range rn {
		rn[i] = rn[i] * w.routerScale[i] * root
	}
	scores := make([]float32, w.nE)
	matmul(be, &w.routerProj, rn, scores, 1)
	probs := softmaxF32(scores)
	idx, topv := topK(probs, w.topK)
	if routerCapture { // DIAGNOSTIC (default-off, observe-only): record the selected experts; see routercapture.go
		routerCaptureDo(func() { routerCaptureBuf = append(routerCaptureBuf, append([]int(nil), idx...)) })
		// rn is the FINALIZED router input (weightless-norm · routerScale · hidden^-0.5) that feeds
		// routerProj — captured so a resident-router unit test can replay the SAME input through the
		// CUDA selection kernels and gate its idx against these (routerCaptureBuf), isolating a routing
		// flip from any expert-GEMV numeric difference. Copy: rn is mutated/reused after this.
		routerCaptureDo(func() { routerRnBuf = append(routerRnBuf, append([]float32(nil), rn...)) })
		// top-k boundary margin: min selected prob − max rejected prob (the gap a quant flip must cross).
		sel := map[int]bool{}
		for _, e := range idx {
			sel[e] = true
		}
		minSel, maxRej := float32(1), float32(0)
		for e, p := range probs {
			if sel[e] {
				if p < minSel {
					minSel = p
				}
			} else if p > maxRej {
				maxRej = p
			}
		}
		routerCaptureDo(func() { routerMarginBuf = append(routerMarginBuf, minSel-maxRej) })
	}
	// Weight residency (idea #2): the router selection is the demand signal. Touch each
	// chosen expert so the pager faults it in and evicts the LRU tail to stay within
	// budget. Keyed by the gateUp element address (newExpertPager's key). Bit-exact —
	// released experts re-fault from the read-only mapping (mmap mode) or are re-pread
	// (pool mode). Issued on a separate goroutine (Lever 3) so the fills run WHILE the
	// dense branch below computes on the calling goroutine, instead of paying both
	// serially; Lock/Unlock still spans touch AND the matmul reads further below (the
	// expert loop over w.expertsGateUp/expertsDown), now across the wg.Wait() boundary
	// — see expertPager's doc comment for why pool mode needs the lock held that long.
	var fillWG sync.WaitGroup
	if pager != nil {
		pager.Lock()
		defer pager.Unlock()
		fillWG.Add(1)
		go func() {
			defer fillWG.Done()
			for _, e := range idx {
				pager.touch(unsafe.Pointer(&w.expertsGateUp[e]))
			}
		}()
	}
	var sum float32
	for _, v := range topv {
		sum += v
	}
	wts := make([]float32, len(idx))
	for j, e := range idx {
		wts[j] = (topv[j] / sum) * w.perExpertScale[e]
	}
	if routerCapture {
		routerCaptureDo(func() { routerWtsBuf = append(routerWtsBuf, append([]float32(nil), wts...)) })
	}

	// dense branch: x1 = post_ffn_norm_1( mlp( pre_ffn_norm(h) ) ), gelu-tanh GeGLU.
	// Runs here — after the fills are issued, before they're waited on — so its compute
	// overlaps the fill goroutine's I/O (Lever 3). Touches only dense weights (never
	// paged) and fresh local buffers, disjoint from what the fill goroutine touches, so
	// this is safe without any lock of its own.
	xd := append([]float32(nil), h...)
	rmsNorm(xd, w.preFFNNorm, 1, hidden, eps, addOne)
	gate := make([]float32, w.denseInter)
	up := make([]float32, w.denseInter)
	matmul(be, &w.mlpGate, xd, gate, 1)
	matmul(be, &w.mlpUp, xd, up, 1)
	for i := range gate {
		gate[i] = geluTanh(gate[i]) * up[i]
	}
	x1 := make([]float32, hidden)
	matmul(be, &w.mlpDown, gate, x1, 1)
	rmsNorm(x1, w.postFFNNorm1, 1, hidden, eps, addOne)
	if routerCapture {
		routerCaptureDo(func() { routerX1Buf = append(routerX1Buf, append([]float32(nil), x1...)) })
	}

	// The moe branch below reads w.expertsGateUp/expertsDown, which the fill goroutine
	// may still be repointing (pool mode) or fault-servicing (mmap mode) — wait for it
	// before touching them. A no-op wait when pager == nil (fillWG never Add'd).
	fillWG.Wait()

	// experts on pre_ffn_norm_2(h): gelu-tanh GeGLU, gate/up = contiguous halves.
	xe := append([]float32(nil), h...)
	rmsNorm(xe, w.preFFNNorm2, 1, hidden, eps, addOne)
	x2 := make([]float32, hidden)
	gu := make([]float32, 2*w.moeInter)
	mid := make([]float32, w.moeInter)
	edown := make([]float32, hidden)
	for j, e := range idx {
		matmul(be, &w.expertsGateUp[e], xe, gu, 1)
		for i := 0; i < w.moeInter; i++ {
			mid[i] = geluTanh(gu[i]) * gu[w.moeInter+i]
		}
		matmul(be, &w.expertsDown[e], mid, edown, 1)
		wj := wts[j]
		for i := range x2 {
			x2[i] += wj * edown[i]
		}
	}
	rmsNorm(x2, w.postFFNNorm2, 1, hidden, eps, addOne)
	if routerCapture {
		routerCaptureDo(func() { routerX2Buf = append(routerX2Buf, append([]float32(nil), x2...)) })
	}

	// join: out = (h + post_ffn_norm(x1 + x2)) * layer_scalar.
	comb := make([]float32, hidden)
	for i := range comb {
		comb[i] = x1[i] + x2[i]
	}
	rmsNorm(comb, w.postFFNNorm, 1, hidden, eps, addOne)
	out := make([]float32, hidden)
	for i := range out {
		out[i] = (h[i] + comb[i]) * w.layerScalar
	}
	return out
}
