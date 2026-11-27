package cpuengine

import (
	"github.com/townsendmerino/goinfer/internal/runtime/archparts"

	"math"
	"slices"
	"sync"
	"unsafe"

	darch "github.com/townsendmerino/goinfer/decoder/arch"
	"github.com/townsendmerino/goinfer/internal/runtime/normact"
)

// Gemma 4 26B-A4B parallel dense-MLP + MoE FFN sub-block (enable_moe_block). The two branches run on the same post-attention
// residual h but through different normalizations; their outputs are joint-normed then residual-added, and a per-layer scalar
// multiplies the whole layer output. Pinned against transformers 5.12.0's Gemma4TextDecoderLayer.Forward.
//
// The router and experts are Gemma-4-specific (a weightless-norm + learned-scale pre-projection, an unconditional renorm, a
// per-expert scale, and gelu-tanh GeGLU experts), so this does not reuse the SiLU moeMLP/swiGLUExpert.

// Gemma4MoEFFN applies the sub-block to one token's post-attention residual h ([hidden]) and returns the layer output ([hidden]).
// It is position-independent, so the decode loop calls it per token.
//
// The router needs only h and the dense branch needs only h, so the two are independent until the join. The router runs first (to
// learn which experts to fetch), the expert fills are issued, and the dense branch's matmuls run on the calling goroutine while a
// second goroutine drives the fills, hiding a cold miss's fault/pread latency behind the dense compute.
// The result is bit-identical either way: the branches touch disjoint memory
// and are summed order-independently at the join (TestGemma4MoEFFN_overlapBitIdentical), and the fill goroutine never touches
// `be`, only the pager.
func Gemma4MoEFFN(be Backend, arch *darch.Architecture, h []float32, w *archparts.Gemma4MoEWeights, pager ExpertPager) []float32 {
	return gemma4MoEFFNWith(be, arch.HiddenDim, arch.NormEps, arch.RMSAddOne, h, w, pager, nil)
}

// gemma4MoEScratch is gemma4MoEFFNWith's working set, reused call after call by one stream (sequentially: the fill goroutine
// never touches it). Every buffer is written or cleared before it is read.
type gemma4MoEScratch struct {
	rn, scores, wts, xd, gate, up, x1, xe, x2, gu, mid, edown, comb, out []float32
}

// gemma4MoEFFNWith is Gemma4MoEFFN with the norms' epsilon and (1+w) given: the program walker's form (opGemma4MoE). The weighted
// norms follow addOne exactly like the dense layer's norms; the router's norm (rmsNormNoWeight below) is genuinely weightless. Its
// buffers come from sc (nil: fresh ones), so the returned slice is sc's and valid until the next call with it.
func gemma4MoEFFNWith(be Backend, hidden int, eps float64, addOne bool, h []float32, w *archparts.Gemma4MoEWeights, pager ExpertPager, sc *gemma4MoEScratch) []float32 {
	if sc == nil {
		sc = &gemma4MoEScratch{}
	}

	// moe branch — router on the RAW residual h (its own weightless RMSNorm + learned
	// scale + hidden^-0.5), softmax over all experts → top-k → UNCONDITIONAL renorm →
	// per-expert scale. Computed BEFORE the dense branch so idx is known in time to
	// issue the fills the dense branch's compute below will overlap.
	rn := scratchF32(&sc.rn, len(h))
	copy(rn, h)
	rmsNormNoWeight(rn, 1, hidden, eps)
	root := float32(math.Pow(float64(hidden), -0.5))
	for i := range rn {
		rn[i] = rn[i] * w.RouterScale[i] * root
	}
	scores := scratchF32(&sc.scores, w.NE)
	Matmul(be, &w.RouterProj, rn, scores, 1)
	probs := SoftmaxF32(scores)
	idx, topv := TopK(probs, w.TopK)
	if RouterCapture { // DIAGNOSTIC (default-off, observe-only): record the selected experts; see routercapture.go
		RouterCaptureDo(func() { RouterCaptureBuf = append(RouterCaptureBuf, slices.Clone(idx)) })
		// rn is the FINALIZED router input (weightless-norm · routerScale · hidden^-0.5) that feeds
		// routerProj — captured so a resident-router unit test can replay the SAME input through the
		// CUDA selection kernels and gate its idx against these (RouterCaptureBuf), isolating a routing
		// flip from any expert-GEMV numeric difference. Copy: rn is mutated/reused after this.
		RouterCaptureDo(func() { RouterRnBuf = append(RouterRnBuf, slices.Clone(rn)) })
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
		RouterCaptureDo(func() { RouterMarginBuf = append(RouterMarginBuf, minSel-maxRej) })
	}
	// Weight residency: the router selection is the demand signal. Touch each chosen expert so the pager faults it in and evicts the
	// LRU tail to stay within budget, keyed by the gateUp element address (newExpertPager's key). Bit-exact: released experts
	// re-fault from the read-only mapping (mmap mode) or are re-pread (pool mode). The fills are issued on a separate goroutine so
	// they run while the dense branch below computes on the calling goroutine; Lock/Unlock still spans the touch and the Matmul reads
	// further below (the expert loop over w.expertsGateUp/expertsDown), across the wg.Wait() boundary (see expertPager's doc for why
	// pool mode needs the lock held that long).
	var fillWG sync.WaitGroup
	if pager != nil {
		pager.Lock()
		defer pager.Unlock()
		fillWG.Go(func() {
			for _, e := range idx {
				pager.Touch(unsafe.Pointer(&w.ExpertsGateUp[e]))
			}
		})
	}
	var sum float32
	for _, v := range topv {
		sum += v
	}
	wts := scratchF32(&sc.wts, len(idx))
	for j, e := range idx {
		wts[j] = (topv[j] / sum) * w.PerExpertScale[e]
	}
	if RouterCapture {
		RouterCaptureDo(func() { RouterWtsBuf = append(RouterWtsBuf, slices.Clone(wts)) })
	}

	// dense branch: x1 = post_ffn_norm_1(mlp(pre_ffn_norm(h))), gelu-tanh GeGLU. It runs after the fills are issued and before they
	// are waited on, so its compute overlaps the fill goroutine's I/O. It touches only dense weights (never paged) and sc's
	// buffers, disjoint from what the fill goroutine touches, so it is safe without a lock of its own.
	xd := scratchF32(&sc.xd, len(h))
	copy(xd, h)
	normact.RMS(xd, w.PreFFNNorm, 1, hidden, eps, addOne)
	gate := scratchF32(&sc.gate, w.DenseInter)
	up := scratchF32(&sc.up, w.DenseInter)
	Matmul(be, &w.MlpGate, xd, gate, 1)
	Matmul(be, &w.MlpUp, xd, up, 1)
	for i := range gate {
		gate[i] = normact.GELUTanh(gate[i]) * up[i]
	}
	x1 := scratchF32(&sc.x1, hidden)
	Matmul(be, &w.MlpDown, gate, x1, 1)
	normact.RMS(x1, w.PostFFNNorm1, 1, hidden, eps, addOne)
	if RouterCapture {
		RouterCaptureDo(func() { RouterX1Buf = append(RouterX1Buf, slices.Clone(x1)) })
	}

	// The moe branch below reads w.expertsGateUp/expertsDown, which the fill goroutine
	// may still be repointing (pool mode) or fault-servicing (mmap mode) — wait for it
	// before touching them. A no-op wait when pager == nil (fillWG never Add'd).
	fillWG.Wait()

	// experts on pre_ffn_norm_2(h): gelu-tanh GeGLU, gate/up = contiguous halves.
	xe := scratchF32(&sc.xe, len(h))
	copy(xe, h)
	normact.RMS(xe, w.PreFFNNorm2, 1, hidden, eps, addOne)
	x2 := scratchF32(&sc.x2, hidden)
	clear(x2)
	gu := scratchF32(&sc.gu, 2*w.MoeInter)
	mid := scratchF32(&sc.mid, w.MoeInter)
	edown := scratchF32(&sc.edown, hidden)
	for j, e := range idx {
		Matmul(be, &w.ExpertsGateUp[e], xe, gu, 1)
		for i := 0; i < w.MoeInter; i++ {
			mid[i] = normact.GELUTanh(gu[i]) * gu[w.MoeInter+i]
		}
		Matmul(be, &w.ExpertsDown[e], mid, edown, 1)
		wj := wts[j]
		for i := range x2 {
			x2[i] += wj * edown[i]
		}
	}
	normact.RMS(x2, w.PostFFNNorm2, 1, hidden, eps, addOne)
	if RouterCapture {
		RouterCaptureDo(func() { RouterX2Buf = append(RouterX2Buf, slices.Clone(x2)) })
	}

	// join: out = (h + post_ffn_norm(x1 + x2)) * layer_scalar.
	comb := scratchF32(&sc.comb, hidden)
	for i := range comb {
		comb[i] = x1[i] + x2[i]
	}
	normact.RMS(comb, w.PostFFNNorm, 1, hidden, eps, addOne)
	out := scratchF32(&sc.out, hidden)
	for i := range out {
		out[i] = (h[i] + comb[i]) * w.LayerScalar
	}
	return out
}
