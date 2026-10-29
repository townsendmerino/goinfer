package decoder

import (
	"fmt"
	"os"
	"sync/atomic"
	"time"

	"github.com/townsendmerino/aikit/linalg"
	prog "github.com/townsendmerino/goinfer/internal/program"
)

// deltaNetTiming (GOINFER_DELTANET_TIMING=1) is a diagnostic that splits gatedDeltaNetStep's cost into the three dominant
// projections, the delta-rule recurrence proper (section 3 below, plain scalar Go) and everything else (conv, gates, gated
// RMSNorm). The accumulators are atomics, not Generate-loop locals, so concurrent decode streams do not race on them.
var deltaNetTiming = os.Getenv("GOINFER_DELTANET_TIMING") != ""

var (
	dnProjNs     atomic.Int64
	dnRecurNs    atomic.Int64
	dnOtherNs    atomic.Int64
	dnStateElems atomic.Int64 // Σ (nv·hk·hv) over every step — the recurrence's own rate-check denominator
	dnCalls      atomic.Int64
)

func deltaNetTimers() *prog.DeltaNetTimers {
	if !deltaNetTiming {
		return nil
	}
	return &prog.DeltaNetTimers{
		ProjNs:     &dnProjNs,
		RecurNs:    &dnRecurNs,
		OtherNs:    &dnOtherNs,
		StateElems: &dnStateElems,
		Calls:      &dnCalls,
	}
}

// PrintDeltaNetTiming reports the accumulated split (no-op, zero cost, if
// GOINFER_DELTANET_TIMING was never set or gatedDeltaNetStep was never called).
// Call after a real decode run.
func PrintDeltaNetTiming() {
	if !deltaNetTiming {
		return
	}
	n := dnCalls.Load()
	if n == 0 {
		return
	}
	proj := time.Duration(dnProjNs.Load())
	recur := time.Duration(dnRecurNs.Load())
	other := time.Duration(dnOtherNs.Load())
	total := proj + recur + other
	elems := dnStateElems.Load()
	msPer := func(d time.Duration) float64 { return float64(d.Microseconds()) / 1000 / float64(n) }
	pct := func(d time.Duration) float64 {
		if total == 0 {
			return 0
		}
		return float64(d) / float64(total) * 100
	}
	var nsPerElem float64
	if elems > 0 {
		nsPerElem = float64(recur.Nanoseconds()) / float64(elems)
	}
	fmt.Printf("DELTANET TIMING (%d steps): proj %.3f ms/step (%.1f%%) | recurrence %.3f ms/step (%.1f%%) | other %.3f ms/step (%.1f%%) | recurrence %.3f ns/state-elem (Σ %d elems)\n",
		n, msPer(proj), pct(proj), msPer(recur), pct(recur), msPer(other), pct(other), nsPerElem, elems)
}

// Gated DeltaNet, the linear-attention primitive of Qwen3.5/3.6-MoE (qwen3_5_moe). It replaces softmax attention on most layers
// with a gated delta-rule recurrence over a fixed-size per-head matrix state, so memory is O(1) in sequence length rather than a
// growing KV cache. The math mirrors HF's torch_recurrent_gated_delta_rule plus the surrounding conv, gates and gated RMSNorm,
// validated op-for-op against a traced golden (deltanet_test.go); see docs/completed/qwen3_5_moe.md.
//
// The CPU reference implementation lives in internal/program so backend kernel tests (metal/kernels) can exercise the single
// production implementation without importing decoder.

// deltaNetWeights holds one Gated DeltaNet layer's parameters, row-major, as HF
// stores them. All projections are bias-free; the depthwise conv is bias-free too.
type deltaNetWeights struct {
	// The three dominant projections (inProjQKV, inProjZ, outProj) are quantizable WeightMats: they dominate the bytes streamed per
	// token and decode is memory-bandwidth-bound, so holding them as f32 would stream them at f32 under any quant. WeightMat keeps f32
	// when no quant is asked for (the tiny goldens still match HF exactly) and carries int8/int4 otherwise.
	inProjQKV linalg.WeightMat // [convDim, hidden]   → [q;k;v]
	inProjZ   linalg.WeightMat // [valueDim, hidden]  → output gate z
	// A and B stay f32: they are small against the projections and feed the write/decay gates, where the recurrence is most
	// sensitive to precision. Quantizing them would buy a fraction of a percent of the bytes for real numerical risk.
	inProjB []float32        // [numV, hidden]      → write-gate logits β
	inProjA []float32        // [numV, hidden]      → decay-gate input a
	convW   []float32        // [convDim, K]        depthwise causal conv ([convDim,1,K] flattened)
	dtBias  []float32        // [numV]
	negExpA []float32        // [numV]              −exp(A_log), precomputed (see negExpAFromLog)
	normW   []float32        // [headVDim]          gated RMSNorm weight
	outProj linalg.WeightMat // [hidden, valueDim]
}

func (p qwen35Params) deltaNetParams() prog.DeltaNetParams {
	return prog.DeltaNetParams{
		ConvKernel:    p.ConvKernel,
		KeyHeadDim:    p.KeyHeadDim,
		ValueHeadDim:  p.ValueHeadDim,
		NumKeyHeads:   p.NumKeyHeads,
		NumValueHeads: p.NumValueHeads,
		NegEigval:     p.NegEigval,
		ONormEps:      p.ONormEps,
	}
}

// negExpAFromLog precomputes the DeltaNet decay coefficient −exp(A_log) that the
// recurrence multiplies by softplus(a+dt_bias).
func negExpAFromLog(aLog []float32) []float32 { return prog.NegExpAFromLog(aLog) }

// matvec computes y[r] = Σ_c W[r*cols+c]·x[c] for a row-major [rows,cols] weight.
func matvec(w []float32, rows, cols int, x []float32) []float32 {
	return prog.Matvec(w, rows, cols, x)
}

// matvecWM is matvec for a WeightMat: the same M=1 projection, but routed through the shared
// quantization-aware matmul so an int4/int8 weight uses the W4A8/W8A8 integer kernel instead of
// being dequantized. f32 WeightMats land on the same MatmulBT path matvec uses, so the no-quant
// numerics are unchanged — which is what keeps the tiny goldens exact.
func matvecWM(be Backend, w *linalg.WeightMat, x []float32) []float32 {
	y := make([]float32, w.Rows())
	matmul(be, w, x, y, 1)
	return y
}

// matvecWMPre is matvecWM using p's block when it was quantized for x and w (R-13; see w4a8Act).
func matvecWMPre(be Backend, w *linalg.WeightMat, p *w4a8Act, x []float32) []float32 {
	y := make([]float32, w.Rows())
	matmulPre(be, w, p, x, y, 1)
	return y
}

func sigmoidf(x float32) float32  { return prog.Sigmoidf(x) }
func softplusf(x float32) float32 { return prog.Softplusf(x) }

// deltaState is one Gated DeltaNet layer's streaming state in the KV cache: the
// last K-1 conv inputs (so the causal conv has its left context at decode) and
// the recurrent matrix state S (one [head_k_dim, head_v_dim] block per value
// head). Fixed size — independent of sequence length, and NOT position-
// truncatable (why qwen3_5_moe falls back from prefix reuse / speculative; see
// docs/completed/qwen3_5_moe.md).
type deltaState struct {
	convWin [][]float32 // up to K-1 prior mixed_qkv vectors, oldest first
	s       []float32   // [numV * head_k_dim * head_v_dim]
	hq      w4a8Act     // h quantized once for in_proj_qkv and in_proj_z (R-13); rewritten every step
}

func newDeltaState(p qwen35Params) *deltaState {
	return &deltaState{s: make([]float32, p.NumValueHeads*p.KeyHeadDim*p.ValueHeadDim)}
}

// gatedDeltaNetStep advances one token through the Gated DeltaNet layer, reading
// and updating st, and returns the layer output [hidden]. Driven per token by
// both prefill and decode (the recurrence is inherently sequential). Its four
// projections are here; everything between them is deltaNetCore, which the
// batched forward (gatedDeltaNetN) shares so the two cannot drift.
func gatedDeltaNetStep(be Backend, h []float32, w *deltaNetWeights, p qwen35Params, hidden int, eps float64, st *deltaState) []float32 {
	proj := func(wm *linalg.WeightMat, x []float32, pre bool) []float32 {
		if pre {
			if wm == &w.inProjQKV {
				st.hq.prepare(be, wm, x, 1)
			}
			return matvecWMPre(be, wm, &st.hq, x)
		}
		return matvecWM(be, wm, x)
	}
	return prog.GatedDeltaNetStep(
		proj, h,
		&w.inProjQKV, &w.inProjZ, &w.outProj,
		w.inProjB, w.inProjA, w.convW, w.dtBias, w.negExpA, w.normW,
		p.deltaNetParams(), hidden, eps,
		st.s, &st.convWin, deltaNetTimers(),
	)
}

// deltaNetCore is one token's sequential middle of the Gated DeltaNet layer, from the projections'
// outputs to out_proj's input: the depthwise causal conv (+ SiLU) over the window in st, the gated
// delta-rule recurrence (state in st.s), and the gated RMSNorm (× SiLU(z)).
func deltaNetCore(core, mixed, bt, at, z []float32, w *deltaNetWeights, p qwen35Params, eps float64, st *deltaState) {
	prog.DeltaNetCore(
		core, mixed, bt, at, z,
		w.convW, w.dtBias, w.negExpA, w.normW,
		p.deltaNetParams(), eps,
		st.s, &st.convWin, deltaNetTimers(),
	)
}

func deltaNetRecurrence(core, conv, at, bt []float32, w *deltaNetWeights, p qwen35Params, st *deltaState) {
	prog.DeltaNetRecurrence(
		core, conv, at, bt,
		w.dtBias, w.negExpA,
		p.deltaNetParams(),
		st.s, deltaNetTimers(),
	)
}

// gatedDeltaNet runs the layer over a whole sequence from a fresh state — a thin
// loop over gatedDeltaNetStep, so the streaming path is what the parity test
// exercises (the forward path uses the same step with a cached state).
func gatedDeltaNet(be Backend, h [][]float32, w *deltaNetWeights, p qwen35Params, hidden int, eps float64) [][]float32 {
	st := newDeltaState(p)
	out := make([][]float32, len(h))
	for t := range h {
		out[t] = gatedDeltaNetStep(be, h[t], w, p, hidden, eps, st)
	}
	return out
}

// l2normScaledInto writes x/‖x‖ (eps 1e-6, matching FLA's l2norm) times s into dst.
func l2normScaledInto(dst, x []float32, s float32) { prog.L2NormScaledInto(dst, x, s) }

// l2normScaled returns x/‖x‖ (eps 1e-6, matching FLA's l2norm) times s.
func l2normScaled(x []float32, s float32) []float32 { return prog.L2NormScaled(x, s) }
