package decoder

import (
	"fmt"
	"math"
	"os"
	"sync/atomic"
	"time"

	"github.com/townsendmerino/aikit/linalg"
)

// deltaNetTiming env-gates the sixth outing of this repo's component-stub timing
// method (GOINFER_DELTANET_TIMING=1): splits gatedDeltaNetStep's ~19%-of-decode-token
// cost (docs/task-zeno-compare.md's diagnostic) into the three dominant projections
// (already W4A8/W8A8-quantized, presumably fast), the delta-rule recurrence proper
// (section 3 below — plain scalar Go, the DeltaNet-CPU-recurrence brief's suspect),
// and everything else (conv, gates, gated RMSNorm). Atomic accumulators, not
// Generate-loop-locals, so concurrent decode streams don't race on them — added,
// used to record the split, then reverted, per this repo's own discipline.
var deltaNetTiming = os.Getenv("GOINFER_DELTANET_TIMING") != ""

var (
	dnProjNs     atomic.Int64
	dnRecurNs    atomic.Int64
	dnOtherNs    atomic.Int64
	dnStateElems atomic.Int64 // Σ (nv·hk·hv) over every step — the recurrence's own rate-check denominator
	dnCalls      atomic.Int64
)

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

// Gated DeltaNet — the linear-attention primitive of Qwen3.5/3.6-MoE
// (qwen3_5_moe). It replaces softmax attention on most layers with a gated
// delta-rule recurrence over a fixed-size per-head matrix state, so its memory
// is O(1) in sequence length rather than a growing KV cache. See
// docs/qwen3_5_moe.md; the math mirrors HF's torch_recurrent_gated_delta_rule +
// the surrounding conv / gates / gated-RMSNorm, validated op-for-op against a
// traced golden (deltanet_test.go).
//
// This is the parity-first reference implementation: plain f32, sequential over
// positions. Perf (a chunked/parallel scan, quantized projections) is a later
// track.

// deltaNetWeights holds one Gated DeltaNet layer's parameters, row-major, as HF
// stores them. All projections are bias-free; the depthwise conv is bias-free too.
type deltaNetWeights struct {
	// THE THREE DOMINANT PROJECTIONS ARE QUANTIZABLE (2026-08-19). They were []float32 —
	// "parity-first", from the qwen3_5_moe bring-up — which meant a 27.8B Qwen3.8 at Quant:"int4"
	// still streamed them as f32: 22.1 GB per token across 48 DeltaNet layers, against ~9.5 GB for
	// the whole int4 FFN. Decode at this size is memory-bandwidth-bound, so that WAS the speed.
	// WeightMat keeps f32 when the caller asks for no quant (the tiny goldens still match HF
	// exactly), and carries int8/int4 when they do.
	inProjQKV linalg.WeightMat // [convDim, hidden]   → [q;k;v]
	inProjZ   linalg.WeightMat // [valueDim, hidden]  → output gate z
	// A and B stay f32: [48, 5120] each is ~1 MB per layer (~94 MB total on the 27B) against the
	// 22 GB above, and they feed the write/decay gates, where the recurrence is most sensitive to
	// precision. Quantizing them would buy ~0.2% of the bytes for real numerical risk.
	inProjB []float32        // [numV, hidden]      → write-gate logits β
	inProjA []float32        // [numV, hidden]      → decay-gate input a
	convW   []float32        // [convDim, K]        depthwise causal conv ([convDim,1,K] flattened)
	dtBias  []float32        // [numV]
	negExpA []float32        // [numV]              −exp(A_log), precomputed (see negExpAFromLog)
	normW   []float32        // [headVDim]          gated RMSNorm weight
	outProj linalg.WeightMat // [hidden, valueDim]
}

// negExpAFromLog precomputes the DeltaNet decay coefficient −exp(A_log) that the
// recurrence multiplies by softplus(a+dt_bias). The safetensors path stores raw
// A_log and converts here at load; the GGUF path stores −exp(A_log) already
// (llama.cpp's converter bakes it), so it loads straight into negExpA. Computing
// it once at load keeps the per-step recurrence free of the exp and lets both
// container formats share one forward.
func negExpAFromLog(aLog []float32) []float32 {
	out := make([]float32, len(aLog))
	for i, v := range aLog {
		out[i] = -float32(math.Exp(float64(v)))
	}
	return out
}

// matvec computes y[r] = Σ_c W[r*cols+c]·x[c] for a row-major [rows,cols] weight
// — the M=1 weight projection for the qwen3_5_moe path (DeltaNet in/out-proj and
// the softmax-attention q/k/v/o). It carries the dominant decode term, so it runs
// on the SIMD A·Bᵀ kernel (W[rows,cols]·x = MatmulBT at M=1) rather than the
// scalar row loop. f32 in, f32 out — this is SIMD reassociation only (no
// precision change of kind), unlike the float64→f32 attention vectorization.
func matvec(w []float32, rows, cols int, x []float32) []float32 {
	y := make([]float32, rows)
	linalg.MatmulBT(x, w, y, 1, cols, rows)
	return y
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

func sigmoidf(x float32) float32 { return float32(1 / (1 + math.Exp(-float64(x)))) }

// softplusf matches torch.nn.functional.softplus (default beta=1, threshold=20:
// linear above the threshold for numerical stability).
func softplusf(x float32) float32 {
	if x > 20 {
		return x
	}
	return float32(math.Log1p(math.Exp(float64(x))))
}

// deltaState is one Gated DeltaNet layer's streaming state in the KV cache: the
// last K-1 conv inputs (so the causal conv has its left context at decode) and
// the recurrent matrix state S (one [head_k_dim, head_v_dim] block per value
// head). Fixed size — independent of sequence length, and NOT position-
// truncatable (why qwen3_5_moe falls back from prefix reuse / speculative; see
// docs/qwen3_5_moe.md).
type deltaState struct {
	convWin [][]float32 // up to K-1 prior mixed_qkv vectors, oldest first
	s       []float32   // [numV * head_k_dim * head_v_dim]
}

func newDeltaState(p qwen35Params) *deltaState {
	return &deltaState{s: make([]float32, p.NumValueHeads*p.KeyHeadDim*p.ValueHeadDim)}
}

// gatedDeltaNetStep advances one token through the Gated DeltaNet layer, reading
// and updating st, and returns the layer output [hidden]. Driven per token by
// both prefill and decode (the recurrence is inherently sequential).
func gatedDeltaNetStep(be Backend, h []float32, w *deltaNetWeights, p qwen35Params, hidden int, eps float64, st *deltaState) []float32 {
	nk, nv := p.NumKeyHeads, p.NumValueHeads
	hk, hv := p.KeyHeadDim, p.ValueHeadDim
	keyDim, valueDim := hk*nk, hv*nv
	convDim := 2*keyDim + valueDim
	K := p.ConvKernel
	rep := nv / nk
	qScale := float32(1 / math.Sqrt(float64(hk)))

	var t0 time.Time
	if deltaNetTiming {
		t0 = time.Now()
	}

	// 1. Projection + depthwise causal conv (+ SiLU). Taps t-K+1..t: the last K-1
	// come from convWin (zero-padded early), the K-th is this token.
	mixed := matvecWM(be, &w.inProjQKV, h)
	if deltaNetTiming {
		dnProjNs.Add(int64(time.Since(t0)))
		t0 = time.Now()
	}
	conv := make([]float32, convDim)
	win := st.convWin
	for c := range convDim {
		s := w.convW[c*K+(K-1)] * mixed[c] // j=K-1 tap = current token
		for j := 0; j < K-1; j++ {
			if idx := len(win) - (K - 1) + j; idx >= 0 {
				s += w.convW[c*K+j] * win[idx][c]
			}
		}
		conv[c] = silu(s)
	}
	// slide the window: keep the last K-1 mixed vectors
	st.convWin = append(st.convWin, mixed)
	if len(st.convWin) > K-1 {
		st.convWin = st.convWin[len(st.convWin)-(K-1):]
	}

	// 2. Gates + output gate.
	bt := matvec(w.inProjB, nv, hidden, h)
	at := matvec(w.inProjA, nv, hidden, h)
	z := matvecWM(be, &w.inProjZ, h)
	if deltaNetTiming {
		dnOtherNs.Add(int64(time.Since(t0))) // conv (above) + gates (this block)
	}

	// 3. Gated delta-rule recurrence, per value head; state persists in st.s.
	core := make([]float32, valueDim)
	for headV := range nv {
		if deltaNetTiming {
			t0 = time.Now()
		}
		headK := headV / rep
		q := l2normScaled(conv[headK*hk:headK*hk+hk], qScale)
		k := l2normScaled(conv[keyDim+headK*hk:keyDim+headK*hk+hk], 1)
		v := conv[2*keyDim+headV*hv : 2*keyDim+headV*hv+hv]
		g := w.negExpA[headV] * softplusf(at[headV]+w.dtBias[headV]) // log-decay (negExpA = −exp(A_log))
		gt := float32(math.Exp(float64(g)))
		beta := sigmoidf(bt[headV])
		if deltaNetTiming {
			dnOtherNs.Add(int64(time.Since(t0))) // l2normScaled × 2 + gate scalars
			t0 = time.Now()
		}

		S := st.s[headV*hk*hv : (headV+1)*hk*hv] // [hk, hv]
		for i := range S {
			S[i] *= gt
		}
		out := core[headV*hv : headV*hv+hv]
		// P-07: kd-outer / vd-inner, walking S row-major (S[kd*hv:kd*hv+hv] is contiguous)
		// instead of the original vd-outer / kd-inner, which strided hv floats (512 B at
		// hk=hv=128) apart on every access. Bit-identical: for a fixed vd, both reductions
		// (kv, then out) still sum their kd=0..hk-1 contributions in the same ascending
		// order; S's columns are disjoint across vd, so computing every vd's read-phase
		// (kv) before any vd's write-phase (the S update + out accumulation), rather than
		// interleaved per vd as the original did, changes nothing either.
		kv := make([]float32, hv)
		for kd := range hk {
			row := S[kd*hv : kd*hv+hv]
			kk := k[kd]
			for vd := range hv {
				kv[vd] += row[vd] * kk
			}
		}
		delta := kv // aliased and overwritten in place: index vd is read once, then written once
		for vd := range hv {
			delta[vd] = (v[vd] - kv[vd]) * beta
		}
		for kd := range hk {
			row := S[kd*hv : kd*hv+hv]
			kk, qq := k[kd], q[kd]
			for vd := range hv {
				row[vd] += kk * delta[vd]
				out[vd] += row[vd] * qq
			}
		}
		if deltaNetTiming {
			dnRecurNs.Add(int64(time.Since(t0)))
			dnStateElems.Add(int64(hk * hv))
		}
	}

	var capPre, capGate []float32
	if deltaCapHook != nil { // test seam: step 3's output, before step 4 overwrites it in place
		capPre = append([]float32(nil), core...)
		capGate = make([]float32, 2*nv)
		for headV := range nv {
			g := w.negExpA[headV] * softplusf(at[headV]+w.dtBias[headV])
			capGate[headV*2+0] = sigmoidf(bt[headV])
			capGate[headV*2+1] = float32(math.Exp(float64(g)))
		}
	}

	// 4. Gated RMSNorm (over head_v_dim, × SiLU(z)) then out_proj.
	if deltaNetTiming {
		t0 = time.Now()
	}
	for headV := range nv {
		seg := core[headV*hv : headV*hv+hv]
		zt := z[headV*hv : headV*hv+hv]
		var ss float64
		for _, x := range seg {
			ss += float64(x) * float64(x)
		}
		inv := float32(1 / math.Sqrt(ss/float64(hv)+eps))
		for vd := range hv {
			seg[vd] = seg[vd] * inv * w.normW[vd] * silu(zt[vd])
		}
	}
	if deltaNetTiming {
		dnOtherNs.Add(int64(time.Since(t0)))
	}
	if deltaCapHook != nil {
		deltaCapHook(mixed, conv, append(append([]float32(nil), bt...), at...), capGate, capPre, core, z)
	}
	if deltaNetTiming {
		t0 = time.Now()
	}
	out := matvecWM(be, &w.outProj, core)
	if deltaNetTiming {
		dnProjNs.Add(int64(time.Since(t0)))
		dnCalls.Add(1)
	}
	return out
}

// deltaCapHook (test seam, gpu/deltanet_test.go) hands a backend every intermediate its kernels
// must reproduce, per step:
//
//	mixed    [convDim]  the in_proj_qkv output, PRE-conv    — the conv kernel's input
//	conv     [convDim]  post-conv, post-SiLU [q|k|v]        — deltaNorm's input
//	gateIn   [2*nv]     bt ‖ at, the raw gate projections   — deltaGates' input
//	betaGate [2*nv]     interleaved (beta, decay)           — deltaGates' output
//	corePre  [valueDim] step 3's output, before step 4      — deltaRule's output
//	gated    [valueDim] post gated-RMSNorm, pre out_proj    — deltaGNorm's output
//	z        [valueDim] the output gate, pre-SiLU           — deltaGNorm's input
//
// `mixed` is here for CUDA rather than WebGPU: WebGPU's causal conv is Mamba-2's, already gated,
// so its test could start from `conv`. CUDA has no SSM engine to reuse, so its conv is new code
// and has to be gated from its own input.
//
// It exists for the same reason mambaCapHook does: these are locals, two of them overwritten in
// place, so a parity gate cannot otherwise reach them — and re-deriving them in the backend's test
// package would be a second unvalidated implementation of the thing under test.
var deltaCapHook func(mixed, conv, gateIn, betaGate, corePre, gated, z []float32)

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

// l2normScaled returns x/‖x‖ (eps 1e-6, matching FLA's l2norm) times s.
func l2normScaled(x []float32, s float32) []float32 {
	var ss float64
	for _, v := range x {
		ss += float64(v) * float64(v)
	}
	inv := float32(math.Sqrt(1/(ss+1e-6))) * s
	out := make([]float32, len(x))
	for i, v := range x {
		out[i] = v * inv
	}
	return out
}
