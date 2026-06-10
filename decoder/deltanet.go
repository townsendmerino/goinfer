package decoder

import (
	"math"

	"github.com/townsendmerino/aikit/linalg"
)

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
	inProjQKV []float32 // [convDim, hidden]   → [q;k;v]
	inProjZ   []float32 // [valueDim, hidden]  → output gate z
	inProjB   []float32 // [numV, hidden]      → write-gate logits β
	inProjA   []float32 // [numV, hidden]      → decay-gate input a
	convW     []float32 // [convDim, K]        depthwise causal conv ([convDim,1,K] flattened)
	dtBias    []float32 // [numV]
	negExpA   []float32 // [numV]              −exp(A_log), precomputed (see negExpAFromLog)
	normW     []float32 // [headVDim]          gated RMSNorm weight
	outProj   []float32 // [hidden, valueDim]
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
func gatedDeltaNetStep(h []float32, w *deltaNetWeights, p qwen35Params, hidden int, eps float64, st *deltaState) []float32 {
	nk, nv := p.NumKeyHeads, p.NumValueHeads
	hk, hv := p.KeyHeadDim, p.ValueHeadDim
	keyDim, valueDim := hk*nk, hv*nv
	convDim := 2*keyDim + valueDim
	K := p.ConvKernel
	rep := nv / nk
	qScale := float32(1 / math.Sqrt(float64(hk)))

	// 1. Projection + depthwise causal conv (+ SiLU). Taps t-K+1..t: the last K-1
	// come from convWin (zero-padded early), the K-th is this token.
	mixed := matvec(w.inProjQKV, convDim, hidden, h)
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
	z := matvec(w.inProjZ, valueDim, hidden, h)

	// 3. Gated delta-rule recurrence, per value head; state persists in st.s.
	core := make([]float32, valueDim)
	for headV := range nv {
		headK := headV / rep
		q := l2normScaled(conv[headK*hk:headK*hk+hk], qScale)
		k := l2normScaled(conv[keyDim+headK*hk:keyDim+headK*hk+hk], 1)
		v := conv[2*keyDim+headV*hv : 2*keyDim+headV*hv+hv]
		g := w.negExpA[headV] * softplusf(at[headV]+w.dtBias[headV]) // log-decay (negExpA = −exp(A_log))
		gt := float32(math.Exp(float64(g)))
		beta := sigmoidf(bt[headV])

		S := st.s[headV*hk*hv : (headV+1)*hk*hv] // [hk, hv]
		for i := range S {
			S[i] *= gt
		}
		out := core[headV*hv : headV*hv+hv]
		for vd := range hv {
			var kv float32
			for kd := range hk {
				kv += S[kd*hv+vd] * k[kd]
			}
			delta := (v[vd] - kv) * beta
			var o float32
			for kd := range hk {
				S[kd*hv+vd] += k[kd] * delta
				o += S[kd*hv+vd] * q[kd]
			}
			out[vd] = o
		}
	}

	// 4. Gated RMSNorm (over head_v_dim, × SiLU(z)) then out_proj.
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
	return matvec(w.outProj, hidden, valueDim, core)
}

// gatedDeltaNet runs the layer over a whole sequence from a fresh state — a thin
// loop over gatedDeltaNetStep, so the streaming path is what the parity test
// exercises (the forward path uses the same step with a cached state).
func gatedDeltaNet(h [][]float32, w *deltaNetWeights, p qwen35Params, hidden int, eps float64) [][]float32 {
	st := newDeltaState(p)
	out := make([][]float32, len(h))
	for t := range h {
		out[t] = gatedDeltaNetStep(h[t], w, p, hidden, eps, st)
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
