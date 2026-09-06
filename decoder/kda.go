package decoder

import (
	"math"

	"github.com/townsendmerino/aikit/linalg"
)

// kdaWeights holds one Bailing Hybrid (Ling 3.0) Kimi Delta Attention layer's parameters,
// row-major, as HF stores them. Unlike Gated DeltaNet, q/k/v are three FULLY SEPARATE
// projections AND three separate depthwise causal convs (verified against the real
// modeling_bailing_moe_v3.py's BailingMoeV3KimiDeltaAttention — independent self.q_proj/k_proj/
// v_proj and self.q_conv1d/k_conv1d/v_conv1d, not Gated DeltaNet's single concatenated in_proj_qkv
// + combined conv). num_k_heads == num_heads and head_k_dim == head_v_dim == head_dim always for
// this family (no GVA-style key/value head expansion), so there is only one width to track.
type kdaWeights struct {
	qProj, kProj, vProj    linalg.WeightMat // [numHeads*headDim, hidden] each
	qConvW, kConvW, vConvW []float32        // [numHeads*headDim, K] each, depthwise, bias-free (fla's ShortConvolution default)
	fProj                  linalg.WeightMat // [numHeads*headDim, hidden] — decay-gate raw input (no_kda_lora path: config.no_kda_lora=true selects this single-linear form over a LoRA'd a/b split, which this family does not implement — see kdaArchitecture)
	dtBias                 []float32        // [numHeads*headDim] — PER-CHANNEL (Gated DeltaNet's is per-head)
	aLog                   []float32        // [numHeads] — raw A_log (kdaLowerBoundGate exponentiates internally, unlike Gated DeltaNet's precomputed −exp(A_log))
	bProj                  []float32        // [numHeads, hidden] — beta write-gate, per-head scalar (small; not quantized, mirrors qwen35's inProjB)
	gProj                  linalg.WeightMat // [numHeads*headDim, hidden] — output gate raw input (no_kda_lora path)
	oNormW                 []float32        // [headDim] gated-RMSNorm weight (sigmoid-activated — Gated DeltaNet's is SiLU)
	oProj                  linalg.WeightMat // [hidden, numHeads*headDim]
}

// kdaState is one KDA layer's streaming state in the KV cache: three independent rolling conv
// windows (one per q/k/v stream — Gated DeltaNet needs only one, since its conv is combined) and
// the recurrent matrix state S, one [headDim, headDim] block per head (K==V dims here, unlike
// Gated DeltaNet's asymmetric key/value head dims).
type kdaState struct {
	convWinQ, convWinK, convWinV [][]float32
	s                            []float32
}

func newKDAState(p kdaParams) *kdaState {
	return &kdaState{s: make([]float32, p.NumHeads*p.HeadDim*p.HeadDim)}
}

// kdaConvStream applies ONE stream's depthwise causal conv + SiLU: identical math to
// gatedDeltaNetStep's combined conv, just over one stream's own weight and window instead of a
// concatenated [q;k;v] buffer — the real per-stream split verified against the real checkpoint's
// separate q_conv1d/k_conv1d/v_conv1d tensors, not assumed reusable-as-one from the source alone.
func kdaConvStream(x, convW []float32, win [][]float32, K int) []float32 {
	width := len(x)
	out := make([]float32, width)
	for c := range width {
		s := convW[c*K+(K-1)] * x[c] // j=K-1 tap = current token
		for j := 0; j < K-1; j++ {
			if idx := len(win) - (K - 1) + j; idx >= 0 {
				s += convW[c*K+j] * win[idx][c]
			}
		}
		out[c] = silu(s)
	}
	return out
}

// kdaSlideWindow appends this token's raw (pre-conv) projection output and trims to the last K-1
// entries — the same rolling-window convention gatedDeltaNetStep's single conv window uses.
func kdaSlideWindow(win [][]float32, x []float32, K int) [][]float32 {
	win = append(win, x)
	if len(win) > K-1 {
		win = win[len(win)-(K-1):]
	}
	return win
}

// kdaMixerStep advances one token through a Bailing Hybrid KDA layer, reading and updating st,
// and returns the layer output [hidden]. Reuses kdaLowerBoundGate/kdaRecurrentStep
// (decoder/kda_rehearsal.go) UNCHANGED for the one genuinely new primitive (per-channel decay);
// everything around it — projections, per-stream conv, q/k L2-norm-in-kernel, beta write-gate,
// gated-RMSNorm-then-out_proj tail — mirrors gatedDeltaNetStep's own shape, parameterized for
// KDA's real departures (three separate convs, per-channel dt_bias, sigmoid gated-norm instead of
// SiLU).
func kdaMixerStep(be Backend, h []float32, w *kdaWeights, p kdaParams, hidden int, eps float64, st *kdaState) []float32 {
	H, D, K := p.NumHeads, p.HeadDim, p.ConvKernel
	projSize := H * D
	qScale := float32(1 / math.Sqrt(float64(D)))

	// 1. Project, then per-stream depthwise causal conv + SiLU.
	qm := matvecWM(be, &w.qProj, h)
	km := matvecWM(be, &w.kProj, h)
	vm := matvecWM(be, &w.vProj, h)
	q := kdaConvStream(qm, w.qConvW, st.convWinQ, K)
	k := kdaConvStream(km, w.kConvW, st.convWinK, K)
	v := kdaConvStream(vm, w.vConvW, st.convWinV, K)
	st.convWinQ = kdaSlideWindow(st.convWinQ, qm, K)
	st.convWinK = kdaSlideWindow(st.convWinK, km, K)
	st.convWinV = kdaSlideWindow(st.convWinV, vm, K)

	// 2. Gates: per-channel decay-gate input, per-head beta.
	gDecayRaw := matvecWM(be, &w.fProj, h) // [projSize]
	betaLogits := matvec(w.bProj, H, hidden, h)

	// 3. Per-head KDA recurrence — the one genuinely new primitive (per-channel decay), reusing
	// kda_rehearsal.go's already-reference-verified functions unchanged.
	core := make([]float32, projSize)
	for hh := range H {
		qh := l2normScaled(q[hh*D:(hh+1)*D], qScale)
		kh := l2normScaled(k[hh*D:(hh+1)*D], 1)
		vh := v[hh*D : (hh+1)*D]
		rawGate := gDecayRaw[hh*D : (hh+1)*D]
		dtb := w.dtBias[hh*D : (hh+1)*D]
		gateLog := kdaLowerBoundGate(rawGate, dtb, w.aLog[hh], float32(p.LowerBound))
		decay := make([]float32, D)
		for i, x := range gateLog {
			decay[i] = float32(math.Exp(float64(x)))
		}
		beta := sigmoidf(betaLogits[hh])
		S := st.s[hh*D*D : (hh+1)*D*D]
		out := kdaRecurrentStep(qh, kh, vh, decay, beta, S)
		copy(core[hh*D:(hh+1)*D], out)
	}

	// 4. Gated RMSNorm (over headDim, × sigmoid(g), NOT SiLU — FusedRMSNormGated's own
	// activation='sigmoid'), then out_proj.
	gOutRaw := matvecWM(be, &w.gProj, h)
	for hh := range H {
		seg := core[hh*D : (hh+1)*D]
		zt := gOutRaw[hh*D : (hh+1)*D]
		var ss float64
		for _, x := range seg {
			ss += float64(x) * float64(x)
		}
		inv := float32(1 / math.Sqrt(ss/float64(D)+eps))
		for d := range D {
			seg[d] = seg[d] * inv * w.oNormW[d] * sigmoidf(zt[d])
		}
	}
	return matvecWM(be, &w.oProj, core)
}
