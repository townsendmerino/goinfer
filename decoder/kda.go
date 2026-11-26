package decoder

import (
	"math"

	"github.com/townsendmerino/goinfer/internal/runtime/archparts"
	"github.com/townsendmerino/goinfer/internal/runtime/hybridattn"
	"github.com/townsendmerino/goinfer/internal/runtime/normact"
)

// kdaState is one KDA layer's streaming state in the KV cache: three independent rolling conv
// windows (one per q/k/v stream — Gated DeltaNet needs only one, since its conv is combined) and
// the recurrent matrix state S, one [headDim, headDim] block per head (K==V dims here, unlike
// Gated DeltaNet's asymmetric key/value head dims).
type kdaState struct {
	convWinQ, convWinK, convWinV [][]float32
	s                            []float32
	hq                           w4a8Act // h quantized once for q, k, v, f and g; rewritten every step
}

func newKDAState(p archparts.KDAParams) *kdaState {
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
		out[c] = normact.SiLU(s)
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
func kdaMixerStep(be Backend, h []float32, w *archparts.KDAWeights, p archparts.KDAParams, hidden int, eps float64, st *kdaState) []float32 {
	H, D, K := p.NumHeads, p.HeadDim, p.ConvKernel
	projSize := H * D

	// 1. Project, then per-stream depthwise causal conv + SiLU.
	st.hq.prepare(be, &w.QProj, h, 1)
	qm := matvecWMPre(be, &w.QProj, &st.hq, h)
	km := matvecWMPre(be, &w.KProj, &st.hq, h)
	vm := matvecWMPre(be, &w.VProj, &st.hq, h)
	q := kdaConvStream(qm, w.QConvW, st.convWinQ, K)
	k := kdaConvStream(km, w.KConvW, st.convWinK, K)
	v := kdaConvStream(vm, w.VConvW, st.convWinV, K)
	st.convWinQ = kdaSlideWindow(st.convWinQ, qm, K)
	st.convWinK = kdaSlideWindow(st.convWinK, km, K)
	st.convWinV = kdaSlideWindow(st.convWinV, vm, K)

	// 2. Gates: per-channel decay-gate input, per-head beta.
	gDecayRaw := matvecWMPre(be, &w.FProj, &st.hq, h) // [projSize]
	betaLogits := hybridattn.Matvec(w.BProj, H, hidden, h)

	// 3. Per-head KDA recurrence — the one genuinely new primitive (per-channel decay), reusing
	// kda_rehearsal.go's already-reference-verified functions unchanged.
	core := make([]float32, projSize)
	kdaRecurrence(core, q, k, v, gDecayRaw, betaLogits, w, p, st)

	// 4. Gated RMSNorm (over headDim, × sigmoid(g), NOT SiLU — FusedRMSNormGated's own
	// activation='sigmoid'), then out_proj.
	gOutRaw := matvecWMPre(be, &w.GProj, &st.hq, h)
	for hh := range H {
		seg := core[hh*D : (hh+1)*D]
		zt := gOutRaw[hh*D : (hh+1)*D]
		var ss float64
		for _, x := range seg {
			ss += float64(x) * float64(x)
		}
		inv := float32(1 / math.Sqrt(ss/float64(D)+eps))
		for d := range D {
			seg[d] = seg[d] * inv * w.ONormW[d] * hybridattn.Sigmoidf(zt[d])
		}
	}
	return matvecWM(be, &w.OProj, core)
}

func kdaRecurrence(core, q, k, v, gDecayRaw, betaLogits []float32, w *archparts.KDAWeights, p archparts.KDAParams, st *kdaState) {
	H, D := p.NumHeads, p.HeadDim
	qScale := float32(1 / math.Sqrt(float64(D)))

	var qhBuf, khBuf, gateLogBuf, decayBuf [256]float32
	var qh, kh, gateLog, decay []float32
	if D <= len(qhBuf) {
		qh = qhBuf[:D]
		kh = khBuf[:D]
		gateLog = gateLogBuf[:D]
		decay = decayBuf[:D]
	} else {
		qh = make([]float32, D)
		kh = make([]float32, D)
		gateLog = make([]float32, D)
		decay = make([]float32, D)
	}

	for hh := range H {
		hybridattn.L2NormScaledInto(qh, q[hh*D:(hh+1)*D], qScale)
		hybridattn.L2NormScaledInto(kh, k[hh*D:(hh+1)*D], 1)
		vh := v[hh*D : (hh+1)*D]
		rawGate := gDecayRaw[hh*D : (hh+1)*D]
		dtb := w.DtBias[hh*D : (hh+1)*D]
		kdaLowerBoundGateInto(gateLog, rawGate, dtb, w.ALog[hh], float32(p.LowerBound))
		_ = decay[D-1]
		_ = gateLog[D-1]
		di := 0
		for ; di+3 < D; di += 4 {
			decay[di] = float32(math.Exp(float64(gateLog[di])))
			decay[di+1] = float32(math.Exp(float64(gateLog[di+1])))
			decay[di+2] = float32(math.Exp(float64(gateLog[di+2])))
			decay[di+3] = float32(math.Exp(float64(gateLog[di+3])))
		}
		for ; di < D; di++ {
			decay[di] = float32(math.Exp(float64(gateLog[di])))
		}

		beta := hybridattn.Sigmoidf(betaLogits[hh])
		S := st.s[hh*D*D : (hh+1)*D*D]
		out := core[hh*D : (hh+1)*D]
		kdaRecurrentStepInto(out, qh, kh, vh, decay, beta, S)
	}
}
