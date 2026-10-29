package decoder

import (
	"os"
	"sync"

	"github.com/townsendmerino/aikit/linalg"
	"github.com/townsendmerino/goinfer/internal/mamba2"
)

// Mamba-2 selective state-space sequence mixer (Granite-4.0 hybrid). Like Gated DeltaNet (deltanet.go) it is an
// inherently sequential recurrence driven one token at a time by both prefill and decode, with a small per-token causal
// conv before the scan. This is the parity-first sequential form, and the only one (a chunked/segsum scan was removed
// unused). It mirrors the HF GraniteMoeHybridMambaLayer single-token (cache) path exactly.
//
// Shapes: nHeads heads of headDim (P), an SSM state of dState (N) per head-channel,
// B/C shared across heads within a group (nGroups). The in_proj emits
// [gate z (dInner) ‖ xBC (convDim) ‖ dt (nHeads)]; conv1d runs over xBC then splits
// into x (dInner) ‖ B (nGroups·N) ‖ C (nGroups·N). dInner = nHeads·P.

type mamba2Params struct {
	NHeads  int
	HeadDim int // P
	DState  int // N
	NGroups int
	DConv   int // K (causal conv kernel)
	Hidden  int
	// NormGroups is the group count for the GATED RMSNorm specifically — families
	// differ here: Granite (GraniteMoeHybridRMSNormGated) normalizes over the FULL
	// dInner (NormGroups 1), Nemotron-H (Zamba2RMSNormGated) normalizes per group
	// (NormGroups = NGroups). Equal when NGroups==1. 0 ⇒ treated as 1.
	NormGroups int
}

// mixer is p's geometry as internal/mamba2 takes it.
func (p mamba2Params) mixer() mamba2.Params {
	return mamba2.Params{NHeads: p.NHeads, HeadDim: p.HeadDim, DState: p.DState, NGroups: p.NGroups, DConv: p.DConv, NormGroups: p.NormGroups}
}

func (p mamba2Params) dInner() int  { return p.mixer().DInner() }
func (p mamba2Params) projDim() int { return 2*p.dInner() + 2*p.NGroups*p.DState + p.NHeads }

// mamba2Weights holds one mixer's tensors, f32 (parity-first, like the qwen35 DeltaNet/softmax weights), row-major [out, in]
// for the projections. On a quantized Nemotron-H load the two projections are int8 instead (inQ, outQ; quantizeProjections),
// with inProj and outProj nil.
type mamba2Weights struct {
	inProj  []float32 // [projDim, hidden]
	convW   []float32 // [convDim, K] (HF conv1d.weight [convDim,1,K] flattened)
	convB   []float32 // [convDim]
	aLog    []float32 // [nHeads] (A = -exp(aLog))
	d       []float32 // [nHeads] skip
	dtBias  []float32 // [nHeads]
	normW   []float32 // [dInner] gated RMSNorm weight
	outProj []float32 // [hidden, dInner]
	// inQ and outQ are in_proj and out_proj as per-row int8, weight-only (f32 activations): linalg.QuantizeInt8 of the f32
	// weights, the codes and scales the GPU backends upload.
	inQ, outQ linalg.WeightMat
}

// mixer is w's tensors between the projections, as internal/mamba2 takes them.
func (w *mamba2Weights) mixer() mamba2.Weights {
	return mamba2.Weights{ConvW: w.convW, ConvB: w.convB, ALog: w.aLog, D: w.d, DtBias: w.dtBias, NormW: w.normW}
}

// mambaProjInt8 reports whether a load at q holds Nemotron-H's Mamba projections as int8: every integer quant does, and
// an f32, bf16 or f16 load keeps them f32 (docs/queue-engineering.md, B27).
func mambaProjInt8(q quantMode) bool {
	switch q {
	case quantInt8, quantInt8I8, quantInt4, quantInt4Mix, quantQ4K:
		return !mambaKeepF32ForTest
	}
	return false
}

// mambaKeepF32ForTest keeps the f32 projections on a quantized load: the f32 arm of B27's fidelity bar, and its planted defect.
var mambaKeepF32ForTest bool

// quantizeProjections replaces the f32 projections with their int8 form.
func (w *mamba2Weights) quantizeProjections(projDim, hidden, dInner int) {
	w.inQ = linalg.QuantizeInt8(w.inProj, projDim, hidden, false)
	w.outQ = linalg.QuantizeInt8(w.outProj, hidden, dInner, false)
	w.inProj, w.outProj = nil, nil
}

// mamba2State is the per-layer recurrent state (the conv window and the SSM state), a second kind of recurrent state in
// the hybrid cache beside deltaState.
type mamba2State = mamba2.State

func newMamba2State(p mamba2Params) *mamba2State { return mamba2.NewState(p.mixer()) }

// mamba2Step advances one token through the Mamba-2 mixer, reading and updating st,
// returning the layer output [hidden].
func mamba2Step(h []float32, w *mamba2Weights, p mamba2Params, eps float64, st *mamba2State) []float32 {
	dInner := p.dInner()

	// 1. in_proj → [z (gate), xBC, dt].
	inProj, outProj := w.inProj, w.outProj
	if ssmQ8CPU { // confirmation seam: match the resident W8A8 — int8 weights AND int8 activations
		inProj = q8RoundTrip(inProj, p.projDim(), p.Hidden)
		outProj = q8RoundTrip(outProj, p.Hidden, dInner)
		h = q8RoundTrip(h, 1, len(h)) // the in_proj activation (resident rmsQuant → int8)
	} else if ssmQ8WeightsOnly { // W8A16 control: int8 mamba WEIGHTS, f32 activations (no act quant)
		inProj = q8RoundTrip(inProj, p.projDim(), p.Hidden)
		outProj = q8RoundTrip(outProj, p.Hidden, dInner)
	}
	var proj []float32
	if w.inQ.Rows() > 0 {
		proj = matvecWM(nil, &w.inQ, h)
	} else {
		proj = matvec(inProj, p.projDim(), p.Hidden, h)
	}

	// 2–5. The conv, the selective scan and the gated norm (internal/mamba2).
	gated := mamba2.Mix(proj, w.mixer(), p.mixer(), eps, st, ssmForceF32)
	if ssmQ8CPU {
		gated = q8RoundTrip(gated, 1, len(gated)) // the out_proj activation (resident quant → int8)
	}
	if mambaCapHook != nil { // test seam: capture real (in_proj output, mixer output) for GPU-kernel parity replay
		mambaCapHook(proj, gated)
	}
	if w.outQ.Rows() > 0 {
		return matvecWM(nil, &w.outQ, gated)
	}
	return matvec(outProj, p.Hidden, dInner, gated)
}

// mambaCapHook (test seam, gpu/mamba_realinput_test.go) captures each mamba2Step's in_proj
// output and mixer output (post gated-norm, pre out_proj) so real per-token inputs can be
// replayed through the GPU conv/ssm/gatedNorm kernels — the "green isolation / broken in-model"
// reconciliation. nil in production.
var mambaCapHook func(proj, gated []float32)

// ssmQ8CPU + q8RoundTrip are a confirmation seam (GOINFER_SSM_Q8CPU): round-trip the
// Mamba projections through int8 so the CPU reference carries the SAME quantization error
// as the resident W8A8 path — isolating kernel-correctness from int8 sensitivity.
var ssmQ8CPU = os.Getenv("GOINFER_SSM_Q8CPU") != ""

// ssmQ8WeightsOnly (W8A16 control, set by tests) round-trips the mamba in/out_proj WEIGHTS
// through int8 but keeps activations f32 — int8-weights / f32-activations, the W8A16 question.
var ssmQ8WeightsOnly bool

// ssmForceF32 (precision-localization seam, set by tests) downcasts the SSM compute
// in mamba2Step to f32 (mamba2.Mix's f32) — the exp(dt·A) argument and the gated-norm
// accumulation — to reproduce the GPU's f32-only SSM on the CPU reference. Off in production.
var ssmForceF32 bool

// q8Memo caches the round-trip of the (stable) weight matrices so the confirmation
// seam doesn't re-quantize 30M-element projections every token. Keyed by the slice's
// backing pointer; weights never move. Activations (N==1) change per token → not cached.
var q8Memo sync.Map

func q8RoundTrip(w []float32, N, K int) []float32 {
	if N > 1 {
		key := &w[0]
		if v, ok := q8Memo.Load(key); ok {
			return v.([]float32)
		}
		out := q8roundtrip(w, N, K)
		q8Memo.Store(key, out)
		return out
	}
	return q8roundtrip(w, N, K)
}

func q8roundtrip(w []float32, N, K int) []float32 {
	q, s := linalg.QuantizeRowsInt8(w, N, K)
	out := make([]float32, N*K)
	for r := range N {
		for k := range K {
			out[r*K+k] = float32(q[r*K+k]) * s[r]
		}
	}
	return out
}

// mamba2Seq runs the mixer over a whole sequence from a fresh state — a thin loop
// over mamba2Step, the form the parity test exercises (the forward path uses the
// same step with a cached state).
func mamba2Seq(h [][]float32, w *mamba2Weights, p mamba2Params, eps float64) [][]float32 {
	st := newMamba2State(p)
	out := make([][]float32, len(h))
	for t := range h {
		out[t] = mamba2Step(h[t], w, p, eps, st)
	}
	return out
}
