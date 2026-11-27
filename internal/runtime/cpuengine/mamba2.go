package cpuengine

import (
	"sync"

	"github.com/townsendmerino/aikit/linalg"
	"github.com/townsendmerino/goinfer/internal/runtime/archparts"
	"github.com/townsendmerino/goinfer/internal/runtime/hybridattn"
	"github.com/townsendmerino/goinfer/internal/runtime/mamba2"
)

// Mamba-2 selective state-space sequence mixer (Granite-4.0 hybrid). Like Gated DeltaNet (deltanet.go) it is an
// inherently sequential recurrence driven one token at a time by both prefill and decode, with a small per-token causal
// conv before the scan. This is the parity-first sequential form, and the only one (a chunked/segsum scan was removed
// unused). It mirrors the HF GraniteMoeHybridMambaLayer single-token (cache) path exactly.
//
// Shapes nHeads heads of headDim (P), an SSM state of dState (N) per head-channel,
// B/C shared across heads within a group (nGroups). The in_proj emits
// [gate z (dInner) ‖ xBC (convDim) ‖ dt (nHeads)]; conv1d runs over xBC then splits
// into x (dInner) ‖ B (nGroups·N) ‖ C (nGroups·N). dInner = nHeads·P.

// Mamba2State is the per-layer recurrent state (the conv window and the SSM state), a second kind of recurrent state in
// the hybrid cache beside DeltaState.
type Mamba2State = mamba2.State

func NewMamba2State(p archparts.Mamba2Params) *Mamba2State { return mamba2.NewState(p.Mixer()) }

// mamba2Step advances one token through the Mamba-2 mixer, reading and updating st,
// returning the layer output [hidden].
func mamba2Step(h []float32, w *archparts.Mamba2Weights, p archparts.Mamba2Params, eps float64, st *Mamba2State) []float32 {
	dInner := p.DInner()

	// 1. in_proj → [z (gate), xBC, dt].
	inProj, outProj := w.InProj, w.OutProj
	if SSMQ8CPU { // confirmation seam: match the resident W8A8 — int8 weights AND int8 activations
		inProj = q8RoundTrip(inProj, p.ProjDim(), p.Hidden)
		outProj = q8RoundTrip(outProj, p.Hidden, dInner)
		h = q8RoundTrip(h, 1, len(h)) // the in_proj activation (resident rmsQuant → int8)
	} else if SSMQ8WeightsOnly { // W8A16 control: int8 mamba WEIGHTS, f32 activations (no act quant)
		inProj = q8RoundTrip(inProj, p.ProjDim(), p.Hidden)
		outProj = q8RoundTrip(outProj, p.Hidden, dInner)
	}
	var proj []float32
	if w.InQ.Rows() > 0 {
		proj = MatvecWM(nil, &w.InQ, h)
	} else {
		proj = hybridattn.Matvec(inProj, p.ProjDim(), p.Hidden, h)
	}

	// 2–5. The conv, the selective scan and the gated norm (internal/mamba2).
	gated := mamba2.Mix(proj, w.Mixer(), p.Mixer(), eps, st, SSMForceF32)
	if SSMQ8CPU {
		gated = q8RoundTrip(gated, 1, len(gated)) // the out_proj activation (resident quant → int8)
	}
	if MambaCapHook != nil { // test seam: capture real (in_proj output, mixer output) for GPU-kernel parity replay
		MambaCapHook(proj, gated)
	}
	if w.OutQ.Rows() > 0 {
		return MatvecWM(nil, &w.OutQ, gated)
	}
	return hybridattn.Matvec(outProj, p.Hidden, dInner, gated)
}

// MambaCapHook (test seam, gpu/mamba_realinput_test.go) captures each mamba2Step's in_proj
// output and mixer output (post gated-norm, pre out_proj) so real per-token inputs can be
// replayed through the GPU conv/ssm/gatedNorm kernels — the "green isolation / broken in-model"
// reconciliation. nil in production.
var MambaCapHook func(proj, gated []float32)

// SSMQ8CPU + q8RoundTrip are a confirmation seam, set by tests through SetSSMQ8CPU: round-trip the
// Mamba projections through int8 so the CPU reference carries the SAME quantization error
// as the resident W8A8 path — isolating kernel-correctness from int8 sensitivity.
var SSMQ8CPU bool

// SSMQ8WeightsOnly (W8A16 control, set by tests) round-trips the mamba in/out_proj WEIGHTS
// through int8 but keeps activations f32 — int8-weights / f32-activations, the W8A16 question.
var SSMQ8WeightsOnly bool

// SSMForceF32 (precision-localization seam, set by tests) downcasts the SSM compute
// in mamba2Step to f32 (mamba2.Mix's f32) — the exp(dt·A) argument and the gated-norm
// accumulation — to reproduce the GPU's f32-only SSM on the CPU reference. Off in production.
var SSMForceF32 bool

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

// Mamba2Seq runs the mixer over a whole sequence from a fresh state — a thin loop
// over mamba2Step, the form the parity test exercises (the forward path uses the
// same step with a cached state).
func Mamba2Seq(h [][]float32, w *archparts.Mamba2Weights, p archparts.Mamba2Params, eps float64) [][]float32 {
	st := NewMamba2State(p)
	out := make([][]float32, len(h))
	for t := range h {
		out[t] = mamba2Step(h[t], w, p, eps, st)
	}
	return out
}
