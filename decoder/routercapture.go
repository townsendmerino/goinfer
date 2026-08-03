package decoder

import "os"

// routerCapture (GOINFER_ROUTER_CAPTURE=1) is a DIAGNOSTIC (default-off, single load-time
// env read): when on, gemma4MoEFFN appends each MoE-layer call's selected top-k expert
// indices to routerCaptureBuf, in call order (token-outer, layer-inner — one entry per
// layer per token). It is OBSERVE-ONLY: it copies out `idx` and changes no compute, so with
// the env unset the forward is byte-identical.
//
// Probe #1 in docs/task-gemma4-moe.md uses it to tell routing collapse from uniform weight
// noise: capture selections for the int8 run and a 4-bit run over the SAME teacher-forced
// token sequence, then compare per-layer top-k overlap and per-layer selection entropy.
// Repetitive-English output is the signature of routing collapse, not of weight noise; if
// the 4-bit run's selections have degenerated vs int8, more bits on the expert weights
// won't close the gap and the plan redirects to router-input cleanliness.
var routerCapture = os.Getenv("GOINFER_ROUTER_CAPTURE") != ""

// routerCaptureBuf accumulates the selected expert-index sets when routerCapture is on.
// Order is deterministic: for token t and layer l (0-based, 30 gemma4 layers), the entry is
// at index t*nLayers + l. The realckpt capture test clears it (routerCaptureBuf = nil) before
// a pass and reads it after — no helper accessors, so nothing here is unused off-tag.
var routerCaptureBuf [][]int

// routerRnBuf records, per MoE decision (same order/index as routerCaptureBuf), a COPY of the
// finalized router input rn = (weightless-norm(h) · routerScale · hidden^-0.5) — the exact f32
// vector that feeds routerProj. It exists so a CUDA resident-router unit test can replay identical
// inputs through the device selection kernels and gate resident idx[] against the CPU idx[]
// (routerCaptureBuf), isolating a ROUTING FLIP from any expert-GEMV numeric difference — the
// "router first" discipline. Captured only when routerCapture is on; observe-only, byte-identical
// with the env unset.
var routerRnBuf [][]float32

// routerMarginBuf records, per MoE decision (same order/index as routerCaptureBuf), the top-k
// BOUNDARY MARGIN: the smallest selected expert's softmax prob minus the largest REJECTED
// expert's prob. This is the quantity that decides whether a small quant perturbation flips the
// top-k — the MoE-specific failure mode. A resident-gate-ready fixture wants this margin to stay
// well above the per-decision quant perturbation on every decision, not merely to AGREE on one
// int4-vs-f32 pair (agreement can be luck; a wide margin is robustness). Captured only when
// routerCapture is on; observe-only, so the forward stays byte-identical with the env unset.
var routerMarginBuf []float32

// SetRouterCaptureForTest toggles router capture and (when enabling) clears the buffers. Exported so
// the CUDA resident-router unit test (package cuda) can drive a CPU forward, capture idx/rn, and
// replay rn through the device kernels — it cannot touch these unexported vars directly.
func SetRouterCaptureForTest(on bool) {
	routerCapture = on
	if on {
		routerCaptureBuf, routerRnBuf, routerMarginBuf = nil, nil, nil
		routerWtsBuf, routerX1Buf, routerX2Buf = nil, nil, nil
	}
}

// RouterCaptureForTest returns the captured per-decision selected experts and router inputs (same
// order/index). Sibling to SetRouterCaptureForTest for the cross-package (cuda) router-first gate.
func RouterCaptureForTest() (idx [][]int, rn [][]float32) { return routerCaptureBuf, routerRnBuf }

// routerWtsBuf / routerX1Buf / routerX2Buf capture the other three gemma4-MoE-layer intermediates
// (same append order as routerRnBuf): the renormalized+scaled top-k weights, the dense-branch output
// x1 (post postFFNNorm1), and the expert-branch output x2 (post postFFNNorm2). With routerRnBuf they
// are the four buffers a resident-vs-CPU whole-forward miss diffs against to localize router vs dense
// vs expert vs join. Observe-only under routerCapture.
var (
	routerWtsBuf [][]float32
	routerX1Buf  [][]float32
	routerX2Buf  [][]float32
)

// Gemma4MoECaptureForTest returns the CPU per-decision wts / x1 / x2 (same order as RouterCaptureForTest).
func Gemma4MoECaptureForTest() (wts, x1, x2 [][]float32) {
	return routerWtsBuf, routerX1Buf, routerX2Buf
}

// RouterMarginForTest returns the per-decision top-k boundary margin (smallest selected expert's
// softmax prob minus the largest rejected expert's), same order/index as RouterCaptureForTest. It
// is the MoE-specific robustness signal a noise-floor check reads: a fixture whose margin sits below
// the int4-vs-f32 routing perturbation can flip top-k under quant and cannot gate a resident router,
// however correct the port (the reason the CUDA MoE fixture was rebuilt at 9275f94).
func RouterMarginForTest() []float32 { return routerMarginBuf }
