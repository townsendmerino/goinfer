package decoder

import (
	"fmt"
	"os"
	"sync"
)

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

// N-27: THE BUFFERS ABOVE ARE PACKAGE-LEVEL AND WERE APPENDED FROM INSIDE THE FORWARD WITH NO
// LOCK AND NO BOUND.
//
// Two separate problems, and the diagnostic framing hid both. Under the documented
// concurrent-sequence contract two goroutines can be in a forward at once, so the appends are a
// data race on a slice header — a crash, not a wrong number. And there is no cap: set on a long
// running `serve` process this grows without limit, one entry per MoE decision per layer per
// token, each carrying a copy of a hidden-sized vector.
//
// Fixed here rather than by refusing under `serve`: the decoder cannot see who its caller is,
// and a diagnostic that is safe everywhere is better than one that is refused in the one place
// it is dangerous. The mutex removes the race; the cap turns an unbounded leak into a bounded
// buffer that says when it stopped.
var (
	routerCaptureMu  sync.Mutex
	routerCaptureOff bool // set once the cap is hit, so the warning prints once
)

// routerCaptureMax bounds each buffer. Generous for the diagnostic's actual use — a
// teacher-forced pass of a few hundred tokens over 30 layers — and small enough that a
// forgotten env var on a serving process costs bounded memory instead of the process.
const routerCaptureMax = 1 << 16

// routerCaptureDo runs fn under the capture lock if capture is on and the cap is not reached.
// Every append site goes through it, so neither the lock nor the bound can be forgotten at one
// of the six.
func routerCaptureDo(fn func()) {
	if !routerCapture {
		return
	}
	routerCaptureMu.Lock()
	defer routerCaptureMu.Unlock()
	if len(routerCaptureBuf) >= routerCaptureMax {
		if !routerCaptureOff {
			routerCaptureOff = true
			fmt.Fprintf(os.Stderr, "[goinfer] GOINFER_ROUTER_CAPTURE: reached %d decisions; "+
				"capture stopped (it is a diagnostic, not a log — unset the variable)\n",
				routerCaptureMax)
		}
		return
	}
	fn()
}

// routerCaptureReset clears every buffer and re-arms the cap. The realckpt capture test calls
// it between passes.
func routerCaptureReset() {
	routerCaptureMu.Lock()
	defer routerCaptureMu.Unlock()
	routerCaptureBuf, routerRnBuf, routerMarginBuf = nil, nil, nil
	routerWtsBuf, routerX1Buf, routerX2Buf = nil, nil, nil
	routerCaptureOff = false
}
