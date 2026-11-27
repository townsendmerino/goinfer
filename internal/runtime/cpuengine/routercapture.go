package cpuengine

import (
	"sync"

	"github.com/townsendmerino/goinfer/internal/runtime/logs"
)

// RouterCapture is a test seam (default off; tests set it directly): when on, Gemma4MoEFFN appends each MoE-layer call's
// selected top-k expert indices to RouterCaptureBuf, in call order (token-outer, layer-inner: one entry per layer per
// token). It is observe-only: it copies out idx and changes no compute, so with it off the forward is byte-identical.
// Probe #1 uses it to tell routing collapse from uniform weight noise: capture
// selections for an int8 run and a 4-bit run over the same teacher-forced token sequence, then compare per-layer top-k
// overlap and selection entropy.
var RouterCapture bool

// RouterCaptureBuf accumulates the selected expert-index sets when RouterCapture is on.
// Order is deterministic: for token t and layer l (0-based, 30 gemma4 layers), the entry is
// at index t*nLayers + l. The realckpt capture test clears it (RouterCaptureBuf = nil) before
// a pass and reads it after — no helper accessors, so nothing here is unused off-tag.
var RouterCaptureBuf [][]int

// RouterRnBuf records, per MoE decision (same order and index as RouterCaptureBuf), a copy of the finalized router input
// rn = weightless-norm(h) · routerScale · hidden^-0.5, the exact f32 vector that feeds routerProj. A CUDA
// resident-router unit test replays these through the device selection kernels and gates resident idx[] against the
// CPU's (RouterCaptureBuf), isolating a routing flip from any expert-GEMV numeric difference. Captured only when
// RouterCapture is on.
var RouterRnBuf [][]float32

// RouterMarginBuf records, per MoE decision (same order and index as RouterCaptureBuf), the top-k boundary margin: the
// smallest selected expert's softmax prob minus the largest rejected expert's. It decides whether a small quant
// perturbation flips the top-k, so a resident-gate fixture wants it well above the per-decision quant perturbation on
// every decision, not merely to agree on one int4-vs-f32 pair (agreement can be luck). Captured only when RouterCapture
// is on.
var RouterMarginBuf []float32

// RouterWtsBuf / RouterX1Buf / RouterX2Buf capture the other three gemma4-MoE-layer intermediates
// (same append order as RouterRnBuf): the renormalized+scaled top-k weights, the dense-branch output
// x1 (post postFFNNorm1), and the expert-branch output x2 (post postFFNNorm2). With RouterRnBuf they
// are the four buffers a resident-vs-CPU whole-forward miss diffs against to localize router vs dense
// vs expert vs join. Observe-only under RouterCapture.
var (
	RouterWtsBuf [][]float32
	RouterX1Buf  [][]float32
	RouterX2Buf  [][]float32
)

// The buffers above are package-level and appended from inside the forward, so they are guarded and bounded. Under the
// documented concurrent-sequence contract two goroutines can be in a forward at once, so unlocked appends race on a
// slice header (a crash, not a wrong number); and without a cap a long-running serve process would grow them without
// limit, one entry per MoE decision per layer per token, each carrying a hidden-sized copy. The decoder cannot see who
// its caller is, so the diagnostic is made safe everywhere rather than refused under serve: the mutex removes the race
// and the cap turns an unbounded leak into a bounded buffer that says when it stopped.
var (
	routerCaptureMu  sync.Mutex
	routerCaptureOff bool // set once the cap is hit, so the warning prints once
)

// RouterCaptureMax bounds each buffer: generous for the diagnostic's use (a teacher-forced pass of a few hundred tokens
// over 30 layers) and small enough that capture left on in a serving process costs bounded memory instead of the
// process.
const RouterCaptureMax = 1 << 16

// RouterCaptureDo runs fn under the capture lock if capture is on and the cap is not reached.
// Every append site goes through it, so neither the lock nor the bound can be forgotten at one
// of the six.
func RouterCaptureDo(fn func()) {
	if !RouterCapture {
		return
	}
	routerCaptureMu.Lock()
	defer routerCaptureMu.Unlock()
	if len(RouterCaptureBuf) >= RouterCaptureMax {
		if !routerCaptureOff {
			routerCaptureOff = true
			logs.Infof("[goinfer] GOINFER_ROUTER_CAPTURE: reached %d decisions; "+
				"capture stopped (it is a diagnostic, not a log — unset the variable)",
				RouterCaptureMax)
		}
		return
	}
	fn()
}

// RouterCaptureReset clears every buffer and re-arms the cap. The realckpt capture test calls
// it between passes.
func RouterCaptureReset() {
	routerCaptureMu.Lock()
	defer routerCaptureMu.Unlock()
	RouterCaptureBuf, RouterRnBuf, RouterMarginBuf = nil, nil, nil
	RouterWtsBuf, RouterX1Buf, RouterX2Buf = nil, nil, nil
	routerCaptureOff = false
}
