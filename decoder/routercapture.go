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
// at index t*nLayers + l. The test resets it before a pass and snapshots it after.
var routerCaptureBuf [][]int

func routerCaptureReset()            { routerCaptureBuf = nil }
func routerCaptureSnapshot() [][]int { return routerCaptureBuf }
