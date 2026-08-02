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

// routerMarginBuf records, per MoE decision (same order/index as routerCaptureBuf), the top-k
// BOUNDARY MARGIN: the smallest selected expert's softmax prob minus the largest REJECTED
// expert's prob. This is the quantity that decides whether a small quant perturbation flips the
// top-k — the MoE-specific failure mode. A resident-gate-ready fixture wants this margin to stay
// well above the per-decision quant perturbation on every decision, not merely to AGREE on one
// int4-vs-f32 pair (agreement can be luck; a wide margin is robustness). Captured only when
// routerCapture is on; observe-only, so the forward stays byte-identical with the env unset.
var routerMarginBuf []float32
