# R1 — Metal W4F16 decode GEMV: kernel proven correct, catastrophic bug found and localized, root cause not yet found

**R1** (`docs/tasks/red-october.md`): a Metal decode GEMV in the f16-FMA regime MLX and llama.cpp
use, instead of the shipped W4A8 (int8-activation) regime. Unblocked by the 2026-09-18 owner
decision (option 2, default-on above the §3.2 pooled fidelity gate). This record is the build
attempt's full trail: what was built and proven, what broke, and what was ruled out before the
investigation was parked without a root cause.

**Status: PARKED.** The kernel itself is correctness-verified in isolation. The dense-path
dispatch wiring is correct at every layer checked except layers 26–27 of qwen2.5-coder-1.5b's 28,
where it diverges catastrophically (cosine goes negative) from a clean, unperturbed input — a real
bug, not a measurement artifact, not accumulated drift, and not model-level chaos (all three were
tested directly and ruled out below). The mechanism was localized to the gate/up GEMV specifically
but not found. Gate (3), the real fidelity gate, was never reached — the lane doesn't clear the
much more basic bar of "produces roughly sane output on one real checkpoint."

## What was built and scoped

First-slice scope, agreed before writing any kernel code: the plain dense (non-MoE, non-paged,
non-sandwich, non-postOnly, non-parallelBlock, non-qGate, non-outBias, non-layerNorm, no
compute-time LoRA) QKV / o-proj / gate-up path only — matching `canUseF16Lane` in `metal/model.go`.
Down-proj (`gemv_w4a8_resid`, the "coal" family — a different kernel shape) and every special-case
family stay on the shipped W4A8 kernels unconditionally. This was a deliberate narrowing from the
brief's full scope, agreed given the real integration surface (13+ existing dispatch call sites)
turned out much larger than the brief's Build section implied.

**`metal/kernels.go`**: `gemv_w4f16_sa` / `_sa_bias` / `_sa_resid` (the weight-stream half — nibble
dequant via the exponent-bias trick named in the brief: pack a raw nibble into a half's mantissa
bits with a fixed exponent, giving `1.0 + nibble/16`, subtract 1.5h and ×16 recovers `nibble-8`
exactly for all 16 values); `rmsnorm_f16_act` (QKV/gate-up's norm, no quantization, half output
instead of the int8 `aq`+scale pair `rmsnorm_quant` produces); `f32_to_f16` (o-proj's input has no
norm/weight — quant_vec is a bare quantizer — so this lane needs a bare convert there instead).

**`metal/model.go`**: `canUseF16Lane(l int) bool`, new `resident` fields (`pRmsF16`, `pF32ToF16`,
`pSAf16`/`_Bias`/`_Resid`, `decodeLaneW4F16`, `axF16`/`mxF16`/`cxF16`), and the dispatch branch at
each of the three target call sites (QKV in `encodeAttentionResidualWith`, o-proj in the same,
gate/up in `encodeLayerResidualWith`) — gated behind `GOINFER_METAL_DECODE_LANE=w4f16`, off by
default, so every existing user's path is untouched. Pipelines are always built ("one binary
carries both arms," per the brief); only dispatch is conditional.

## Gate (1): kernel correctness — PASSED

`metal/gemv_w4f16_test.go`. Two independent checks, both against the real Metal compiler and GPU,
not simulated:

1. **`TestNib2HalfAllValues`** — the exponent-bias dequant trick recovers `nibble-8` exactly (bit-
   for-bit, not approximately) for all 16 nibble values, isolated in its own tiny dispatch.
2. **`TestGemvW4F16_vsScalarReference`** — all three kernel variants (base, `_bias`, `_resid`)
   scored against an independent Go scalar reference (not reusing any Metal-side code), reusing
   `packW4A8Row` (the same, already-proven-correct weight packer `gemv_w4a8_sa` itself uses) at
   Qwen2.5-1.5B's real shapes (K=1536, K=8960), N=64 rows, random weights and activations. Result:
   **cosine 1.000000, max abs diff ~1e-6 at K=1536, ~4e-6 at K=8960** — exactly at f32 rounding
   noise, no correctness issue in the kernel itself. Includes a fixed-input determinism check
   (two runs, same input, bit-identical output — Metal's `simd_sum` reduction order is fixed per
   dispatch shape).

`gofmt` / `go vet` / staticcheck clean throughout; the existing non-heavy `./metal/...` suite (135s,
no regressions) passes unchanged with the lane off (default) — the wiring does not disturb any
existing shipped path.

## The bug: real-checkpoint sanity check catastrophically fails

`TestW4F16Lane_sanity` (real checkpoint, greedy decode, W4A8 vs W4F16 interleaved, 8 positions,
repeated-token prompt): **cosine 0.48, identical at every position** — the two lanes diverge
sharply. This was NOT expected; the brief's own registered prediction was that the f16 arm would
be *closer* to a CPU f32 reference than W4A8 on most prompts, not far apart from W4A8 itself.

### Ruling out "it's just comparing the wrong layer"

The first read of this (comparing `r.qkv` after a full `Forward` call) was itself misleading — `r.qkv`
is scratch shared across all 28 layers, so that comparison captured the *last* layer's QKV, not
layer 0's. Isolating layer 0 directly (`r.encodeAttention`, bypassing the full `Forward` loop):
**cosine 1.000000, maxAbs 0.059** — layer 0 is fine. Tracing every layer's residual stream (`x`)
after each full `encodeLayer` call, independently on both lanes:

| layer | cosine | maxAbs |
|---|---|---|
| 0 | 0.99966666 | 0.051 |
| 1–25 | 0.99999877 – 0.99999950 | 14.7 – 28.7 |
| **26** | **−0.67197818** | **1589.13** |
| 27 | −0.54344852 | 1425.46 |

Layers 0–25: small, stable, expected quantization-noise-level divergence (the shape of noise you'd
want from a higher-fidelity-in-principle arm). Layer 26: catastrophic, sudden break — not a
gradual compounding.

### Hypothesis 1 — a "massive activation" channel overflowing half's precision: FALSIHED

The residual stream's own max abs value jumps to ~3620 by layer 2 and sits there through layer 25
(a well-documented LLM phenomenon), dropping sharply to ~1037 right at layer 26 — the point where
something clearly transforms or partially cancels that channel. The natural first theory: half's
absolute rounding step at magnitude ~3620 (~3–4 units) becomes large enough, at the point of
cancellation, to matter.

Checked directly: the actual *normed* activation entering layer 26's QKV projection has amax
**23.57**, not ~3620 — RMSNorm's own math (dividing by the RMS, which is dominated by that same
channel) washes the raw magnitude out; the normed value depends on the channel's trained *weight*
times √H, not on how large the raw residual got. The value the GEMV actually sees was never large.

Tested anyway: added a per-tensor amax scale to the *same* clean activation (dividing before
storing to half, multiplying the accumulated GEMV output back up afterward — the natural first fix,
mirroring how the weight side already uses a group scale), re-ran layer 26's QKV projection with
the *same, unmodified* kernel. Result: **maxAbs 0.307 unscaled vs 0.307 scaled, cosine 0.999268 vs
0.999268 — no difference.** Consistent with IEEE half's relative precision being scale-invariant:
multiplying by a constant doesn't change how many significant digits a value keeps. The scale idea
doesn't touch the actual mechanism, whatever it is.

### Hypothesis 2 — model-level chaos (tiny input differences amplify catastrophically at this point): FALSIFIED

If true, this would mean R1's kernel is irrelevant — *any* numerically-different-but-correct
implementation would trigger it. Tested directly, entirely within the trusted W4A8 path (no f16
kernel involved at all):

- **One ULP** flipped on one element of `x` entering layer 26, propagated through layers 26–27:
  **cosine 1.00000000, maxAbs ~1e-6** — no effect whatsoever. Rules out chaos at any perturbation
  scale near machine precision.
- The **real, measured (w4f16 − w4a8) difference vector** at layer 26's boundary (relative
  magnitude 0.855%) added directly to the clean W4A8 residual, propagated through layers 26–27
  using *only* W4A8 kernels: **cosine 0.998, maxAbs 41.7** — a real but mild divergence, nowhere
  near catastrophic.
- **Random noise of the same magnitude** as a control: **cosine 0.999, maxAbs 16.7** — similarly
  mild.

A perturbation of the size the f16 lane actually produces, fed through the *trusted* kernels, does
not reproduce the blowup. The model is not chaotically sensitive to a difference of this size at
this point — whatever's wrong is specific to what the f16 kernels themselves compute at layers
26–27, not to input sensitivity.

### Confirmed: the f16 kernels alone, from a clean input, reproduce it

Fed the exact clean, ground-truth `x1` (W4A8's own layer-25 output, no perturbation of any kind)
into the W4F16 lane starting at layer 26: **cosine −0.58513828, maxAbs 1467.18** — matches the
original break almost exactly. This is decisive: the bug is in what the F16 kernels compute at
layers 26–27, independent of any accumulated drift.

### Localized to the FFN block, most likely the gate/up GEMV

Layer 26's **attention block alone** (`r.encodeAttention`, isolated, clean input): **cosine
0.99999995, maxAbs 0.31** — fine, matching every other layer's fidelity. So the entry point is the
**FFN block**. Down-proj is unconditionally W4A8 in this slice (out of scope, unchanged), so the
only f16-touched part of FFN is gate/up. Reading `r.gu` (gate/up's raw GEMV output) directly after
a full `encodeLayer(26)` call from clean input: **cosine 0.9976** overall (worse than attention's
near-1.0, not yet catastrophic on its own) — **with at least one output row ~31% off** (row 2908:
54.7 want vs 71.6 got). The largest channel in the activation feeding this GEMV (idx 408, amax
146.6 — a real but unremarkable "massive-ish" channel, nothing like the earlier 3620 red herring)
was checked directly against row 2908's own weight at that channel: **the weight nibble is exactly
8, i.e. zero contribution** — so this specific outlier is not explained by "one dominant channel,
one row that weights it heavily" either. The investigation was parked here.

## What this record does and does not establish

**Established, with direct evidence:** the kernel's own math is correct; the dispatch wiring is
correct through at least 26 of 28 layers and through attention at layer 26 itself; the bug is real,
reproducible from a clean input, and specific to gate/up's GEMV (or its immediate norm producer)
at layer 26 particularly, not a property of the model's general sensitivity or of half precision's
range as such.

**Not established:** the actual mechanism. Candidates not yet checked: whether other layers' gate/up
GEMVs have the same defect at a smaller (masked-by-noise) scale and layer 26 is where it first
crosses into visibly catastrophic territory (rather than layer 26 being uniquely broken); whether
this is specific to `qwen2.5-coder-1.5b`'s particular trained weights at that layer or would show
up on any model given the right values; whether it's in `rmsnorm_f16_act` (the activation producer)
rather than the GEMV itself; whether it's a genuine kernel defect (a boundary condition in the
`SA_F16_BODY` reduction, a threadgroup-memory sizing issue that only bites at a specific K or row
count) as opposed to a numerics-design issue with no simple fix.

## Reproducer

`metal/gemv_w4f16_layer26_repro_test.go`, `TestW4F16Lane_layer26Reproduction`
(`GOINFER_HEAVY_TESTS=1 go test -tags goinfer_testhooks ./metal/ -run TestW4F16Lane_layer26Reproduction -v`).
Deliberately fails (loudly, by design) until this is fixed — reproduces the exact cosine from a
clean input in one run, so whoever picks this back up starts from a known-good minimal repro
instead of rebuilding one from scratch.

## Decision rule

Not reached. Gate (3) — the real teacher-forced fidelity gate against a CPU f32 reference — was
never run; there is no point running a rigorous statistical fidelity gate on a lane that fails a
basic "produces roughly sane greedy output" check first. R1's Build phase is **parked**, not
killed: the kernel-level correctness result (gate 1) and the precise localization here are real,
reusable groundwork, not a dead end — this is a debuggable, bounded problem (one FFN block, one
GEMV family, two adjacent layers), not a fundamental flaw in the lane's premise.

## Out of scope (unchanged from the brief)

The LM head, KV precision, any change to the W4A8 path, prefill, the 26B/35B paged path — plus, for
this narrowed slice specifically: down-proj (the "coal" family), MoE, paged families, Gemma
sandwich/postOnly/parallelBlock, qGate, outBias, compute-time LoRA.
