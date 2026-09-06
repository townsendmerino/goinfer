# Quantization: what goinfer stands behind, and why

**Status: policy, 2026-09-06.** This page states a stance. Every number in it is measured and
points at the record that measured it; where there is no evidence, it says so rather than
extrapolating.

---

## The short version

| | |
|---|---|
| **Recommended, gated, default** | `int4` (W4A8) for dense models · `int8int8` (W8A8) where accuracy matters more than RAM |
| **Supported with a caveat** | `int4mix` (attention int8, FFN int4) · `f32` (the reference, and the slowest thing here) |
| **Read but not endorsed** | 15 ggml types, including Q2_K and Q3_K |
| **Measured and refused** | 3-bit KV cache · resident int8 for Mamba-2 hybrids · weight-space per-row IMMA scale search |

## Reading a format is not standing behind it

This is the distinction that makes goinfer's answer different from a runtime that computes in the
format it loads.

goinfer **reads** 15 ggml types — `F32`, `F16`, `Q4_0`, `Q4_K`, `Q5_0`, `Q5_K`, `Q6_K`, `Q8_0`,
`Q2_K`, `Q3_K`, `IQ2_S`, `IQ3_S`, `IQ4_NL`, `IQ4_XS`, `MXFP4` — and then **requantizes** to one of
five resident precisions it actually computes in: `f32`, `int8`, `int8int8`, `int4`, `int4mix`.
The on-disk type is a transport question; the resident precision is what the arithmetic runs at.

So the honest floor is set by **whichever of the two is lossier**, and reading a Q2_K file at
`--quant int8int8` does not buy you int8 quality — it buys you Q2_K quality computed carefully.
The reverse is also true and less obvious: a pristine f32 checkpoint run at `int4` is bounded by
int4, not by the source.

**We deliberately do not refuse low-bit sources**, and copying a "no Q1–Q3" rule would contradict
our own practice: goinfer's real-model gates for GLM-4.5-Air (106B-A12B) and Llama-4-Scout
(109B) both start from **Q2_K** checkpoints, because that is the only form in which those models
fit the hardware they were validated on. Refusing to read them would not have raised anyone's
quality; it would have removed two families from the gated set. What we do instead is decline to
*recommend* them, and say what is and is not known about them — below.

## What we recommend, and the measurement behind it

### `int4` (W4A8) — the default

int4 weights with int8 activations. It is the default because it was measured to be the best
accuracy-per-byte on the shapes this project targets, not because it is smallest.

- **Nemotron-H, int4-resident vs f32:** 92.5% greedy agreement, ppl 1.677 against f32's 1.695,
  KL 0.058 — and, decisively, **top-2 agreement 99.6%**, with 16 of 17 disagreements picking f32's
  own #2 and **zero** harmful. That is what flipped it default-on
  ([`nemotron-resident.md`](nemotron-resident.md)).
- **Coherence at scale:** Llama-4-Scout 109B from a Q2_K source at int4 generates coherent,
  factual text; gemma4-26b at int4 is fully coherent once its real chat template is rendered.

> **A trap worth naming, because it cost real time:** an apparent int4 "quality deficit" on
> gemma4-26b was entirely an evaluation artifact — raw completion prompts on an instruction-tuned
> model are off-distribution. Render the model's chat template before judging quantization
> coherence, and do not read greedy repetition as a broken quantizer.

### `int8int8` (W8A8) — when accuracy matters more than RAM

int8 weights and int8 activations on native SDOT/VNNI. Higher accuracy, and **required** for the
dense Metal resident path (int4 declines to CPU there).

> **On Apple Silicon it is also the SMALLER option, which contradicts the intuition and our own
> older help text.** Measured 2026-09-06 in CI: int4 costs **1.2500 bytes/element** on arm64
> against int8's **1.0156**, because the arm64 W4A8 row4 repack keeps a second buffer beside the
> canonical nibbles. int4 is still the *faster* option there — that repack is what buys the speed
> — so the trade on Apple Silicon is "int4 is faster and larger", not "int4 is smaller".
> See [`task-first-hour.md`](task-first-hour.md).

### `int4mix` — attention int8, FFN int4

A load-time policy, not a resident precision: attention (plus embed/head/router) stays int8 where
a calibration spike found the int4 loss concentrated, and the FFN bulk goes int4. GGUF load path
only. Near-int8 quality at below-int8 RAM.

## What we refuse, and why — each one measured, not assumed

A stance is only worth stating if something is actually excluded. These were funded, measured, and
closed:

- **3-bit KV cache (TurboQuant).** Spiked and killed: **22–30× worse than int8 KV**, even with a
  foldable Hadamard. int8 KV shipped; int4 KV is deferred behind it because int8 was sufficient.
  The NO-GO still stands as of the 2026-09-02 audit (L-10).
- **Resident int8 for Mamba-2 hybrids.** Granite's resident int8 decode sat at **66% agreement**
  with f32 and the gap proved **precision-invariant** — W8A8 66% ≈ f16 64% ≈ W8A16 — so it is not
  a precision knob but distributed accumulation over a 40-layer MoE stack. Closed, opt-in,
  greedy-only, and explicitly *do not re-fund* ([`nemotron-resident.md`](nemotron-resident.md)).
- **Weight-space per-row IMMA scale search.** A 1.24× win in weight space; the forward gate put it
  at **ppl 108 against 28.5**. The cheap path died there, and the lesson stuck: a weight-space
  proxy can look fine while the forward is destroyed.

## Why MoE changes the answer

The single most useful thing this project has measured about low-bit inference is that **routers
have a cliff dense models do not.**

A **bit-identical** router still flips its top-k selection in **779 of 3200 (step, layer) pairs**
under ~0.5% input noise — routing flips are pervasive, not rare near-ties. So on a MoE, a small
weight perturbation is not a small output perturbation: it can select a different expert entirely,
which is discrete.

Nemotron-H is the control that confirms the mechanism: it has no MoE router, and its int4
perturbations stayed **smooth** — 92.5% agreement with benign disagreements — where granite's
MoE stack hit a wall. That is why `int4` is the confident default on dense and why low-bit MoE
carries a caveat rather than a blanket blessing.

**Practical consequence:** treat a quantization result on a dense model as saying nothing about a
MoE of the same size, and floor the *mean* rather than the min-over-N when judging router
stability.

## What we do not know

Stated so nobody reads absence as endorsement:

- **No perplexity or agreement numbers for the Q2_K/Q3_K/IQ2_S/IQ3_S read paths.** They are gated
  for *correct unpacking* — the bytes become the numbers the reference library says they should —
  and for end-to-end coherence on the two real checkpoints named above. Nobody has measured what
  those sources cost in quality against their own f16 originals, and this page will not guess.
- **`--embed-int4`** is disclosed as lossy at roughly **2.3 points of top-1**, mostly on rare
  tokens. It is off by default for that reason.
- The recommendation is for the families and shapes in `parity_manifest.json`. A family outside
  the gated set inherits no promise from this page.

## Sources

[`nemotron-resident.md`](nemotron-resident.md) (int4 default-on evidence; the granite contrast) ·
[`task-first-hour.md`](task-first-hour.md) (the arm64 int4/int8 footprint inversion) ·
[`audit-2026-09-02.md`](audit-2026-09-02.md) L-10 (the 3-bit KV NO-GO) ·
[`release-1.0-gate.md`](release-1.0-gate.md) (router flips, 779 of 3200) ·
[`benchmarks.md`](benchmarks.md) (speed rows and their provenance) ·
`internal/serveapp/main.go` (`-quant`, the flag this page explains)
