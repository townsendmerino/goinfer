# goinfer task: does `defaultVerifyWidth = 8` hold on a SECOND pairing? (Mac cell)

> Written 2026-08-25 against goinfer `f49c6b1`. **Box: `macbook-arm64`.** This is a
> measurement task with a one-character outcome — it either clears a default change or kills
> it. Do not build anything.

> **UPDATED 2026-08-25 after a strengthened Linux sweep — read this before running.** At n=12
> paired (`docs/measurements/default-verify-width-2026-08-25.md`) the finding SPLIT: the code
> half replicates hard (+8.7% at int4 on 11/12 pairs, +8.6% at int8 on 10/12) and **the math
> half does not** (+3.1% on 6/12 — a coin flip). Width 8 is not optimal in any cell, but the
> optimum is **quant-dependent**: 7 at int4, **6** at int8. So the question here is no longer
> "does +7.1%/+5.1% reproduce". It is: **does 7 beat 8 on CODE, paired, on a different pairing —
> and where does that pairing's own quant sweep put the optimum?** Measure math too and expect
> little from it. Report PAIRED deltas with win counts; pooled means have prompt-to-prompt sd
> around 10–35 tok/s and hide the effect entirely.

## The finding to confirm or kill

The adaptive-width ship-gates (`docs/measurements/adaptive-width-shipgates-2026-08-25.md`)
failed — the controller is not shipping — but they produced one clean, shippable side result:

| suite | `static7` | `static8` (today's default) | gain |
|---|---|---|---|
| code | 1.502× | 1.402× | **+7.1%** |
| math | 1.680× | 1.598× | **+5.1%** |

`decoder.defaultVerifyWidth` is 8. If 7 is really better, that is free throughput for every
`--drafter` user. **It was deliberately not changed**, because the evidence was two prompts per
suite, one session, one target quant, on one pairing — the same bar the controller itself was
held to and failed.

## Why this needs the Mac specifically

**The Linux box has exactly one viable block-drafting pairing**, so it cannot supply the
second: `qwen3-4b` + `qwen3-4b-dflash-f32` is the only dense target that fits 8 GB with a
drafter on disk. `dflash2-qwen38` needs a 27B target that does not fit; the gpt-oss and gemma4
drafters are not present, and both of those targets are MoE — where a batched verify touches
~8× the expert weight, which is why Laguna measured 0.82× and why an MoE pairing is a poor
instrument for a width question.

So the Linux box is running a *strengthened same-pairing* sweep (6 prompts/suite, repeats,
**both int4 and int8 targets**) — results will be in
`docs/measurements/` — and the Mac owns the two things that box cannot give: a second machine,
and ideally a second pairing.

## What to run

**Minimum (the cross-box cell):** the same pairing on Apple Silicon. Whatever the Mac's
resident block-drafting path is, sweep `VerifyWidth` over `{5,6,7,8,9,10}` on the code and math
suites, interleaved within each prompt so thermal drift hits every width alike, ≥4 prompts per
suite and ≥2 repeats. Report mean tok/s and standard deviation per width, and the 7-vs-8 delta
against that spread. `cuda/default_width_sweep_test.go` is the shape to copy — it is CUDA-tagged,
so port it rather than importing it.

**Better, if a second pairing exists locally:** any dense target with a matching DFlash drafter
that is not qwen3-4b. That answers the question the Linux box structurally cannot.

## What would make this decisive either way

- **7 > 8 on the Mac too, beyond the spread** → change `defaultVerifyWidth` to 7, citing both
  boxes. One character, ~5–7%.
- **8 ≥ 7 on the Mac** → the Linux result is pairing- or box-specific. Do **not** change the
  default; record it as measured-and-not-shipped, which is a real outcome and costs nothing.
- **The best width differs by target QUANT** (the Linux sweep is testing this directly) → then
  no single constant is right and the honest answer is either a quant-dependent default or
  leaving 8 alone. Say so rather than picking the winner of whichever arm you ran.

## Do not

- Do not tune anything else while you are in there. This is one number.
- Do not read a speed number off the CPU acceptance harness (`TestDFlashAcceptance`) — it
  verifies with sequential single-token forwards and its own header says wall-clock does not
  transfer from it.
- Do not re-litigate the adaptive controller. It failed its gates, it is default-off, and its
  fate now rides on Phase 2's premise test, not on this.

## Deliverable

A short `docs/measurements/` note: the per-width table with spreads, the 7-vs-8 delta, the box
and pairing, and one sentence saying whether the default should move. If it should, make the
change in the same commit — `decoder/blockspec.go`, `defaultVerifyWidth`, and update the
comment above it, which currently justifies 8 from math's then-measured optimum.
