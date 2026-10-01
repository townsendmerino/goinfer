# TestQwen35GGUF_gate on amd64: bisected to aikit v1.50.2's int8 rounding (2026-10-01)

> **Status: root-caused and fixed.** The v0.20.0 release sweep's one numeric blocker came from a rounding defect in
> aikit's amd64 AVX2 int8 quantizer, not from goinfer. aikit v1.51.1 fixes it, and the gate's v0.19.0 result comes
> back exactly. The scoped re-validation on the fixed aikit is `run-scoped-revalidation.sh` (pre-registered in
> `2d0b664f`). All runs on nobara-pc (linux/amd64, Ryzen 7 3700X, 62 GB), go1.27.0, checkpoint
> `~/models/qwen3.6-35b-a3b-Q8_0.gguf` against the banked bf16 golden `~/models/qwen35_real_golden`.

## What failed

Sweep run 2 at `bcf50a49` (2026-10-01 06:24–08:48 PDT): **argmax 57/80** against the gate's 66 floor, 5/10 prompts
coherent, cosine min 0.98786 / mean 0.99619, worst divergence gap 0.0072 of the logit range. Every divergence was a
rank-2 near-tie. The same box passed it in the v0.19.0 sweep.

## Bisect

Pre-registered in `c1bc6328` (`run-bisect-qwen35-gguf.sh`): GOOD at argmax ≥ 66/80, v0.19.0 tested first, over the
164 commits touching `decoder/`, the root packages it imports (`constrain`, `internal/giw`, `tokenizer`) and
`go.mod`/`go.sum`. 08:48–09:51 PDT, 8 tests.

| commit | date | verdict | argmax | coherent | cosine min / mean |
|---|---|---|---|---|---|
| `c7f8eff7` (v0.19.0) | 09-18 | GOOD | 68/80 | 7/10 | 0.98740 / 0.99608 |
| `972d2ed5` | 09-25 | GOOD | 68/80 | 7/10 | 0.98740 / 0.99608 |
| `ee7bf67a` | 09-27 | GOOD | 68/80 | 7/10 | 0.98740 / 0.99608 |
| `84475acc` | 09-28 | BAD | 57/80 | 5/10 | 0.98786 / 0.99619 |
| `1c9a6054` | 09-28 | GOOD | 68/80 | 7/10 | 0.98740 / 0.99608 |
| `cd6ce336` | 09-28 | BAD | 57/80 | 5/10 | 0.98786 / 0.99619 |
| `96d0ef26` | 09-28 | GOOD | 68/80 | 7/10 | 0.98740 / 0.99608 |
| **`81c250d7`** | 09-28 | **BAD, first bad** | 57/80 | 5/10 | 0.98786 / 0.99619 |

Every GOOD is identical to the digit and so is every BAD: one deterministic change. `81c250d7` changes only the aikit
pin, v1.50.1 → v1.50.2, so the change is inside aikit. v1.50.1's production changes are arm64-only (its own release
note), and v1.50.2's one change is the amd64 AVX2 activation quantizer.

## Mechanism

v1.50.2's `quantizeF32AVX2` (aikit `linalg/quant_act_amd64.s`) rounds half away from zero as
`trunc(y + copysign(0.5, y))` in float32. The scalar code it replaced, and arm64's `FCVTAS`, round with
`math.Round`. For y = 0.49999997 (`nextafter32(0.5, 0)`) the exact sum 1 − 2⁻²⁵ is a float32 tie, which rounds to
even, i.e. to 1.0, so the kernel gives 1 where the reference gives 0 (and −1 for −0.49999997).

- **Crafted row on nobara's AVX2** (`qwen35-gguf-bisect/quant-probe.go.txt`: a row whose max is 127, so the scale is
  exactly 1 and every value is rounded as itself, through the public `linalg.QuantizeRowInt8`). aikit v1.50.1: 0 of 16
  values differ from the scalar reference. aikit v1.51.0: exactly two differ, ±0.49999997 → ±1. Exact .5 ties and the
  float32 values just below 1.5, 3.5 and 126.5 all match.
- **Exhaustive check** over every float32 with |y| < 2³¹ (Go, on the Mac). Adding 0.5 misrounds 8,388,610 values: ±0.49999997
  and the odd integers in [2²³, 2²⁴), which the kernel's ±127 clamp makes unreachable. Adding `nextafter32(0.5, 0)`
  misrounds none.
- **Reach.** The core is shared: activation quantization before every W8A8/W4A8 matmul, and int8 weight quantization at
  load (`QuantizeRowInt8`, `QuantizeRowsInt8`: int8 and int8int8 loads, the GGUF Q8_0 → int8 requant, `.giw` sidecars
  written on amd64). Both roundings are equally close to 0.49999997, so no output got less accurate. What changed is
  bit-identity with the scalar path and arm64, which moves near-ties. The f32 paths, arm64, Metal and CUDA never call
  this kernel. It never reached a goinfer release: v0.19.0 pins aikit v1.45.1, and goinfer moved to v1.50.2 on
  2026-09-28 (`81c250d7`).

The gate's argmax score is fragile to exactly this. A free-running greedy count over 8 steps turns one early near-tie
flip into up to 7 misses, and the flips go both ways: the defect cost prompts 0, 3, 6 and 7, and gained prompt 2
(2/8 → 8/8).

## Confirmation

Pre-registered in `f37b34f6` (`run-qwen35-gguf-roundfix.sh`): main at `f0cf6f2d` with aikit v1.51.0 replaced by a copy
patched in that one constant (`0x3F000000` → `0x3EFFFFFF`), predicted to reproduce every GOOD step exactly. Result,
09:57–10:03 PDT: **argmax 68/80, 7/10 coherent, cosine min 0.98740 / mean 0.99608, worst gap 0.0080, PASS**, and every
prompt's generated tokens identical to v0.19.0's (`qwen35-gguf-bisect/qwen35-gguf-roundfix.log` against
`step-c7f8eff76c.log`).

## The fix

aikit `5c144cc` (the constant, the corrected kernel comments, and a regression corner) and `3fbbf05` (the v1.51.1
CHANGELOG; perfgate exception by owner decision, since the instruction stream is unchanged; vulncheck clean, 16/16
modules). The new `every-rounding-boundary` corner feeds the float32 below, at and above every k+0.5 for k = 0..126,
both signs, at scale 1, through the vector path. On nobara it fails on the v1.51.0 kernel (`q[1] differs: scalar 0
dispatched 1 (row[1]=0.49999997, scale=1)`) and passes with the fix. aikit's existing random-row and corner tests
passed on both.

## The other blocker

`TestLagunaGGUF_gate` was refused by the fit guard in sweep run 2. With its per-model guard knob (`a6ce3d4a`) it passes
at `f0cf6f2d`, 259.7 s (`qwen35-gguf-bisect/laguna-gguf-rerun.log`).

## Files

`qwen35-gguf-bisect/`: `provenance.txt`, `progress.log` (one line per step), `bisect.log` and `bisect-run.log` (git's
own record), `step-<sha>.log` (each step's full test output), `qwen35-gguf-roundfix.log`, `laguna-gguf-rerun.log`,
`quant-probe.go.txt`.
