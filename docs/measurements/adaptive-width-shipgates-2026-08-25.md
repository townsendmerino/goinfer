# Adaptive verify-width: ship-gates FAILED on all four suites — do not flip default-off

> Measured 2026-08-25 on `linux-62gb` (RTX 2070 SUPER), goinfer `d5c8ee8`+, CUDA-resident
> Qwen3-4B int4 + DFlash-f32, `maxNew=96`, greedy, real `BlockSpec` path (not the
> `dflashLoop` copy in `cuda/drafter_loop_test.go`). Harness:
> `cuda/adaptive_width_gate_test.go`. Answers Phase 1 of
> `docs/prompts/adaptive-speculation.md`.

## Verdict

**Adaptive loses to the best static width on every suite, and on two of them nothing beats
running no drafter at all.** The controller works exactly as designed — it settles where its
target function says — and the design is wrong. Default-off stays, and now has a measured
reason rather than caution.

| suite | best arm | adaptive | verdict |
|---|---|---|---|
| chat | `static3` **1.000×** | 0.824× | adaptive −18%, **and off ties the best arm** |
| code | `static7` **1.502×** | 1.412× | adaptive −6% |
| math | `static7` **1.680×** | 1.589× | adaptive −5% |
| mixed | `static4` **0.943×** | 0.898× | adaptive −5%, **and NOTHING beats off** |

All ratios are against **OFF** — plain resident greedy with no drafter (off ≈ 103–107 tok/s
across suites). Including `off` as a competitor was decisive and is the reason two of these
rows say something the static-only comparison could not.

## Three findings, in order of what they cost

### 1. On losing suites, STOPPING beats NARROWING — the binary guard was encoding something real

This was the open question behind shipping default-off, and it is now answered against the
controller.

**Read the static column correctly:** with `Adaptive:false` the controller pins
`min == max == width`, so its floor case (`cur <= min && avg < breakEven`) fires and the arm
**stops and falls back to plain decode**. The static arms are therefore *guarded* static —
the shipped behaviour. So chat's `static3 1.000×` is not width 3 winning; it is the guard
correctly switching the drafter off and landing at parity with no drafter.

Against that, adaptive narrows to ~4 and keeps drafting, and pays for it: **0.824× on chat**,
worse than either endpoint it moves between (`static8` 0.887×, `static4` 0.993×). A workload
below break-even does not become profitable at a narrower width; it just loses more slowly
while still paying a draft and a verify every round.

### 2. The mixed suite — the case the whole idea exists for — does not pay at ANY width

Every arm loses to off; the best is `static4` at **0.943×**. This is not a tuning result and
it is not adaptivity's fault: block drafting simply does not pay on prose→structured content
with this pairing. The premise the campaign was built on — that a prose→structured boundary
is where a fixed width cannot win and adaptivity can — assumed the structured half is the
high-acceptance regime. **On this pairing it is not.**

The windowed trace says so directly. Width and committed tokens *fall* across the transition:

```
"Explain ... why hash tables are fast, then output a JSON object"
    transition @ round 19/26   width 5.3 -> 4.0   committed after: 2.83
"In one sentence, say what a CSV file is. Then output five rows of CSV"
    transition @ round 10/26   width 6.5 -> 4.0   committed after: 2.17
```

The predicted failure mode was a **lag**: a cumulative average weighting round N at 1/N, left
drafting wide into a regime that no longer supports it. That is not what happened — the
controller tracked *downward* correctly. The assumption that broke is upstream of the
controller: **JSON and CSV are not more predictable to this drafter than prose is.** The
llama.cpp fork's structured-content result was obtained on different hardware with a
different drafter, and this is the part that did not transfer.

### 3. A shippable win that has nothing to do with adaptivity: `defaultVerifyWidth` should be 7, not 8

`static7` is the best arm on **both** winning suites, and by more than the noise between
neighbours:

| suite | static7 | static8 (today's default) | gain |
|---|---|---|---|
| code | 1.502× | 1.402× | **+7.1%** |
| math | 1.680× | 1.598× | **+5.1%** |

The current default of 8 was chosen as "the one number that works everywhere" from math's
then-measured optimum. On this pairing 7 dominates it on both suites that pay. That is a
free ~5–7% for every `--drafter` user, independent of everything else here.

**NOT changed in this commit.** One session, one pairing, two prompts per suite — the same
evidentiary bar this campaign applied to its own controller applies here. It is filed as a
queue item with these numbers, needing a second pairing before the default moves.

## What this means for the target function

`width = committed + 2` was fitted to three measured points and flagged in the code as the
first thing to suspect. It is the thing to suspect: adaptive settles at 8 on math (avg 5.88 +
2) and 8 measures 1.598× where 7 measures 1.680×. The fit was one position too generous.

**But correcting +2 to +1 would not save Phase 1.** It would move adaptive from 1.589× to at
best `static7`'s 1.680× on math — matching the best static, not beating it — while chat and
mixed lose for a different reason entirely (narrowing instead of stopping, and a suite where
nothing pays). A controller that at best ties the best constant is not worth its complexity.

## Disposition

- **Phase 1 controller: NOT shipped default-on.** It stays behind `Adaptive` (default false),
  which the unit gates prove is byte-identical in behaviour to the guard it would replace.
- **Phase 2 (grammar prior) is not started, and its premise is now in question.** It assumed
  grammar-active means the high-acceptance regime, known at request time. Finding 2 measured
  the opposite on this pairing — structured output was the LOWER-acceptance half. Phase 2
  should begin by testing that premise directly on a constrained-JSON suite, and be prepared
  to stop there.
- **Phase 3 (MTP self-drafting) is untouched** by this result: it changes what drafts, not
  how wide. Its own kill-gates stand.
- The mixed-content suite is committed to the harness either way — it is the suite that
  showed the premise does not hold, and it should run against any future drafter change.
