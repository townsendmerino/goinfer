# `defaultVerifyWidth = 8` is dominated everywhere — but the optimum is QUANT-DEPENDENT

> Measured 2026-08-25 on `linux-62gb` (RTX 2070 SUPER), goinfer `a1f6541`, CUDA-resident
> Qwen3-4B + DFlash-f32, `maxNew=96`, greedy. 6 prompts/suite x 2 repeats = **n=12 per width**,
> widths interleaved **within** each (prompt, repeat). Harness:
> `cuda/default_width_sweep_test.go`. Follows up the side-finding in
> `adaptive-width-shipgates-2026-08-25.md`.

## Result

| quant | suite | best width | 7-vs-8 (paired) | 7 wins |
|---|---|---|---|---|
| int4 | code | **7** | **+8.7%** (sd 6.5) | **11/12** |
| int4 | math | **7** | +3.1% (sd 7.6) | 6/12 |
| int8 | code | **6** | **+8.6%** (sd 8.7) | **10/12** |
| int8 | math | **6** | +1.9% (sd 5.5) | 8/12 |

**Width 8 is not the best width in ANY of the four cells**, and 7 beats 8 on the paired mean in
all four. But the optimum is 7 at int4 and **6** at int8.

## Read the paired column, not the pooled one

The per-width tables carry standard deviations around 10–35 tok/s, which makes an 8% effect look
like noise. That spread is almost entirely **prompt-to-prompt**: prompts differ in length and
content, and pooling them buries the comparison. Widths are measured interleaved within each
(prompt, repeat), so the matched observations difference cleanly — and the paired sd drops to
5.5–8.7 with a win count attached. Same data, and the two views disagree about whether there is
an effect at all. Pooled statistics were the wrong instrument here.

## Two corrections to the original finding

**1. The math half does not replicate.** The ship-gate run read `static7` beating `static8` by
+5.1% on math. At n=12 paired that is **+3.1% on 6/12 pairs at int4 — a coin flip** — and
+1.9% on 8/12 at int8. The original math number came from two prompts in one session, and it
does not survive. **The code half replicates strongly** (+8.7%/11-of-12, +8.6%/10-of-12), and
it is what carries the result.

**2. The optimum moves with target quant, which was the pre-registered "then no single constant
is right" case.** It moves for a mechanical reason: optimal width is set by the ratio between a
plain decode step and a batched verify, and int8 decode is roughly half the rate of int4 here
(code 82 vs 145 tok/s), which shifts where the tail positions stop paying.

## Recommendation, and what it still waits on

**7 is the best single constant.** It beats 8 in all four cells, so moving 8 → 7 is a strict
improvement even where 6 is optimal. A quant-aware default (7 at int4, 6 at int8) would gain a
little more at int8 and is a second, larger change with its own comment to justify.

**NOT changed here.** The standing bar is a **second pairing**, and this box structurally cannot
supply one: qwen3-4b dense + DFlash is the only viable block-drafting pairing that fits 8 GB —
`dflash2-qwen38` needs a 27B target that does not fit, and the gpt-oss/gemma4 drafters are absent
with MoE targets, where a batched verify touches ~8x the expert weight. What this run adds is a
second **condition** (quant) and 6x the statistics, not a second pairing. The cross-box/cross-
pairing cell is `docs/prompts/mac-default-verify-width.md`.

**What the Mac should now be looking for has changed** because of correction 1: the question is
no longer "does +7.1%/+5.1% reproduce" but **"does 7 beat 8 on code, paired, on a different
pairing — and where does its own quant sweep put the optimum"**. Math should be measured and
expected to show little.
