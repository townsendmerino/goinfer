# A/B pre-registration — aikit v1.16.0 vs v1.17.1, BenchmarkDecode

**Written 2026-08-12, BEFORE the first sample of this comparison exists.** Unlike the v1.17.0
pre-registration — which was written mid-run with five samples already visible and says so — nothing
of this comparison has been observed at the time of writing.

## The expectation, stated up front because that is the risk

**I expect flat.** aikit v1.17.1 reverts the eight-column W8A8 span, and `w8a8Span`'s executable body
is byte-identical to v1.16.0 (extracted from both tags and diffed; the entire difference is 38 comment
lines). Upstream reports parity restored at both production shapes.

Knowing the expected answer is precisely when a reading bends toward it — a marginal result gets
called flat, an awkward sample gets a reason to be excluded. So the decision rule and every branch are
fixed here, in writing, before any data.

## Instrument — identical to the v1.17.0 A/B, deliberately

`BenchmarkDecode`, DeepSeek-V2-Lite-Chat-Q4_K_M at `Quant: "int8int8"` (W8A8), `-benchtime 30x`,
batch=1 greedy. Ryzen 7 3700X, GOMAXPROCS 16, linux/amd64. Both arms in detached git worktrees,
`GOWORK=off`, no `replace`, aikit resolved from the module cache at the version under test.
**Interleaved pre/post/pre/post in one session on one box.**

**Warm-up discard: the first sample of each visit**, carried over unchanged from the recorded
calibration rather than re-derived — re-deriving it now, against data I have an expectation about,
is exactly the freedom this rule exists to remove. Applied identically to both arms. 6 retained each.

**Floor: 2.0%** of the pooled mean, carried over unchanged from the same calibration (8-sample
characterization, relative sd 0.82%, so ≈2.4σ).

**Statistic: median of the 6 retained samples per arm.**

## Primary comparison: v1.16.0 vs v1.17.1. All three branches, fixed now

1. **Within ±2.0%** → **flat.** The regression is resolved. Recorded as "below this instrument's
   noise floor" — a result, not an absence of one, and P9 closes on it.
2. **Still regressed (v1.17.1 slower by ≥2.0%)** → the fix did not cover goinfer's shape. Reported
   upstream with the same discipline as the first: direction, magnitude, method, and the explicit
   statement of what was not isolated.
3. **Faster than v1.16.0 by ≥2.0%** → a win, and it gets the **same scoping a loss would**: decode
   only, one model, one quantization, batch=1, one box. Not published as a speedup, and not carried
   into CHANGELOG, docs or release notes on this evidence.

**A secondary v1.17.0 vs v1.17.1 arm runs ONLY if branch 2 or 3 fires** — i.e. only if the primary
does not come back flat, to establish whether the patch moved anything at all. Under branch 1 it is
not needed and will not be run to manufacture a second number.

## What this cannot answer, and what that gates

`linalg/matmul_blocked.go` is **unchanged in v1.17.1** — the f32 blocked-matmul rework from v1.17.0
is still live. That path is a **prefill** shape which this decode instrument barely exercises, so
**no result here says anything about prefill, in either direction, for v1.17.0 or v1.17.1.**

Consequently every statement of this result names **decode** explicitly, and a prefill measurement is
owed before a goinfer tag carrying this bump: a release characterizing one phase while silently
carrying an unmeasured change to another is a claim by omission.
