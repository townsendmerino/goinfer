# A/B pre-registration — aikit v1.16.0 vs v1.17.0, BenchmarkDecode

Written 2026-08-12 14:43, while the run was in progress. **Disclosure of what was already
visible:** round-1/pre's four samples (0.9545, 0.9621, 0.9629, 0.9723) and round-1/post's first
sample (0.9234). Rounds 1-post-remainder and all of round 2 were unobserved. This is therefore a
*partial* pre-registration and weaker than one written before the run started; it is recorded as
such rather than presented as clean.

The noise floor below is derived from the **8-sample characterization**, which was collected before
the A/B existed and is independent of it. That part is not contaminated.

## Instrument

`BenchmarkDecode`, DeepSeek-V2-Lite-Chat-Q4_K_M loaded at `Quant: "int8int8"` (W8A8),
`-benchtime 30x`, batch=1 greedy decode. Box: AMD Ryzen 7 3700X, GOMAXPROCS 16, linux/amd64.
Both arms: detached git worktrees, `GOWORK=off`, no `replace`, aikit resolved from the module
cache at the version under test. Interleaved pre/post/pre/post in one session.

## Warm-up discard — defined before the data, applied identically to both arms

**Discard the first sample of each process invocation (each visit).** Basis: the 3-sample run's
first sample was 0.8733 against a 0.94 plateau, and round-1/pre's first sample is its lowest of
four. Both arms pay the same discard, so the correction cannot favour either. Remaining: 3 samples
per visit × 2 visits = **6 per arm**.

## Noise floor — pre-registered

From the independent 8-sample characterization (0.9396, 0.9390, 0.9605, 0.9461, 0.9500, 0.9417,
0.9422, 0.9548): mean 0.94674, sample sd 0.00777 → **relative sd 0.82%**.

**Floor = 2.0% of the pooled mean** (≈2.4σ). Chosen above 2σ so a single stray sample cannot
manufacture a result.

## Decision rule — fixed now

Statistic: **median of the 6 retained samples per arm**.

- `|median_post − median_pre| < 2.0%` → **"below this instrument's noise floor."** That is the
  result. It gets recorded as a finding, not left open and not re-run until it moves.
- `≥ 2.0%` → a real difference at this shape, reported with direction and magnitude, attributable
  to the aikit bump because that is the only compiled-code change between the arms.

**Either way, no performance claim enters CHANGELOG, docs or release notes.** A regression is
reported to the user and upstream; a win stays unreported until it survives more than one shape.
This benchmark is one model, one quantization, batch=1, on one box.

## What this cannot answer

Decode at batch=1 with M=1. The v1.17.0 changes that reach goinfer are a new AVX2 int8 kernel
(`dotI8x8AVX2`, entered at `hasAVX2 && K>=16`) in `w8a8Span`, and a new f32 path
(`dot8ColsInto` + `blockRows3x4`) in the blocked matmul. The blocked f32 path is largely a
*prefill* shape; this instrument barely exercises it. A prefill benchmark would be a different
measurement and is not being substituted for this one.

---

# Result — appended after the run, analysis exactly as pre-registered above

Raw, in the order run. **Bold = the pre-registered warm-up discard** (first sample of each visit).

| visit | samples (tok/s) |
|---|---|
| 1 · pre  `v1.16.0` | **0.9545**, 0.9621, 0.9629, 0.9723 |
| 1 · post `v1.17.0` | **0.9234**, 0.9292, 0.9359, 0.9401 |
| 2 · pre  `v1.16.0` | **0.9634**, 0.9745, 0.9649, 0.9675 |
| 2 · post `v1.17.0` | **0.9522**, 0.9515, 0.8988, 0.9631 |

| arm | n | median | mean | sd | min | max |
|---|---|---|---|---|---|---|
| pre  | 6 | 0.9662 | 0.9674 | 0.0051 | 0.9621 | 0.9745 |
| post | 6 | 0.9380 | 0.9364 | 0.0220 | 0.8988 | 0.9631 |

**Median delta −0.0282 tok/s = −2.96% of the pooled mean (0.95190). The pre-registered floor was
2.0%, so this is ABOVE it: a real difference, with v1.17.0 slower.**

## What the numbers support, and what they do not

*Robust:* the direction. Per-visit means do not overlap — pre {0.9658, 0.9690} against post {0.9351,
0.9378} — and 34 of 36 pairwise comparisons put a post sample below a pre sample.

*Not robust:* the exact magnitude. The post arm's sd is 4× the pre arm's, concentrated entirely in
visit 2 (per-visit sd 0.0055 then 0.0343), with one low outlier (0.8988) and one sample (0.9631)
reaching into the pre range. Quote this as **~3%**, not 2.96%.

*Unresolved:* which of the two new kernels causes it. This shape is int8int8 at M=1, so the AVX2
int8 routine behind `w8a8Span` is the likely locus and the blocked f32 path is barely touched — but
no ablation was run, so that is inference. It is recorded here as inference and must not be repeated
as a finding.

## The honest weakness of this record

The pre-registration above was written **during** the run with round-1/pre and one round-1/post
sample already visible, and says so. The noise floor itself came from an independent characterization
collected before the A/B existed, so the floor is uncontaminated — but a fully clean pre-registration
would have been written before any sample was seen. Next time, write it first.
