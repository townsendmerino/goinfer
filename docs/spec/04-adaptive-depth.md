# 04 — Adaptive draft depth

> Status: **proposal**. Depends on [00-core](./00-core.md) (the acceptance model and
> predictor). Cheap, broad win; can ship on top of any single drafter before the
> [03 router](./03-router-tree.md) exists.

## Idea

A fixed draft window `K` is wrong almost everywhere. After "the capital of France
is", or inside a grammar-forced run, or mid-way through a copied code line,
acceptance is near 1 and you should speculate far. At a genuine fork in reasoning,
acceptance is low and even one drafted token is likely wasted. Spend draft depth
**where acceptance predicts it pays**, and stop otherwise.

## The rule

From [00-core](./00-core.md) §2, the per-chain speedup is

```
S(γ) = (1 − α^(γ+1)) / [ (1 − α)·(γ·c + 1) ]
```

but `α` is not constant — it varies by position. Using the calibrated predictor
`α̂(features)` from [00-core](./00-core.md) §4, extend the draft while the **expected
marginal committed token exceeds the marginal cost** of drafting+verifying it:

```
keep drafting while   (∏ α̂ over the run so far) · target_step_time  >  marginal_cost(next node)
stop when it isn't
```

Equivalently: bound depth by a running acceptance product. In high-`α̂` regions the
product stays high and the run goes deep (10+); in low-`α̂` regions it collapses
after a token or two. This recovers `γ*` adaptively without a global constant, and
specializes to the actual `c` measured on the backend (CPU vs WebGPU) by the harness.

Cheap signals that drive `α̂` here (all draft-time, [00-core](./00-core.md) §3):
draft top-1 prob, draft entropy, depth-so-far, source, grammar-forced flag, n-gram
match length, current accept streak.

## Relationship to the literature

This is the principled version of EAGLE-2's dynamic draft trees and of GammaTune's
adaptive-`γ` calibration (arXiv:2504.00030): both observe that a static window
leaves speed on the table. The contribution here is to drive depth from goinfer's
**own measured** `α̂` and **own measured** `c`, rather than a tuned heuristic — the
instrumentation in [00-core](./00-core.md) makes that data first-class.

## Why it suits goinfer

- It needs only the acceptance predictor and a per-backend cost number — both
  produced by the [00-core](./00-core.md) harness.
- It is drafter-agnostic: works with a single n-gram or head drafter immediately,
  and slots straight into the [03 router](./03-router-tree.md) as the depth policy
  for each tree branch.

## Risks / open questions

- **Predictor calibration matters more than accuracy here:** an over-confident `α̂`
  makes over-long drafts that waste verify width. Prefer a *calibrated* `α̂` (good
  Brier score) and test with reliability diagrams.
- **Cost asymmetry:** the marginal cost of one more draft node is not the same as one
  more verify node (tree width vs draft compute). Use the two measured costs
  separately, not a single `c`.
- **Stability:** avoid thrashing depth step-to-step; a little hysteresis / smoothing
  on the running product may help. Measure.

## Validation plan

- Correctness: lossless by construction; assert output ≡ baseline.
- Speed/acceptance: sweep against fixed-`K` baselines (`K = 1,2,4,8`) on all suites;
  show adaptive depth dominates the best single fixed `K` across suites (the whole
  point — no single `K` is best across workloads).
- Report committed/verify and the realized depth distribution per suite.
