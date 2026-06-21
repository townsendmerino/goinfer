# 03 — Router + draft trees

> Status: **proposal**. Depends on [00-core](./00-core.md), and is most useful once
> [01](./01-grammar-fused.md), [02](./02-cache-ngram.md), and ideally [05](./05-eagle3-head.md)
> exist as sources. This is the controller that ties them together.

## Idea

No single drafter is best everywhere: the grammar wins on structure, the n-gram
index wins on copied text, a feature head wins on novel prose. Most engines pick
one. The controller instead, **per position**, asks each available source for a
cheap proposal and **merges them into one draft tree**, then verifies the whole tree
in a single target pass. Two wins stack:

1. **Source selection** — spend draft compute where it pays, take free tokens
   (grammar, n-gram) where available.
2. **Tree verification** — instead of betting on one linear continuation, verify
   several candidate branches at once (SpecInfer / Sequoia / EAGLE-2 style), branching
   where the drafters disagree and going deep where they agree. More committed tokens
   per verify FLOP than any single chain at equal width.

## The cost model

From [00-core](./00-core.md) §4 we have a calibrated acceptance predictor
`α̂_s(features)` per source `s`. Routing is then a per-position cost minimization:

```
choose source(s) minimizing   draftcost(s) + (1 − α̂_s(context)) · penalty
```

and, given a fixed verify **budget** `B` (how many tree nodes fit in one target pass
at equal latency on this backend), allocate `B` to maximize expected committed
length:

- put a node where its **marginal** `α̂` × reach justifies its share of `B`;
- branch (spend width) where sources disagree and each branch's `α̂` is moderate;
- extend (spend depth) where `α̂` is high (often grammar/n-gram runs);
- stop a branch when `α̂^depth` drops below the marginal-cost threshold ([04](./04-adaptive-depth.md)).

Tree shape is thus *derived from measured acceptance*, not hand-tuned. `B` is
backend-specific (CPU SIMD vs WebGPU have different "free" batch widths), so the
harness measures `B` and the router reads it.

## Why it suits goinfer

- The sources are already in-process and cheap to poll; the grammar and n-gram
  sources cost ~nothing to consult even on a miss.
- goinfer owns the verifier, so a tree/causal mask over draft nodes is a local change
  to the existing forward pass rather than a cross-process protocol.
- The router is the place the §4 analysis becomes a live policy: `α̂` is trained
  offline from `SpecTrace`, loaded as a small table/model, and consulted per step.

## Build sequence

1. Start with a **fixed priority** router (grammar → n-gram → model/head → plain),
   linear drafts only. Establishes plumbing and a baseline.
2. Add **tree** verification (multi-branch mask, longest-accepted-path selection in
   the verifier — already specified in [00-core](./00-core.md) §6).
3. Replace fixed priority with the **`α̂`-driven** allocation above.

Each step is independently benchable, so regressions are attributable.

## Increments

- **Inc 1 (done — priority router, linear):** `decoder.RouterDrafter{Sources}` returns
  the first source's non-empty proposal (grammar → n-gram), feeding the existing
  masked verify (`GenerateGrammarSpeculative` now takes a pluggable `Drafter` +
  threads the generation context so an n-gram source can search it). Lossless:
  `TestGrammarRouterSpec` — both grammar-only and the fused router are token-identical
  to plain constrained `Generate`; the drafter only affects *which* tokens are
  proposed, the masked verify decides.
  - **Result (the first clear fusion win in the program):** on a generic "fill a
    schema from prose" prompt the router = grammar-only (1.13 tok/round) — n-gram
    can't fire because the JSON scaffolding context doesn't echo the prose. But on
    the **agent-loop / repeated-tool-call** case (the prompt contains the JSON to
    reproduce), the router hits **4.25 tok/round (acc 0.82) vs 1.13 grammar-only** —
    grammar validates the structure while n-gram copies the repeated content, both
    under one masked verify. Fusion compounds exactly where the doc predicted: output
    that is *both* structured *and* repetitive.
- **Inc 2 (next):** tree verification — verify multiple candidate branches per pass
  (the data-backed lever to recover the sampled n-gram loss, where a single linear
  chain breaks on the first probabilistic reject). The hardest correctness surface;
  gate hard. **Inc 3:** α̂-driven allocation.

## Risks / open questions

- **Verifier complexity:** tree masks + longest-path selection + correct rollback to
  the accepted path (including the SSM/linear state-checkpoint case from
  [00-core](./00-core.md) §6) is the hardest correctness surface in the project.
  Gate hard against non-speculative decode.
- **Budget mis-estimation:** if `B` is set too high the verify pass stops being
  "free" and latency regresses; measure `B` per backend and leave headroom.
- **Predictor drift:** `α̂` trained on one workload mix may mis-rank sources on
  another; keep a cheap online correction (running per-source accept rate) as a
  fallback when `α̂` is uncertain.
- **Diminishing returns:** if [01]+[02]+[04] already capture most of the win, the
  tree may not pay for its complexity — decide with harness numbers, not faith.

## Validation plan

- Correctness: output ≡ non-speculative baseline across all suites, greedy + sampled;
  explicit tests for multi-branch rollback and for the SSM/linear state restore.
- Speed/acceptance: all four suites; report committed/verify and tok/s for each build
  step (priority → tree → `α̂`) so the marginal value of each layer is visible.
