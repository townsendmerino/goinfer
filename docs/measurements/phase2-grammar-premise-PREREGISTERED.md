# Phase 2 premise test — PRE-REGISTERED decision rule (written before the first run)

> Written 2026-08-25 against goinfer `3ce3532`, **before any measurement exists**. The point is
> that the result reads against a line drawn in advance instead of being interpreted afterwards.
> Nothing below may be edited once a number exists; the outcome goes in a sibling results doc
> that cites this one. Decider: Francis.

## The question, and why the existing measurements do not answer it

Phase 2 of `docs/prompts/adaptive-speculation.md` rests on one claim: **when a `constrain`
grammar is active, output is structured by construction — the high-acceptance regime, known at
request time.** The fork it came from *infers* "structured now" from trailing acceptance, with
lag; goinfer would *know*.

That claim is now wounded. `adaptive-width-shipgates-2026-08-25.md` Finding 2 measured
acceptance **falling** across a prose→structured boundary, and no width beating no-drafter on
that suite. But it measured **unconstrained** structured output, and grammar-constrained
decoding is mechanically different: the target's distribution is masked, so it could **rescue**
acceptance (both models funnelled into the same few legal tokens) or **worsen** it (the drafter
proposing tokens the grammar forbids outright, auto-rejected).

**The prior grammar work does not settle it either.** `docs/spec/01-grammar-fused.md` reports
~1.05–1.45 tok/round and α̂_grammar ≈ 0.20 — but that is the **grammar automaton used AS the
drafter**, reading forced runs off its own states, and the doc says so explicitly: *"a verdict on
this drafter … not on grammar speculation in principle."* Phase 2 proposes something else
entirely: a **model** drafter (DFlash) whose target is grammar-masked. Different mechanism,
unanswered question.

## A structural fact that constrains the answer

**`BlockSpec` refuses a `LogitProcessor` outright** (`decoder/blockspec.go`: greedy only), so
grammar-constrained decoding does not run through the block-drafting path today. Phase 2 is
therefore a plumbing change *on top of* a premise question. That is a reason to test the premise
before building anything, not after — and it means this test is a purpose-built measurement, not
a production run.

The measurement is legitimate on **CPU**, per `decoder/dflash_accept_test.go`'s own header:
acceptance is a property of the drafter's distribution against the target's — numerics — and does
not depend on the backend. Wall-clock does not transfer and is not measured here.

## Design

Same DFlash drafter, same target, same prompts, same verify width, **paired per prompt**.

| arm | target | drafter | decision-bearing? |
|---|---|---|---|
| **B** unconstrained | unmasked | unmasked | baseline |
| **A2** constrained | grammar-masked | grammar-masked (clone rolled over the drafted prefix) | **YES** |
| A1 diagnostic | grammar-masked | unmasked | no — diagnostic only |

**A2 is the decision arm** because it is what Phase 2 would actually ship. A1 exists only to
separate "the mask helps" from "the mask hurts because the drafter proposes illegal tokens";
it can explain a result but cannot decide one.

**A1 IS RECORDED UNCONDITIONALLY, including when A2 dies — it is the autopsy.** The runs are
paired anyway, so it is free, and the two failure modes it distinguishes are different verdicts
about the future, not the same one twice:

- **A1 healthy, A2 dead** → masking the DRAFTER is what costs acceptance. A grammar-aware
  drafter, or Phase 3's MTP head under constraint, is still worth a thought.
- **A1 and A2 both dead** → the drafter cannot draft into this distribution at all, masked or
  not, and no variant of the same idea deserves one.

A DIES with a cause is worth more than a DIES alone.

**Metric: committed tokens per round (tok/verify)** — the same unit the block-drafting break-even
is expressed in, so the number is directly comparable to the thresholds that already exist:
guard threshold **2.5**, measured true break-even **~3.5** for this pairing, unconstrained
structured content measured at **2.17–2.83**, code (the paying case) at ~5–6.

**Paired, from day one.** `default-verify-width-2026-08-25.md` showed pooled and paired views
disagreeing about whether an effect existed at all; the arms here are run on identical prompts,
so they difference cleanly and a win count is reported alongside the mean.

## The rule

Let **Δ** be the paired mean improvement of A2 over B (percent), **W** the number of prompt pairs
favouring A2 of N, and **T** A2's absolute mean tok/round.

- **PHASE 2 LIVES** — all three must hold:
  **Δ ≥ +15%** and **W ≥ ¾ of N** and **T ≥ 3.5** (true break-even; below it a widened draft
  cannot pay no matter how confident the prior is).
- **PHASE 2 DIES** — any one suffices:
  **Δ ≤ +5%**, or **W ≤ ½ of N** (a coin flip — the lesson from math replicating at 6/12), or
  **T < 2.5** (the guard would disable the drafter regardless of what the prior said).
- **AMBIGUOUS** — anything else. Phase 2 is **parked, not built**, pending a second pairing.
  Ambiguity is an outcome here, not a tie to be broken by judgement after the fact.

## Consequences, also pre-registered

- **If Phase 2 DIES:** the width controller is removed in the same cleanup. A flag-gated
  mechanism that measured worse than both endpoints it moves between should not outlive the last
  phase that might have reused it. Until then `Adaptive` stays default-off as shipped.
- **If Phase 2 LIVES:** the first build step is the plumbing, not the prior — `BlockSpec` has to
  accept a masked verify at all before a prior can steer anything. **A LIVES verdict therefore
  arrives with a bill, and the results doc must carry a rough plumbing price** (what has to
  change in `BlockSpec` to admit a stateful mask through the batched verify, and what that costs
  the greedy-only guarantee) so the build decision is taken against a cost rather than against a
  green light. The `LogitProcessor` refusal cuts both ways: it is why the premise is tested
  first, and it is why "it lives" is not by itself an instruction to build.
- **Either way**, the constrained-JSON suite is committed, like the mixed suite before it.
