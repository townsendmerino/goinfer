# Phase 2 premise: DEAD — grammar-masking makes acceptance WORSE, −27.5% on 1-of-6 pairs

> Measured 2026-08-25 on `linux-62gb`, goinfer `5b01066`, CPU (acceptance is numerics and
> transfers; wall-clock is not measured here), Qwen3-4B int8 + DFlash-f32, `maxNew=64`,
> 6-prompt constrained-JSON suite, paired per prompt.
> **Rule: `phase2-grammar-premise-PREREGISTERED.md`, committed at `dcc32ef`/`5b01066` before
> this number existed.** Harness: `decoder/grammar_premise_test.go`.

## Verdict: DIES, on all three conditions independently

| arm | mean tok/round | paired vs B | wins |
|---|---|---|---|
| **B** unconstrained | **3.53** | — | — |
| A1 target-masked *(diagnostic)* | 2.16 | −35.3% | 1/6 |
| **A2 both-masked *(decision arm)*** | **2.39** | **−27.5%** | **1/6** |

The rule kills Phase 2 if **Δ ≤ +5%**, **or** win rate **≤ ½**, **or** absolute **< 2.5**.
Measured: **Δ = −27.5%**, **wins = 1/6**, **absolute = 2.39**. All three fire. This does not
reach the AMBIGUOUS band — grammar-masking does not merely fail to help, it costs about a
quarter of the acceptance the unconstrained arm already had.

Per prompt:

| prompt | B | A1 | A2 |
|---|---|---|---|
| weather | 2.83 | 4.05 | **5.08** |
| person | 4.06 | 1.90 | 2.00 |
| book | 3.95 | 1.41 | 1.50 |
| config | 3.82 | 1.84 | 1.94 |
| order | 3.41 | 2.04 | 2.00 |
| measurement | 3.10 | 1.68 | 1.82 |

## The autopsy — and it closes the door rather than leaving it ajar

The pre-registration made A1 mandatory precisely so a DIES would arrive with a cause, and
pre-committed the two readings:

- *A1 healthy, A2 dead* → masking the **drafter** is the cost; a grammar-aware drafter or MTP
  under constraint would still be worth a thought.
- *A1 and A2 both dead* → the drafter cannot draft into this distribution at all, and no
  variant of the idea deserves one.

**Both are dead** (2.16 and 2.39 against 3.53), so it is the second reading. But the ordering
carries the mechanism, and it is worth stating exactly:

**A2 > A1.** Masking the drafter *helps* — by +10.6% over masking only the target. So "the
drafter proposes tokens the grammar forbids, which are auto-rejected" is real and is worth
about a tenth of the acceptance. It is simply not the dominant cost. Fixing it completely
(which A2 does, by construction) still leaves the arm 27.5% below unconstrained.

**What the dominant cost is:** the mask forces a *legal* tokenization, not the model's *own*
one. A drafter trained to predict what the target naturally emits mispredicts a target being
steered token-by-token into a different-but-equivalent encoding of the same JSON. That is the
tokenization-boundedness `docs/spec/01-grammar-fused.md` identified for the grammar-automaton
drafter — *"free grammar drafting must guess how the model tokenizes the forced bytes under the
mask"* — now reproduced for a **model** drafter, which is the case that doc explicitly declined
to rule on. The premise was a live question and it has now been answered on its own terms.

## The one prompt that went the other way, because it says what the mask is for

`weather` is the single win, and it is large (+79%). It is also the only prompt where the
**unconstrained** arm was weak (B = 2.83, against 3.10–4.06 everywhere else). Where the model
already emits well-formed JSON on its own, the mask can only push it off its natural
tokenization, and it costs 50–62%. Where the model was floundering, the mask substitutes for
competence and pays.

So the honest generalisation is the inverse of Phase 2's premise: **a grammar helps a drafter
exactly when the target is bad at the format, and hurts when it is good at it.** A prior that
widens the draft whenever a grammar is active would be widening precisely when the target is
most fluent — the worst possible time. 1-of-6 is not a knob to tune; it is the wrong sign.

## No plumbing price is owed

The pre-registration required a LIVES verdict to carry the cost of admitting a stateful mask
through `BlockSpec`'s batched verify (which refuses a `LogitProcessor` outright). **That bill is
not payable and not owed** — the premise died before the plumbing was priced, which is the whole
reason it was tested first. The `LogitProcessor` refusal stands as shipped.

## Consequence, also pre-registered: the width controller is removed

Phase 1's controller was kept default-off pending a phase that might reuse it. That phase is
gone, and the controller measured *worse than both endpoints it moves between* (chat 0.824×
against static8 0.887× and static4 0.993×). It comes out in the same cleanup, per the rule.

**Kept, deliberately:** the `OnRound` telemetry seam (the brief lists per-request accept-rate
and tok/verify as shipping regardless), the mixed-content suite, and this constrained-JSON
suite. Both suites are how the next drafter change gets checked against Finding 2 and against
this one, rather than re-assuming either.
