# 10 — Gating optimistic forward

> **Status: DESIGN, awaiting sign-off. No code.** This page exists to settle the *shape* of the gate
> before anything is built, because the measurement below rules out the shape that would otherwise be
> the obvious choice. Dated design disclosure, same convention as the rest of this directory.
>
> Evidence: G26 in [`../QUEUE.md`](../QUEUE.md). Raw cells `docs/measurements/g26-tsweep*.json`.

## The problem, measured

`optFwdEligible` is unconditional on sampled decode: `useGPU && !fastGreedy && … && GOINFER_NO_OPTFWD == ""`.
It is therefore on for every `Temperature > 0` request. Two models, n=6, CUDA, depth 128, same
binary, `GOINFER_NO_OPTFWD=1` as the only difference between arms:

| T | phi3-mini (vocab 32064) | 1.5B (vocab 151936) |
|---|---|---|
| 0.2 | −1.1% **wins** | **−7.4% wins** |
| 0.4 | +2.8% loses | **−6.0% wins** |
| 0.6 | +6.3% loses | **−5.1% wins** |
| 0.8 | +6.8% loses | −0.9% no effect |
| 1.0 | +5.5% loses | −0.9% no effect |
| **crossover** | **≈ 0.26** | **≈ 0.95** |

(Negative = optFwd wins. Typical chat sampling is T = 0.7–1.0.)

**No temperature constant is correct for both**, and they differ only in vocab size:

- Safe for phi3-mini (T ≤ 0.2) → the 1.5B forfeits **6.0%** at T=0.4 and **5.1%** at T=0.6.
- Tuned for the 1.5B (T ≤ 0.95) → phi3-mini pays **2.8–6.8%** across T=0.4–1.0.

**A fitted static rule is what this data argues against, not for.** Two models broke a constant;
nothing suggests a four-model fit survives the fifth, or survives a prompt distribution not tested.
That is the whole reason this page exists rather than a table of thresholds.

## Why the split happens

optFwd overlaps the CPU sampler with a speculative next forward, so its upside is bounded by **how
much sampler there is to hide**, and its downside is a discarded forward on a miss. Both terms move:

| model | vocab | decode step | sampler | sampler share of the sampled token |
|---|---|---|---|---|
| phi3-mini | 32064 | 8.026 ms | 0.457 ms | **5.4%** |
| 1.5B | 151936 | 4.535 ms | 1.009 ms | **18.2%** |

phi3-mini can win at most ~5.4% and was measured losing 6.8% — the miss cost simply exceeds
anything the overlap can return there. The harm concentrates on **small-vocab** models, which is the
opposite of where intuition looks.

    value  ≈  p_hit · c_sampler  −  (1 − p_hit) · c_miss

`p_hit` is already tracked (`OptFwdStats{Guessed, Hit}`), and falls with temperature (98% at T=0.2,
55.6% at T=1.0 by the feature's own gates). `c_sampler` is measured above. **`c_miss` has never been
measured — it is inferred from end-to-end deltas only**, and that gap decides between the two designs
below.

## Two candidate shapes

### A — cost-model gate

Run a warmup window, read `p_hit`, estimate the two costs, evaluate the inequality, disable if
negative; re-probe occasionally.

**The objection is structural, not effort.** `c_miss` is not directly timeable in situ. The waste on
a miss is not a duration you can bracket — the speculative forward ran *concurrently*, and its cost
is the contention it imposed on the real one. Timing the discarded call measures the wrong thing.
So A's decisive term must itself be inferred from an A/B — at which point the A/B is the gate and
the cost model is a layer on top of it.

### B — empirical A/B, decided per (model, sampling config) and persisted per process

Alternate windows with the flag on and off, compare realized tok/s, keep the winner, re-probe rarely.

**Safe because the feature is output-neutral.** Its own gate is `TestOptFwd_bitIdenticalStream` —
token-identical streams — so toggling mid-generation cannot change what a user receives. This is the
property that makes runtime experimentation legitimate here and would not be available for most
flags.

**The decision must persist across requests, not per generation.** A window big enough to resolve a
5% effect is larger than many chat replies, so a per-generation A/B would rarely conclude before the
generation ended. But the answer is a property of (model, sampling config) and is stable across
requests, so a server learns it once and every later request benefits.

## Recommendation

**B**, for one reason: it needs no quantity that cannot be measured where the code runs, and it
self-corrects across models, vocabs, quantizations, cards and prompt mixes — none of which a fitted
table handles. A is a cost model wrapped around an experiment it still has to run.

## Kill gate, pre-registered

**B is dead if a 5% effect cannot be resolved in a practical number of tokens.** Falling back to
default-off is then the answer, accepting the loss of the large-vocab win.

Feasibility estimate from existing data, and its limits: per-run spread (512 tokens: 64 × 8) was
~0.5–3% of the mean across the ladders. Scaling to a 128-token window suggests ~1–6%, so ~4–8 windows
(≈512–1024 tokens) to separate a 5% effect. That is affordable **only** because the decision persists
across requests. **This is an extrapolation from RUN-level variance to WINDOW-level variance, which
has not been measured** — and per this repo's own experience today, a number extrapolated out of its
regime is exactly what needs checking rather than assuming.

## What must be measured before building

1. **Window-level tok/s variance** at 64 / 128 / 256 tokens, both arms, both models. This decides
   the window size, the number of windows, and whether B clears its kill gate at all. Cheap — it is
   a re-slice of the sweep already in the harness.
2. Nothing else. In particular **`c_miss` is not needed for B**, which is a large part of B's appeal.

## Validation — the two ladders are the test set, and they disagree

A gate is correct only if it reproduces both, without per-model constants:

| model | required behaviour |
|---|---|
| phi3-mini | optFwd **off** above T ≈ 0.26 — including at T=0.6, where it must switch off |
| 1.5B | optFwd **on** at T ≤ 0.6 — including at T=0.6, where it must stay on |

**T=0.6 is the discriminating cell**: the two models require opposite decisions at the same
temperature, so any rule keyed on temperature alone fails it by construction.

## Risks, and the do-nothing arm

- **Include the do-nothing arm.** The gate must be measured against *no gate at all*, both arms
  present, or "the gate helps" is unfalsifiable. This repo has already found a speculation suite
  where nothing beat running no drafter.
- **The gate must not cost more than it saves.** Probing spends tokens in the worse arm by
  construction; with re-probes that is a standing tax and needs bounding.
- **Flapping.** Needs hysteresis and a minimum sample; a gate that oscillates is worse than either
  fixed choice.
- **Convergence is a claim to test, not assume** — including on short generations and on servers
  that see mixed sampling configs.

## Out of scope

Changing what optFwd *does*, the lossless invariant (non-negotiable and untouched), any
serving/router surface, and MTP/EAGLE drafting. This page is only about when the existing feature
should run.
