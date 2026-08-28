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

## Kill gate result (2026-08-27): **PASSES**

Measured, not extrapolated. 12 cells, 32 windows each, T=0.6, CUDA, goinfer-only, **every cell at
loadavg <= 1.00** (see the contamination note below). Raw `docs/measurements/g26-winvar-*.json`.
Windows/arm is for t = 2 on the model's own measured effect at T=0.6.

| model | ngen | CV_on | CV_off | effect | windows/arm | tokens to decide |
|---|---|---|---|---|---|---|
| phi3-mini | **64** | 3.29% | 0.32% | 6.3% | **2** | **256** |
| phi3-mini | 128 | 2.59% | 0.23% | 6.3% | 1 | 256 |
| phi3-mini | 256 | 1.75% | 0.49% | 6.3% | 1 | 512 |
| 1.5B | **64** | 7.86% | 2.99% | 5.1% | **11** | **1408** |
| 1.5B | 128 | 6.40% | 2.32% | 5.1% | 8 | 2048 |
| 1.5B | 256 | 6.89% | 2.83% | 5.1% | 9 | 4608 |

**Worst case 1408 tokens.** Affordable for a decision persisted per (model, sampling config), and
infeasible per generation — which is the design's reason for persisting it, now quantified rather
than asserted.

**USE 64-TOKEN WINDOWS. Larger ones are strictly worse, and the reason matters.** CV does not fall
as 1/sqrt(ngen); it FLOORS OUT, and on the 1.5B it gets worse from 128 to 256 (6.40% -> 6.89%).
There is a persistent per-window component that averaging cannot remove, so doubling the window buys
less than the sqrt(2) needed to break even and total tokens rise. **The design assumed longer windows
would help and the measurement inverted it.**

**The models that most need the gate are the cheapest to gate.** phi3-mini — the one losing 6.3% —
decides in 256 tokens because its OFF arm is nearly noiseless (CV 0.32%). The costly case is the
1.5B, where the feature WINS and being slow to decide costs least. The gate's error budget therefore
falls where the errors are cheapest.

**Where the earlier estimate landed.** This page predicted ~4-8 windows (512-1024 tokens) from
run-level variance and flagged it as an extrapolation needing checking. It was right in magnitude and
wrong in spread: 2 windows/256 tokens on phi3-mini, 11/1408 on the 1.5B. A single figure would have
been wrong for both.

**A contamination the first attempt suffered, kept because it is the reason to trust these numbers.**
The first run of this sweep passed preflight at loadavg 0.75, then ran five cells at 1.00-2.43
because another job started on the box ten minutes in. All five were discarded. Worse here than for
most measurements: contention adds variance, and variance is the quantity under test. I predicted the
contaminated CVs were inflated and the true values lower — **measured clean, they are slightly
HIGHER** (3.29 vs 2.71, 7.86 vs 7.45), so this variance is the feature's own hit/miss lottery and not
the neighbouring job. `bench_peer.py` now re-checks idle before every cell and REFUSES on timeout
rather than measuring anyway.

## What must be measured before building

1. ~~**Window-level tok/s variance**~~ — **DONE, see above. Gate passes; window size is 64.**
2. Nothing else. In particular **`c_miss` is not needed for B**, which is a large part of B's appeal.

**Nothing now blocks building B.** The open work is design detail rather than measurement:
hysteresis, the re-probe interval and its standing tax, what happens when a server sees mixed
sampling configs, and the do-nothing arm in the risks section below.

## Validation — the two ladders are the test set, and they disagree

A gate is correct only if it reproduces both, without per-model constants:

| model | required behaviour |
|---|---|
| phi3-mini | optFwd **off** above T ≈ 0.26 — including at T=0.6, where it must switch off |
| 1.5B | optFwd **on** at T ≤ 0.6 — including at T=0.6, where it must stay on |

**T=0.6 is the discriminating cell**: the two models require opposite decisions at the same
temperature, so any rule keyed on temperature alone fails it by construction.

## CORRECTION (2026-08-28): AN ADAPTIVE GATE ALREADY EXISTS. This page designed one anyway.

**`optFwdGate` in `decoder/spec_optfwd.go` is a trailing hit-rate EMA with hysteresis — EnableAt
0.90, DisableAt 0.75, λ 0.95 — and it shipped with the feature.** Design B, as specified below, is
substantially that. I wrote this page after reading `OptFwdStats` in the same file and never scrolled
to the gate underneath it. **The design work was largely redundant and the page is left standing
only because the arithmetic in it is not.**

**What the existing gate gets wrong is its CONSTANTS, and the mistake is the one this whole item is
about.** Its own comment records the calibration: *"pinned to the WORST measured break-even across
depth (90.9% at a shallow qwen2.5-coder-0.5b context)"* — a **152k-vocab** model. Break-even hit rate
is not model-independent: it rises as the sampler share falls, so on phi3-mini (5.4% share against
the 1.5B's 18.2%) the true break-even sits ABOVE 0.90 and the dead band lies entirely below it. The
gate cannot fire in the regime where the feature loses. **A mechanism calibrated on one model and
applied to all — the same error, one layer down.**

**A second defect, filed separately: the hysteresis is a ONE-WAY LATCH in the wiring.** `Observe` is
called only from `optFwdStep`, which the caller invokes only when `Should()` is true. So once the
gate turns off, no further outcomes are observed, α freezes, and the `α ≥ EnableAt` re-enable branch
is unreachable for the rest of that Generate. `TestOptFwdGate_hysteresis` passes because it calls
`Observe` in an unconditional loop — valid for the component, defeated by the composition.

**None of this changes what shipped.** The T ≤ 0.2 cap means the overlap does not run above 0.2 at
all, so both the mis-calibrated thresholds and the latch stop mattering in the losing regime. It also
means the cap is doing a job the existing gate was supposed to do, which is worth knowing if anyone
later fixes the constants: **the right long-term fix is probably a break-even threshold that scales
with the sampler share — computable at LOAD time, no probing** — and that is exactly what the third
model is being run to test.

## The do-nothing arm is a FIXED T ≤ 0.2 THRESHOLD, and that is what B must beat

**Not the always-on default.** A one-line constant is a real, measured, well-understood alternative:
no probing, no hysteresis, no persistence, no state, nothing to converge. **If this page compares B
only against today's unconditional default, B wins by being the thing that got built.** So the
comparison is fixed here, against the ladders, before any code exists.

Value against the always-on baseline, per measured cell (+ve = the policy gains):

| model | T | fixed T ≤ 0.2 | oracle | B's headroom over fixed |
|---|---|---|---|---|
| phi3-mini | 0.4 | **+2.8%** | +2.8% | 0.0% |
| phi3-mini | 0.6 | **+6.3%** | +6.3% | 0.0% |
| phi3-mini | 0.8 | **+6.8%** | +6.8% | 0.0% |
| phi3-mini | 1.0 | **+5.5%** | +5.5% | 0.0% |
| 1.5B | 0.4 | **−6.0%** | 0.0% | **+6.0%** |
| 1.5B | 0.6 | **−5.1%** | 0.0% | **+5.1%** |
| 1.5B | 0.8 | −0.9% | 0.0% | +0.9% |
| 1.5B | 1.0 | −0.9% | 0.0% | +0.9% |

**The fixed threshold captures ALL of phi3-mini's available value and none of the 1.5B's — and in
the 1.5B's mid-band it is actively WORSE than shipping nothing at all**, turning a 6.0% win into a
6.0% loss. **B's entire advantage over the do-nothing arm is that band. There is no other cell where
B is worth anything.**

**Do not read a single average here.** Meaning over these ten cells (fixed +0.85%, oracle +2.14%,
headroom +1.29%) weights every cell equally, which is a statement about the sweep and not about
traffic. The real number depends on how much serving actually happens on large-vocab models at
T = 0.4–0.8, and this repo has no traffic data. **If that band is rare, the fixed threshold is the
better engineering choice and B should not be built.**

## Probe tax — the number, before the implementation

| model | windows/arm | probe cost | lost in the worse arm | over 10k tok | over 100k tok |
|---|---|---|---|---|---|
| phi3-mini | 2 | 256 tok | ~8.1 token-equivalents | 0.081% | 0.008% |
| 1.5B | 11 | 1408 tok | ~35.9 token-equivalents | 0.359% | 0.036% |

**The tax is only standing if the probe is on a TIMER, and it must not be.** The decision is keyed on
(model, sampling config) — both known inputs, not things to poll for — so the design probes **once
per unseen key** and never again. That makes the cost one-time and amortising: 0.036% worst case
over 100k tokens on a key, falling with every token after.

**If a periodic re-probe is added anyway** — as a safety net against drift — it must be no more
frequent than **every ~36k tokens** on the 1.5B to hold a standing tax ≤ 0.1%. Recorded so that
number exists before someone picks a convenient interval in code.

**The residual risk this leaves, stated plainly: prompt distribution is NOT in the key.** Hit rate
depends on what is being generated — code versus prose — so a key probed on one traffic type can
carry a wrong answer into another. Keying more finely is not obviously possible; accepting the error
is the likely answer, but it is an accepted error rather than an absent one.

## Pre-registered decision rule

**B ships only if it beats the fixed T ≤ 0.2 threshold on the validation set by more than its probe
tax.** Concretely, on the two ladders:

- **B must reach ≥ +4% over fixed at 1.5B/T=0.4 and T=0.6** (against the +6.0/+5.1% oracle headroom,
  allowing for probe error), **and lose nothing beyond the tax anywhere else.**
- **If B cannot, ship the fixed threshold** — it is one line, has no failure modes, and captures
  phi3-mini's entire win.
- **If B's probe misidentifies either 1.5B mid-band cell**, that is a fail, not a tuning exercise:
  those two cells are the only reason B exists.

Where the effect is near zero (1.5B at T=0.8/1.0, ±0.9%) the probe will be inconclusive, and that is
acceptable **because the cost of guessing wrong there is also ±0.9%** — the same property the kill
gate found, that errors are cheapest exactly where deciding is hardest.

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

## Third model — PRE-REGISTERED before the run (2026-08-28)

**The question changed with the G27 discovery, and the run is built for the new one.** Not "does
crossover track vocab" — vocab was only ever a proxy. **Does BREAK-EVEN HIT RATE track SAMPLER
SHARE?** That is mechanistic rather than correlational: the overlap can only repay the sampler it
hides, and loses a contended forward on a miss, so share is what sets break-even. If it holds, the
existing gate's thresholds become computable **at load time with no probing** — which fixes G27's
real defect and retires design B rather than building it.

**The third model is `qwen2.5-7b`, and NOT another large-vocab model from a different family.** The
brief asked for different-family/similar-vocab, but the reframing inverts that: **the 7B holds vocab
CONSTANT at 151936 — identical to the 1.5B — while its far larger decode step makes the sampler
share much smaller.** That is the controlled comparison the hypothesis needs, because it separates
the two variables that are confounded in the existing points:

| model | vocab | sampler share | what it tests |
|---|---|---|---|
| phi3-mini | 32064 | 5.4% | small vocab, small share |
| 1.5B | 151936 | 18.2% | large vocab, large share |
| **7B** | **151936** | **expected SMALL** | **large vocab, small share — the discriminator** |

**If crossover tracks vocab, the 7B behaves like the 1.5B. If it tracks share, the 7B behaves like
phi3-mini despite sharing the 1.5B's vocabulary.** One run separates them; a different-family
large-vocab model would have added a point without resolving the confound. (Gemma-3 was the obvious
different-family candidate and is ineligible anyway: sliding-window attention, which
`specRollbackSafe` refuses.)

**What is measured, per model:** the ON/OFF throughput ladder (crossover in T), greedy-vs-sampled
step cost (sampler share), and — new — the realized hit rate across the same ladder
(`TestOptFwd_hitRateLadder`), giving **α at the crossover, which IS that model's break-even hit
rate**, measured rather than inferred through a temperature proxy.

**WHAT "HOLDS" MEANS, fixed before seeing the numbers.** Three points lie on some curve; a criterion
after the fact is how 0.90 got set from one model. From the value model, break-even satisfies

    c_miss  =  c_sampler · α_be / (1 − α_be)

so the mechanism's real claim is that **`c_miss` is a property of the DECODE STEP, not of the
sampler.** The test is therefore not a curve fit:

- **HOLDS if `c_miss / c_decode` is within a factor of 2 across all three models.** Then break-even
  is computable at load time from the sampler share alone, and the gate's constants can be derived
  rather than fitted.
- **REJECTED if it varies by more than 2x**, or if the 7B's crossover lands with the 1.5B's despite
  a small share. Then break-even is not predictable from static properties, B's premise gets real
  support, and the fixed T ≤ 0.2 cap stays as the shipped answer.
- **A factor of 2 is deliberately loose**, because α_be is read off a crossover located to about
  ±0.1 in T and the window-CV floor limits how much better that gets. A tighter criterion would be
  claiming resolution the measurement does not have.

**THE 7B MOVES MORE THAN SHARE, AND THE READING MUST NOT PRETEND OTHERWISE.** Against the 1.5B it
also changes memory-bandwidth pressure, cache behaviour, and how much of the decode step is
weight-streaming — all of which travel with parameter count. So there is a THIRD outcome, and it is
not a failure of the run:

- Lands **with phi3-mini** → clean support for share, since vocab is held fixed and only share moved.
- Lands **with the 1.5B** → share is not the driver; vocab or family is. Reject, and the cap stays.
- Lands **BETWEEN them** → **AMBIGUOUS, and report it as ambiguous.** The confound is size, not a
  refuted hypothesis, and forcing a middling result into the accept/reject binary is exactly the
  motivated reading this page exists to prevent. Resolving it would need a fourth point that moves
  share without moving parameter count — a different quantization of the SAME model is the obvious
  candidate, since it changes the decode step while holding vocab, family and architecture fixed.

**THE 2x BAND TRAVELS WITH THE NUMBER, WHEREVER THE NUMBER GOES.** Whatever `c_miss / c_decode` comes
out at, it is located to about ±0.1 in T against a CV floor, and it is loose for that reason. It must
never be quoted bare — **this ratio is what would go on to set the gate's constants**, so a figure
that arrives without its band will be read as tight by whoever sets them. Label it at the point of
writing, not at the point of use; this repo has already recorded that numbers travel out of their
regime when the caveat lives somewhere else.

**Resolution decided in advance: ±0.1 in T is enough to tell "tracks share" from "scatters", and the
run stops there.** Chasing ±0.02 would cost a night against a CV floor for a distinction that changes
no decision.

**The shipped T ≤ 0.2 cap does not wait on this.** It beats always-on by a wide margin at any value
in range, and refining the number later is a one-line change.

## Third-model RESULT (2026-08-28): vocab is rejected, share is supported, and the mechanism is two numbers

**THE HEADLINE IS THE SAMPLER-COST DECOMPOSITION, because it explains why vocab looked like the
driver for two models running.** Same vocabulary, different model:

| model | vocab | sampler COST | decode step | sampler SHARE of the token |
|---|---|---|---|---|
| 1.5B | 151936 | **1.009 ms** | 4.535 ms | **18.2%** |
| 7B | 151936 | **1.102 ms** | 13.685 ms | **7.5%** |

**Vocab sets the sampler's ABSOLUTE cost (1.009 vs 1.102 ms — a 9% difference across a 4.6x model);
the DECODE STEP sets what fraction of the token that cost is.** Share is the product of the two, and
it is share that the overlap can repay. Across the first two models vocab and share moved together,
so vocab was an adequate proxy and looked causal. It is not.

**7B ladder** (n=6, CUDA, depth 128, `GOINFER_OPTFWD_MAX_TEMP=2.0` so the shipped cap does not
silently turn the ON arm into a second OFF arm). Raw `g27-7b-optfwd-{on,off}.json`:

| T | ON | OFF | Δ |
|---|---|---|---|
| 0.2 | 67.12 ± 0.39 | 67.82 ± 0.09 | **+1.0% loses** |
| 0.4 | 64.92 ± 0.87 | 67.83 ± 0.07 | +4.5% loses |
| 0.6 | 63.45 ± 0.98 | 67.69 ± 0.20 | +6.7% loses |
| 0.8 | 62.22 ± 0.79 | 67.63 ± 0.11 | +8.7% loses |
| 1.0 | 61.69 ± 1.10 | 67.62 ± 0.13 | +9.6% loses |

| model | vocab | share | crossover |
|---|---|---|---|
| phi3-mini | 32064 | 5.4% | ~0.26 |
| 1.5B | **151936** | **18.2%** | **~0.95** |
| 7B | **151936** | **7.5%** | **< 0.2** |

**The confound is controlled in BOTH directions, which is what makes three points carry this.**
7B vs 1.5B holds vocab and family fixed while share and size move — behaviour FLIPS. 7B vs phi3-mini
differs in vocab, family AND size while share is similar — behaviour AGREES. Low share is the only
common factor. **Vocab is rejected as the driver.**

### What is NOT established, stated rather than smoothed over

**The pre-registered quantitative criterion could not be evaluated.** It required `c_miss/c_decode`
within 2x, and `c_miss` needs α_be, which the α ladder failed to deliver (below). So the mechanism is
supported QUALITATIVELY and unconfirmed QUANTITATIVELY — and the missing number is exactly the one
the gate's constants would be derived from. **The constants therefore cannot be derived yet.** No
substitute criterion is offered after the fact; that is how 0.90 was set from one model.

**The ordering is NON-MONOTONIC between the two low-share models.** Share 5.4% → crossover 0.26, but
7.5% → below 0.2. Higher share should raise the crossover; these invert. Both sit inside the ±0.1
resolution pre-registered, so it is noise-consistent — **but it is equally what would be seen if
share does not determine the crossover alone and something else travels with model size. Both
readings fit these three points.** The safe consequence is the one taken: **share predicts WHICH SIDE
of the cap a model lands on, not a fine ordering, and nothing is fitted to three points.**

### Three models, three different right answers for the cap

phi3-mini WINS 1.1% at T=0.2; the 7B LOSES 1.0% there; the 1.5B wants the cap at ~0.95. **The shipped
T ≤ 0.2 constant is therefore slightly wrong for the 7B on the third model examined** — small, and
inside that cell's noise, but the point is not the mistuning. **Three models produced three different
correct constants. That is evidence a per-model CONSTANT is the wrong SHAPE, not that this one needs
tuning.**

### Where this points: a BINARY at load time, not a threshold

If share cannot give a fine ordering but does sort models into "the overlap can pay here" and "it
cannot", then the load-time calculation is **a binary, not a continuous rule**: compute the sampler
share when the model loads, and either **enable the existing `optFwdGate` EMA** (letting it adapt
within the profitable regime) or **disable optFwd outright**. That is a far smaller target than
deriving a continuous threshold, and it is what these three points actually support — two models
plainly in "cannot", one plainly in "can".

**It is downstream of fixing the α measurement**, because choosing the share boundary needs
break-even α, which needs the junk-prompt and latch problems fixed first (G27).

### The α ladder failed, and G27 is why

`TestOptFwd_hitRateLadder` ran on all three models and produced nothing usable, for two independent
reasons:

- **The prompt is degenerate** — `[]int{1, 7, 42}`. phi3-mini returned α = 1.0000 (200/200) at T=0.8
  where its throughput ladder measured optFwd LOSING 6.8%; the 1.5B returned α = 0.61 at T=0.6 where
  its ladder measured it WINNING 5.1%. **Neither α reconciles with its own throughput**, so the
  prompt is unrepresentative in both directions rather than biased one way. α measured here cannot be
  paired with crossovers measured on real prompts at depth 128.
- **The latch truncates exactly the measurements needed.** Guess counts collapse from 200 to 6 / 10 /
  14 / 18 as soon as α falls — the gate turns off, stops observing, and low hit rates become
  unmeasurable. Observed on **all three models**.

**G27 has now cost twice: once as an unreachable re-enable branch, once as a blocked measurement on
the critical path to the binary gate. It is no longer latent.**

## THE PROMPT WAS THE CONFOUND (2026-08-28) — and removing it retires most of this page

**Every measurement above ran on `scripts/prompts.json`, in which EVERY prompt has FOUR UNIQUE
WORDS**: `"Continue this text. the the the ... the"`. That file is calibrated for token DEPTH, which
is exactly right for peer throughput rows — decode cost per token is content-independent, and filler
gives exact token counts. It is exactly wrong for optFwd, whose entire value is *how predictable the
generated text is*.

Re-ran both ladders against a 127-token prose paragraph (raw `g28-realprompt-*.json`):

| T | phi3-mini filler → real | 1.5B filler → real |
|---|---|---|
| 0.2 | −1.1% → **+0.2%** | −7.4% → **−4.8%** |
| 0.4 | +2.8% → +2.8% | −6.0% → **+0.9%** |
| 0.6 | +6.3% → +7.6% | −5.1% → **+4.3%** |
| 0.8 | +6.8% → +8.5% | −0.9% → +4.2% |
| 1.0 | +5.5% → **+9.2%** | −0.9% → +3.8% |

(+ve = optFwd loses. OFF baselines flat in all four ladders: 117.98 / 118.44 / 181.84 / 184.13.)

**The 1.5B's crossover moves 0.95 → 0.37. phi3-mini's stays below 0.2.** The swing runs
systematically against optFwd, and it is **largest exactly where design B's entire case lived**:
+9.4 points at the 1.5B's T=0.6, a 5.1% win that is really a 4.3% loss. **B existed to capture two
cells. Both were prompt artifacts.**

### What this retires

- **The binary load-time gate, the sampler-share calculation, and design B are all unnecessary** —
  not refuted, but aimed at a problem that does not exist on realistic input.
- **"Three models, three different right answers for the cap" was mostly the prompt.** On real prose
  every crossover sits at or below ~0.37 and **the shipped `T ≤ 0.2` constant is approximately
  optimal for all three**: neutral for phi3-mini at 0.2, capturing the 1.5B's remaining 4.8% win
  there, and avoiding every loss above it.
- **The share hypothesis survives but stops mattering.** Ordering still holds (18.2% → 0.37,
  5.4% → <0.2), but the spread collapses from [0.26, 0.95] to [<0.2, 0.37] — narrow enough that one
  constant covers it, which is why no per-model machinery is needed.
- **G27's urgency drops back to latent.** It was escalated because α was on the critical path to the
  binary gate; there is no binary gate. The latch and the model-fitted thresholds are still real
  defects and still unfixed, but nothing waits on them.

### What it does NOT retire, and is worse than it looks

**`scripts/prompts.json` silently invalidates ANY content-dependent measurement**, and speculation is
entirely content-dependent. In scope, unaudited:

- **optFwd's own `EnableAt = 0.90`**, whose comment sources it to a "90.9% worst-case break-even
  measured on qwen2.5-coder-0.5b" — plausibly on this prompt set, since it is the only one in the
  repo.
- **02's n-gram drafter "wins on copy-heavy traffic"** — a claim about content, and a repetitive
  filler prompt is maximally copy-heavy.
- **05/06's EAGLE acceptance figures**, and 09's MTP Gate 1, though 09 used its own suite prompts
  rather than this file.

**Filed as its own item rather than chased here.** The rule going in is simple: a prompt file
calibrated for depth may be used for throughput, never for acceptance.