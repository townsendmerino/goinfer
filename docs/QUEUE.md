# QUEUE — the index over four queues

The work list is split by **success criterion**, not by component. An entry lives in exactly one
queue, keeps its original ID, and keeps the section it was filed under.

| queue | holds | the question it answers |
|---|---|---|
| [**performance**](queue-performance.md) | throughput, latency, kernels, residency, memory | *how fast, how much memory* |
| [**correctness**](queue-correctness.md) | parity, numerics, goldens, quantization, families | *does it compute the right thing* |
| [**engineering**](queue-engineering.md) | gates, lints, censuses, tooling, process rules | *would we find out* |
| [**release**](queue-release.md) | release gates, tagging, v1.0 criteria, capability claims | *can we tag* |

**Task docs are not queues.** `docs/task-*.md` are design records — why a thing is built as it is —
and they are cited from **88 code comments**. A queue entry cannot carry that, so they stay where
they are and the queues hold only the open work. Finished ones are archived to `docs/completed/`.

**This file keeps** the cross-cutting material that is not any one queue's: the sweeps, the
sequencing notes, the release draft, and the generated citation indexes below.

# Work queue — the shared, claimable list

> **Why this file exists.** The queue used to live in conversation, where only the top of it gets
> restated each turn and everything below silently sinks. Three items aged out that way — the Metal
> consumer window, the out-of-tree consumer audit, the drain fix's CUDA verification — none through
> carelessness. And two boxes pulling from the same unstated queue independently built two
> mechanisms for running the heavy tier, because neither could see the other's progress.
>
> That makes the conversational queue an instance of the class this fortnight has been cataloguing:
> an artifact that exists and is not composed into any decision. A check that cannot fail.
>
> **This file is the queue.** If it is not written here, it is not queued.

## How to use it

- **Claim before starting.** Move the item to `In flight` and put your box and the date on it.
  A claim is what stops the other box duplicating it.
- **Release on finish** — move to `Done` with the commit, or back to `Queued` with what you learned.
- **Never delete an item to tidy up.** Strike it with a reason, so "we decided not to" is
  distinguishable from "it sank".
- **Add the whole item, not a title.** Enough that whoever picks it up does not have to reconstruct
  the context from a transcript they may not have.

Boxes: `linux` (nvidia-rtx2070s, CUDA) · `mac` (Apple Silicon, Metal).

## In flight

## Queued

Ordered roughly by priority within each group. Each item carries enough context to be picked up
cold. Where something is believed done but unconfirmed, it says so — **verify before striking**.

### A. Open investigation

**G26 · phi3-mini lost 5.8% at `temperature 1.0`, and only phi3-mini** — **CLOSED 2026-08-27: the
cause is optimistic forward (`6a4e0ae`), a net loss above T ≈ 0.26; about half the headline was the
anchor's own spread. Full result at the end of this entry — the investigation below is left intact,
including two findings it had to retract.** Measured
2026-08-27 in the §B5 re-anchor (`docs/benchmarks.md` §B5.1): 116.6 → **109.8** tok/s on CUDA at
depth 128, against an Ollama side that read 125.6 both times. Well outside the old ±0.5 spread.

**What makes it worth a look rather than a shrug:** the same configuration *gained* on the other
two models in the same sweep — qwen2.5-coder-0.5b 219.2 → 237.7 (+8.4%) and gemma3-1b 131.7 → 143.7
(+9.1%). A uniform sampler change cannot produce that split, so it is model-specific.

**Do not guess the cause from the numbers.** Two anchors differ by more than one thing: goinfer
moved several releases, and the driver/distro moved on 2026-08-25. The greedy phi3-mini CUDA cell
is *unchanged* across the same pair (124.9 vs Ollama 125.9, parity), which points at the sampling
path rather than the forward — but that is a lead, not a finding.

Raw cells: `docs/measurements/b5-reanchor-61b1e03.json`. Reproduce with
`BENCH_MODELS=phi3-mini BENCH_CONFIGS=temp1.0_notrunc python3 scripts/bench_peer.py <out>`.

**2026-08-27, two findings — and the second demotes this from "regression" to "not established".**

**1. P10 is NOT the cause, and the obvious argument for it was backwards.** `4da116d` (P10) is the
only sampler change between the anchors, and it removes a per-token full-vocab allocation whose size
scales with vocab — so the tempting story is that phi3-mini's 32k vocab gains least. Its benchmark
population was only 152k and 262k, so the small vocab was genuinely never measured, which made this
look like the int8-bar-on-int4 shape: a change validated on one population, shipped for another.

Measured it instead of arguing it. `BenchmarkFilterFresh*` (added here) is the pre-P10 shape — a
fresh buffer per call — paired against the reused-scratch path, `-count 5`:

| vocab | pre-P10 | post-P10 | P10 effect |
|---|---|---|---|
| 32k | 220,801 ns | 200,779 ns | **+9.1% faster** |
| 152k | 989,363 ns | 908,777 ns | +8.1% faster |
| 262k | 2,800,565 ns | 2,687,574 ns | +4.0% faster |

P10 helps **most** at 32k. It cannot be the cause. The vocab-scaling intuition was exactly wrong.

**2. The baseline is a different protocol, and this cell is known to move under one.** §B5's
footnote ᵈ records the phi3-mini rows as **5 runs × 8 completions**; the re-anchor used
`bench_peer`'s default **2 runs × 8**. The same footnote records this cell moving **112.4 → 116.6**
(+3.7%) purely from re-measurement. So the recorded values are **112.4, 116.6, 109.8 across three
protocols** — a spread of the same magnitude as the "regression".

**2026-08-27, protocol-matched re-measure: the regression is REAL.** phi3-mini,
`temp1.0_notrunc`, 5 runs x 8 completions — the anchor's own protocol:

    goinfer  109.7   runs [108.1, 110.5, 107.8, 110.8, 111.2]
    ollama   125.2   runs [125.5, 125.3, 125.1, 125.2, 125.1]

109.7 against the anchor's 116.6 at matched protocol: **-5.9%**, and essentially identical to the
2-run 109.8, so it was never a protocol artifact. The peer is unchanged (125.2 vs 125.6), so it is
goinfer-side. Promoted from unestablished to confirmed.

**Two corrections to the analysis above, both mine.**

*The "greedy is at parity" argument was unsupported.* There is no historical greedy phi3-mini cell
— §B5 carries only the two sampled rows — so parity with Ollama today says nothing about whether
greedy moved since the anchor. It cannot be used to localise the fault.

*The P10 elimination measured the wrong function.* `benchFilter` exercises `topFilterLogits`, the
**top_p** path; temp-only with no truncation goes through `sampleChunked`/`expChunked` instead. That
matters because P10 touched both, and the two configs moved in OPPOSITE directions on this model
(top_p +2.7%, temp-only -5.9%) — exactly the split those two paths would produce. Re-measured on
the correct path (`BenchmarkExpChunked{Fresh,Reuse}*`): P10 is **+23.1% at 32k**, +20.9% at 152k,
+21.8% at 262k. Uniformly a win. **P10 is eliminated on both paths.**

**The more useful result is a bound on where the cause can be.** At 116 tok/s a token costs ~8.6 ms;
`expChunked` at 32k costs ~96 us. Even counting the whole sampler generously, sampling is a
low-single-digit percentage of per-token time — so **no sampler change can move the end-to-end
figure by 5.9% in either direction**. The cause is in the decode path or the stack beneath it, not
the sampler. That also means the top_p/temp-only split is probably not the signal it looks like.

**What is left, and what is not worth doing.** The driver/distro upgrade is a candidate but moved
the other two models the other way (+8-9%). The goinfer range between `ca29d6c` and HEAD is the
other. Bisecting a 5.9% end-to-end delta across that range costs a 40-minute peer sweep per point;
before spending that, get a phi3-mini GREEDY number at the anchor commit — if greedy regressed too,
this is a decode-path question and the sampled framing has been a distraction from the start.

**2026-08-27, running: the anchor commit BUILT AND MEASURED TODAY, which separates the two
confounded causes instead of bisecting into them.** The reason the greedy number was missing is
that it never existed historically; the reason it was expensive is that a targeted cell used to
drag a ~40-minute sweep behind it. Both are now fixed — `scripts/bench_peer.py` gained `BENCH_BACKENDS`
and `BENCH_DEPTHS=none`, so this question costs **4 cells** rather than 13, and `ca29d6c` is built
from a worktree with the aikit its own `go.mod` pins (**v1.16.0**, against HEAD's v1.28.0).

Measuring the anchor *today* holds the driver fixed at `595.91.07` and varies only goinfer, which
the original anchor-vs-re-anchor comparison could not do — those two differed by both at once.
Both goinfer builds are compiled today with the same toolchain; ollama runs in both sweeps as the
drift control.

**Reading, fixed before the numbers land.** HEAD today reads greedy **124.6** and temp1.0 **109.7**;
the anchor recorded temp1.0 **116.6**. Let *A* be the anchor build's temp1.0 measured now:

| A | reading |
|---|---|
| ≈ 116.6 (matches its own record) | The regression lives in `ca29d6c..HEAD`. **Driver exonerated**, bisect is then justified and cheap. |
| ≈ 109.7 (matches HEAD) | goinfer's commit range is **exonerated**; the 116.6 was an old-driver number and the delta is environmental. G26 closes as environment, not code. |
| between, or neither | **Ambiguous → parked**, not argued either way. The band exists because both endpoints are ~6% apart and a middling value supports whichever story the reader brought. |

And independently, on greedy — the question the queue actually asked for:

- **anchor greedy ≈ 124.6** → greedy did not move; the effect is confined to the sampled path,
  which sits awkwardly with the arithmetic bound above (no sampler change can move end-to-end by
  5.9%). That combination is a finding to explain, not to wave through.
- **anchor greedy materially > 124.6** → greedy regressed too, the sampled framing was a
  distraction from the start, and this is a decode-path question.

**2026-08-27, RESULT at n=5 — greedy is clean, and the sampled cell turns out to be UNSTABLE on the
anchor build.** Both binaries built today from the same toolchain (anchor from a worktree at its own
pinned aikit v1.16.0; HEAD at v1.28.0), same driver `595.91.07`, ollama in both sweeps as the drift
control. Raw: `docs/measurements/g26-anchor.json`, `docs/measurements/g26-head.json`, log
`g26-anchor-vs-head_run.log`.

| cell | anchor `ca29d6c` | HEAD | Δ |
|---|---|---|---|
| **greedy** | 124.61 ± 0.12 | 124.85 ± 0.04 | +0.2% |
| **temp1.0_notrunc** | 114.17 ± **2.63** | 111.05 ± 0.45 | −2.7% |
| *ollama greedy* | *125.73* | *125.74* | *0.0%* |
| *ollama temp1.0* | *125.52* | *125.63* | *0.1%* |

**The drift control is as good as this box gets** (≤0.1% on both ollama cells across two sweeps), so
the goinfer-side comparison is sound rather than session noise.

**1. Greedy did not move — the decode path is clean.** 124.61 vs 124.85 at sd ≤ 0.12. That answers
the question this item was parked on, and it answers it the *other* way: the sampled framing was not
a distraction hiding a decode regression. Nothing in `ca29d6c..HEAD` (746 commits) moved greedy
phi3-mini at depth 128.

**2. The anchor's temp1.0 cell has 6× the variance of anything else measured here** — sd 2.63,
runs spanning **110.8 → 116.9**, against HEAD's sd 0.45 and every other cell on this page at sd
≤ 0.15. **The recorded anchor value of 116.6 sits essentially at the top of that build's own range,
and 109.7/109.8 sit near the bottom of HEAD's.** A large part of the original "5.9% regression" is
therefore a high draw compared against a low one, across protocols that never repeated the cell
enough to see its spread.

**3. What survives is at most −2.7%, and at n=5 that is not established.** Welch on the two run
sets gives t ≈ 2.6, p ≈ 0.06 — inside the band this item pre-registered as **ambiguous → parked**.
Per that rule the conclusion is parked and NOT argued in either direction.

**Parking the conclusion is not the same as shelving the question, and this cell is now cheap.**
`BENCH_BACKENDS` + `BENCH_DEPTHS=none` cut it from 13 cells to 4, so the correct response to an
ambiguous result is more of it rather than a shelf: **n=15 on both builds is running**, sweep order
reversed (anchor first) so an order or thermal effect shows up as disagreement with the pair above
rather than hiding inside it. Raw will be `g26-anchor-n15.json` / `g26-head-n15.json`.

**Do not bisect yet.** A bisect over 746 commits searches for a step change; if the effect is
substantially cell variance, every step reads as noise and the search converges on whichever commit
happened to draw high. Variance first, then bisect only if a real step survives it.

## G26 RESOLVED, 2026-08-27 (n=15) — real, HALF the claimed size, and the sampler is back

Raw: `docs/measurements/g26-anchor-n15.json`, `g26-head-n15.json`, log `g26-n15_run.log`.
Sweep order reversed from the n=5 pair (anchor first) as an order/thermal control; ollama held
within 0.15% of itself across all four sweeps.

| cell | anchor `ca29d6c` | HEAD | Δ |
|---|---|---|---|
| greedy | 124.41 ± 0.20 | 124.60 ± 0.13 | **−0.15%** (nil) |
| temp1.0_notrunc | 114.40 ± **2.07** | 111.40 ± 0.55 | **+2.7%**, t = 5.42, 95% CI [+1.91, +4.08] |

**1. The regression is real but roughly HALF what was recorded.** −2.7%, not −5.8/−5.9%. The
missing half is a sampling artifact of the original comparison: at n=15 the anchor build's temp1.0
cell spans **109.6 → 116.9**, and the recorded **116.6 is the 14th of 15 sorted values — the ~90th
percentile of that build's own distribution.** The anchor figure was a high draw; 109.7/109.8 were
low-to-central draws of HEAD's. Comparing them measured the spread as much as the change.

**2. Greedy did not move at all** (−0.15% at sd ≤ 0.20, n=15), across 746 commits. The decode path
is clean and the question this item was parked on is answered negatively.

**3. THE COST IS ON THE SAMPLED PATH, and the arithmetic that eliminated the sampler was wrong.**
Because greedy and temp1.0 differ only in the sampling config and generate identical token counts
(2560 every cell, verified), the same-build gap between them isolates the sampled-path tail:

| build | greedy | temp1.0 | sampling step | share of token |
|---|---|---|---|---|
| anchor | 8.038 ms/tok | 8.741 ms/tok | **0.703 ms** | 8.0% |
| HEAD | 8.026 ms/tok | 8.976 ms/tok | **0.950 ms** | 10.6% |

That tail got **+35% more expensive**, and the +0.247 ms/token accounts for **2.8%** of end-to-end
against the **2.7%** observed — the effect is fully explained with nothing left over.

> **CORRECTION, made the same day and before the number was acted on: this tail is NOT the sampler
> alone, and calling it "the sampling step" would have sent the next reader to the wrong function.**
> On CUDA the two configs do not differ only in host-side sampling. `cuda/resident.go:2388`
> documents `ForwardArgmax` as the greedy fast path that "reduce[s] the argmax on-device and read[s]
> back 4 B instead of the whole logits vector", and `cuda/softcap.go:25` records the consequence:
> the sampled path is "the path that also does the ~1 MB readback", and pays softcap where the
> family has it. So the 0.703 → 0.950 ms gap is **the whole sampled-path tail**: the full-vocab
> device→host readback that greedy skips entirely, plus any softcap, plus `Sampler.Sample`.
>
> The arithmetic below still refutes the recorded "sampler is too small" bound — the tail *is*
> 8–10.6% of the token, not low-single-digit. What it does not do is name which component moved.
> For scale, phi3-mini's 32064 logits are ~128 KB, which is tens of microseconds of PCIe, not 700 —
> so readback bandwidth alone does not explain the tail's size either. Attributing the +0.247 ms
> requires measuring the components, which is what the next step does.

**This refutes the bound recorded above**, which read: *"Even counting the whole sampler generously,
sampling is a low-single-digit percentage of per-token time — so no sampler change can move the
end-to-end figure by 5.9%. The cause is in the decode path or the stack beneath it, not the
sampler."* **That is wrong.** It was built from `expChunked`'s ~96 µs microbenchmark, but the real
end-to-end sampling step costs **703–950 µs — seven to ten times more.** The microbenchmark measures
one function; the config's actual path costs an order of magnitude more than it. The lesson is the
one this repo already writes down elsewhere: a sub-component measurement does not bound the total,
and here it under-bounded it by ~10x and retired the correct hypothesis for two rounds.

**4. A second, unexplained effect: HEAD is 4x more STABLE and slower** (sd 0.55 vs 2.07). Whatever
changed made the sampling step both costlier and more consistent. Note P10 — the one sampler change
between the anchors — was measured *faster* on both paths, so something else in the range more than
cancelled it; P10's elimination stands, the conclusion drawn from it does not.

**Next step, now cheap and well-aimed: do NOT bisect end-to-end.** The signal is a 0.247 ms/token
step in the sampled-path tail. Split it before bisecting:

1. **Host-side `Sampler.Sample`, whole path** (not `expChunked` alone — that is the error above), at
   phi3-mini's 32064 vocab, at both commits. `decoder/g26_sampler_bench_test.go` does this, with a
   greedy arm as the control so it decomposes exactly as the end-to-end numbers did. Minutes.
2. **Whatever Sample does not account for is above it** — the readback/sync path, which greedy's
   on-device argmax never touches. That is a different search, in `cuda/`, not `decoder/`.

Note for reading step 1: those logits are SYNTHETIC. If the regression is data-dependent, a flat
result there is evidence about the cause's shape rather than its absence — this repo has the
matching rule already, that a synthetic reproduces shape and not pressure.

## G26 CAUSE FOUND, 2026-08-27 — optimistic forward is a NET LOSS at temperature 1.0

Both steps above ran. Raw: `docs/measurements/g26-sampler-bench.log` (host sampler, both commits),
`g26-head-nooptfwd-n15.json` (the A/B), plus the n=15 pair already cited.

**Step 1 exonerated the host sampler, in the opposite direction.** `Sampler.Sample` over phi3-mini's
32064 vocab, `-count 5`, at both commits:

| benchmark | `ca29d6c` | HEAD | Δ |
|---|---|---|---|
| Sample, temp 1.0, 32k | 627.6 us | 552.6 us | **−11.9% (HEAD FASTER)** |
| Sample, greedy, 32k | 25.6 us | 18.0 us | −29.9% |
| Sample, temp 1.0, 152k | 1358.1 us | 1676.1 us | **+23.4% (HEAD slower)** |

HEAD's sampler is *faster* at the vocab that regressed, so it cannot be the cause; the remainder
(tail − Sample) went 75 us → 397 us, putting the whole move above the sampler. **Recorded separately
because it is a real finding and not this item's: the sampler CROSSES OVER with vocab — HEAD is
−12% at 32k and +23% at 152k.** That is not what P10's "uniformly faster" microbenchmarks predicted,
and the 152k regression is unexamined.

**Step 2 identified it: `6a4e0ae`, optimistic next-token forward.** It is in HEAD, absent from the
anchor, and gated `useGPU && !fastGreedy && optFwdEligible(sp) && GOINFER_NO_OPTFWD == ""` — so it
runs on **sampled decode only and never on greedy**, which is precisely the observed signature. It
overlaps the CPU sampler with a speculative next forward; on a miss that forward is discarded.

A/B on the same HEAD binary, n=15, phi3-mini, temp1.0, CUDA:

| arm | mean | sd |
|---|---|---|
| HEAD, optFwd ON (the default) | 111.40 | 0.55 |
| HEAD, optFwd OFF (`GOINFER_NO_OPTFWD=1`) | **117.84** | 0.16 |
| anchor `ca29d6c` (predates the feature) | 114.40 | 2.07 |

**Turning the feature off is worth +5.8%** — `+0.491 ms/token`. And it lands HEAD **+3.0% above the
anchor**, so the accounting is: the other 745 commits made sampled decode ~3% faster, optFwd gave
back ~5.8%, and −2.6% net is exactly what the peer sweep measured. Greedy is unmoved in every arm,
as the `!fastGreedy` gate requires.

**Why it loses here.** The feature's own gates record the hit rate collapsing with temperature —
98.0% at T=0.2, **55.6% at T=1.0**. Near-greedy it nearly always hits and the overlap is close to
free; at T=1.0 roughly half the speculative forwards are discarded, and contending for the GPU costs
more than the ~0.55 ms of host sampler the overlap hides. **This is the do-nothing-arm rule in the
repo's own words: the feature was validated where it wins and shipped enabled for every sampled
config.** It is not a bug and the mechanism is sound — the default is what is wrong.

**SCOPE, so this is not over-read.** One model, one temperature, one depth, CUDA, greedy-comparable
protocol. It does NOT show optFwd is a loss generally; at T=0.2 the same evidence suggests it wins.
What is established is that **at T=1.0 on this pairing it costs 5.8%**, and that the on/off decision
is currently made without reference to temperature.

**A PREDICTION I MADE AND GOT WRONG, recorded because it was pre-registered.** I predicted that
disabling optFwd would restore the anchor's *spread* as well as its mean — the story being that
overlap hides the sampler's data-dependent cost. It does not: HEAD-no-optFwd has **sd 0.16**, tighter
than either other arm. The variance story is refuted. **The anchor build's sd 2.07 on this cell
remains unexplained and is now a separate open question**, not a side effect of this one.

**Follow-ups, none of them started:**
1. **Gate optFwd on temperature or on realized hit rate.** `OptFwdStats{Guessed, Hit}` already
   tracks it per run, so an adaptive gate has its input. Needs a sweep across T before a rule.
2. ~~**The 152k sampler crossover** (+23.4%)~~ — **RETRACTED 2026-08-27, measured end-to-end and
   the microbenchmark had the DIRECTION WRONG.** See the retraction below.
3. **The anchor's temp1.0 variance** (sd 2.07 vs 0.16-0.55) — mechanism unknown.

**Follow-up 1 RUNNING (2026-08-27): the temperature ladder, and how it will be read.** T ∈ {0.2,
0.4, 0.6, 0.8, 1.0}, temp-only/no-truncation, optFwd ON vs OFF on the same HEAD binary, n=6,
phi3-mini/CUDA, peer kept in both arms as drift control. Raw: `g26-tsweep-optfwd-{on,off}.json`.
Configs added to `scripts/bench_peer.py` as a ladder because two endpoints cannot locate a crossover.

Read Δ = (OFF − ON)/ON at each T. **Δ > 0 means the feature LOSES at that temperature** (turning it
off is faster); Δ < 0 means it wins.

| outcome | reading, fixed in advance |
|---|---|
| Δ < −1% at low T, > +1% at high T | A real crossover. Gate optFwd on temperature, with the threshold set BELOW the measured crossover, not at it. |
| Δ > +1% at every T | The feature does not pay on this pairing at all. Default should be off pending a config where it wins — its 98%-hit-rate case is T=0.2, so a null there is the strong result. |
| Δ < −1% at every T | T=1.0's 5.8% is not temperature-driven and the mechanism story above is wrong; reopen. |
| \|Δ\| < 1% throughout | No effect to gate; the T=1.0 result needs re-examining before anything is changed. |

**LADDER RESULT (2026-08-27), n=6, phi3-mini/CUDA/depth 128 — a real crossover at T ≈ 0.26.**
Raw: `g26-tsweep-optfwd-{on,off}.json`, log `g26-tsweep_run.log`.

| T | optFwd ON | optFwd OFF | Δ = (OFF−ON)/ON | verdict |
|---|---|---|---|---|
| 0.2 | 119.34 ± 1.17 | 118.03 ± 0.15 | **−1.1%** | optFwd **wins** |
| 0.4 | 114.80 ± **3.51** | 118.00 ± 0.13 | +2.8% | loses |
| 0.6 | 111.23 ± 1.26 | 118.20 ± 0.17 | +6.3% | loses |
| 0.8 | 110.34 ± 0.76 | 117.80 ± 0.17 | +6.8% | loses |
| 1.0 | 111.72 ± 0.44 | 117.89 ± 0.12 | +5.5% | loses |

**The OFF arm is FLAT** — 117.80 to 118.20, a 0.40 tok/s spread across the whole range — which is
what makes it a baseline rather than a second variable: the sampler's work does not depend on
temperature, only the drawn tokens do. Every movement in the table is the feature's.

**Reading it by the rule fixed above: a real crossover, so gate on temperature with the threshold
set BELOW the measured 0.26 — T ≤ 0.2 on this evidence.** Note the payoff is badly asymmetric: the
win is **1.1%** and the losses run to **6.8%**, so a threshold placed at the crossover risks ~6x what
it gains if the crossover moves on another model. That asymmetry, not the crossover, is the argument
for a conservative default.

**What the current default costs.** optFwd is on for ALL sampled decode. Typical chat sampling sits
at T = 0.7–1.0, which is squarely the losing regime, so goinfer's default sampled decode on this
pairing is **~5.5–6.8% slower than simply turning the feature off**. That is the practical result of
this whole item, and it is larger than the regression that started it.

**Second-order, and it is the hit-rate lottery made visible: the ON arm's VARIANCE peaks mid-ladder**
(sd 3.51 at T=0.4, against 0.12–0.17 everywhere on the OFF arm). At T=0.2 the guess nearly always
hits and at T≥0.6 it nearly always misses; in between, each run draws a different hit rate. So the
feature adds unpredictability as well as cost in the middle of its range — and this, not the anchor
build, is where an overlap-related variance story actually holds. **The anchor's sd 2.07 is still
unexplained**; my earlier attempt to attribute it to overlap was refuted (HEAD-no-optFwd is the
tightest arm measured, sd 0.16).

**SCOPE — n=1 model.** phi3-mini, depth 128, CUDA, one card. The crossover is a property of the
hit-rate curve, which is model- and prompt-dependent; a second model is needed before a threshold
ships. Shipping a gate off one model would repeat precisely the error that put this feature on by
default.

**SECOND MODEL RUNNING (2026-08-27), and it is chosen to TEST the mechanism rather than just add a
point.** qwen2.5-coder-1.5B, vocab **151936** against phi3-mini's 32064. optFwd's benefit is bounded
by how much sampler the overlap can hide, and that share differs by 3.4x between the two (both from
today's HEAD, optFwd OFF):

| model | vocab | decode step | sampler | sampler share of the sampled token |
|---|---|---|---|---|
| phi3-mini | 32064 | 8.026 ms | 0.457 ms | **5.4%** |
| 1.5B | 151936 | 4.535 ms | 1.009 ms | **18.2%** |

**Prediction, fixed before the run: the 1.5B's crossover should sit at a HIGHER temperature than
phi3-mini's 0.26.** More sampler to hide means the overlap keeps paying at hit rates that would
already be uneconomic on phi3-mini. The reasoning is falsifiable in both directions and the outcomes
decide the SHAPE of the gate, not merely its constant:

| outcome | what it means for the gate |
|---|---|
| crossover moves HIGHER, as predicted | The mechanism story holds and the crossover is **model-dependent**. A single temperature constant is then the wrong shape — it must be conservative enough for the WORST model, giving up most of the win on the best. Argues for the hit-rate-adaptive gate. |
| crossover lands at ~0.26 again | Temperature alone predicts the crossover across a 4.7x vocab range. A constant threshold is defensible and much simpler; ship T ≤ 0.2. |
| crossover moves LOWER | The sampler-share reasoning is wrong and something else drives it. Do not gate on either axis until that is understood. |

**SECOND-MODEL RESULT (2026-08-27): PREDICTION CONFIRMED — the crossover moves 0.26 -> 0.95, and a
single temperature constant is therefore the WRONG SHAPE for the gate.** Raw:
`g26-tsweep15-optfwd-{on,off}.json`, log `g26-tsweep15_run.log`. n=6, CUDA, depth 128.

| T | phi3-mini (vocab 32064) | 1.5B (vocab 151936) |
|---|---|---|
| 0.2 | −1.1% **wins** | **−7.4% wins** |
| 0.4 | +2.8% loses | **−6.0% wins** |
| 0.6 | +6.3% loses | **−5.1% wins** |
| 0.8 | +6.8% loses | −0.9% no effect |
| 1.0 | +5.5% loses | −0.9% no effect |
| **crossover** | **≈ 0.26** | **≈ 0.95** |

The sampler-share reasoning predicted both the direction and roughly the size: 18.2% of the sampled
token available to hide on the 1.5B against 5.4% on phi3-mini. **On the 1.5B optFwd is never a
significant loss anywhere in the measured range** — win or neutral throughout. The harm is
concentrated on SMALL-VOCAB models, which is the opposite of where a reader would look for it.

**A SINGLE TEMPERATURE THRESHOLD IS PROVABLY WRONG IN BOTH DIRECTIONS, and this is what the second
model was run to establish:**

- Set it safe for phi3-mini (T ≤ 0.2) and the 1.5B forfeits **6.0% at T=0.4 and 5.1% at T=0.6** —
  real wins, thrown away.
- Set it for the 1.5B (T ≤ 0.95) and phi3-mini pays **2.8 / 6.3 / 6.8 / 5.5%** across T=0.4–1.0.

There is no constant that is right for both, and they differ by only a vocab size.

**REFINEMENT ON WHAT I PRE-REGISTERED, because "gate on hit rate" is not quite sufficient either.**
The pre-registration said this outcome argues for a hit-rate-adaptive gate. That is the right
direction but an incomplete rule: the feature's value is roughly

    value  ~=  p_hit * c_sampler  -  (1 - p_hit) * c_miss

so hit rate alone does not decide it — **`c_sampler` is the other term, and it is exactly what
differs between these two models.** A hit-rate threshold tuned on phi3-mini would still be wrong on
the 1.5B, for the same reason a temperature threshold is. The gate needs both, and both are
measurable at runtime: `OptFwdStats{Guessed, Hit}` already gives the hit rate, and the sampler's
share is the greedy-vs-sampled step cost, which the decode loop can time directly.

**RECOMMENDATION (not implemented — this is a design change and wants sign-off): do not ship a
temperature constant.** The defensible interim is that the current unconditional default is wrong on
small-vocab models and should not stay as-is. Two models is enough to rule out the constant; it is
not enough to fit the two-term rule, which needs points spanning `c_sampler` rather than two of them.

**Temperature is a PROXY and the ladder should not be mistaken for the mechanism.** What determines
the feature's value is the realized hit rate, which `OptFwdStats{Guessed, Hit}` already measures per
run; temperature only predicts it. A hit-rate-adaptive gate is strictly better-targeted than a
temperature threshold and has its input available today — but it can only be *evaluated* against a
ladder like this one, which is why the ladder comes first. **One model, one depth: a gate shipped
off n=1 model would repeat exactly the mistake that put this feature on by default.**


**G25 · Does the oracle bar need a sparsity axis, or is per-precision enough?** — PARKED with a
trigger, 2026-08-27. Either box, desk work first.

**The precision half is DONE** (`c62f2b7`). `decoder/real_oracle_test.go` had one bar, 0.99, whose
own comment said it was the int8 W8A8 number — and `qwen3_next` was measured against it as the
repo's first int4 T3, missing by 0.000124. Now `int8int8`/`int8` keep 0.99, `int4` takes 0.98 (G5,
pre-registered before the checkpoint finished downloading), and an unregistered precision fails hard
rather than inheriting someone else's bar. The oracle passes at its measured 0.989876 and the
`qwen3_next` manifest row is `validated` / `real-model-oracle` (`8f003f2`).

**What is still open is the SHAPE, not the number.** A scalar bar per precision assumes precision is
the only thing that moves the cosine. In a sparse MoE it is not: quantization can change *which*
experts fire, a discrete flip no averaging smooths, so divergence should track sparsity too.

| family | routing | active | measured |
|---|---|---|---|
| `qwen3_next` | 10/512 | 1.95% | 0.98988 at int4 |
| `nemotron3nano` | 6/128 | 4.7% | 0.978 at int8 activations, 0.9977 at f32 |
| dense families | — | — | no expert flipping at all |

Three families at one precision could each want a different bar; `int4 = 0.98` may be loose for a
dense model and tight for something sparser still.

**Parked rather than solved, deliberately.** There is exactly ONE int4 T3 family today, so any
sparsity rule derived now would be fitted to n=1 — and this queue's own standard is that a bar moves
only with a mechanism, not with a number. Fitting a curve through a single point is the same error
wearing a lab coat.

**Trigger to re-open — any one of:**
1. A **second** int4 T3 family lands, giving a real second point.
2. An int4 gate fails within ~0.005 of 0.98, where the bar's shape starts deciding outcomes rather
   than merely being defensible.
3. A dense family is added at int4 — the case where 0.98 is most likely too loose, since it has no
   expert-flip term at all.

Until one fires, the per-precision split is correct and sufficient, and nothing is mis-gated.

## Struck — decided against, kept so the decision is visible

- ~~**Default `top_k`**~~ — truncating the distribution changes which tokens are reachable, which is
  a silent substitution of something other than what was asked. Document it; do not default it.
- ~~**Change the global `--quant` default**~~ — CPU inverts the CUDA quant ordering, so a single
  global default cannot be right for both, and the evidence is one model on one box, never
  reproduced at 1.5B.
- ~~**Force cross-architecture float agreement**~~ — explicit `math.FMA` everywhere is a software
  fallback on amd64 that costs the SIMD performance the CPU backend exists for. Scoped in the policy
  instead.
- ~~**Slab restructure for expert slots**~~ — the control produced the reverse of fragmentation's
  prediction: a fresh heap with ~10 large allocations had *worse* contiguity (32–64 MiB) than the
  slot-loaded heap (96–128 MiB) at the same free figure.
- ~~**aikit branch protection**~~ — required checks force PR-only merges, which is friction against
  a threat model aikit doesn't have. The gate ritual is the enforcement. Revisit at v1.0.
- ~~**Metal `ResidentGreedy` gap**~~ — measured **net-negative**. Kept here rather than under group
  P because it is not work. The 2026-08-12 audit reached the same conclusion **independently**, from
  code, without access to the measurement — recorded as a corroboration of that audit's calibration,
  which is the only reason the entry is worth keeping at all.

## Done

_(append with commit sha and date)_

**G12 · `role: "developer"` silently demoted to a user turn on the OpenAI surfaces** — `mac`,
**DONE `4ca19e9` (2026-08-25).** Claimed and finished the same day. One `case "developer":` arm in
`messagesToTurns` (`internal/serveapp/openai.go`), which all four OpenAI-side surfaces funnel
through, so chat/completions, the Responses message-item path, the tools path and the vision path
were covered at once. Precedence needed no new concept — two `system` messages already resolve
last-one-wins and the alias inherits it.

**Two corrections to the record, both worth more than the fix:**

1. The entry was first written (`9a9594c`) as "the known first blocker for the dsh Tier-0 run". It
   was not a blocker — DeepSeek Harness ships `compat.supportsDeveloperRole: false`, so the run
   could always have happened. Corrected at `0221d32`. What made it worth sequencing first was
   that the failure was **silent**, not that it was blocking.
2. **The remaining Step-0 assumption was wrong, and the wrong half was the one nobody checked.**
   The task doc said a developer-role message on `/v1/messages` was "structurally impossible"
   because Anthropic carries `system` as a top-level field, and asked only that the error be
   confirmed clean. There is no error: `anthropicMessage.Role` is a free string, `anthropicRole`
   maps everything that is not `"assistant"` to a user turn, and **there is no role validation
   anywhere in `internal/serveapp`** — so the same silent demotion was live on that surface too.
   Left demoted **deliberately** (the Anthropic API has no developer role; honoring one invents
   behavior upstream does not have, on a surface whose bar is "works for the apps that matter"),
   but now pinned by `TestAnthropicDeveloperRoleStaysUser` so the decision is visible rather than
   silent. A prediction that a thing is impossible is not a check that it is.

**Gate power, both directions:** a test pinning the old demotion was written first and shown
failing against the fix; the two new alias gates were then shown failing with the arm reverted. The
three that pin unchanged behavior (`instructions` plumbing, the non-goal guard, the Anthropic pin)
pass either way, as they should. Equality gate is byte-identical rendering against the equivalent
`system` request across all six families plus the `rawPrompt` fallback.

**Unblocks** Tier 0 of `docs/scoping-dsh-goinfer.md` — the dsh run can now start past this.

**Follow-up filed as G13** (`docs/queue-engineering.md`): demote-vs-alias was the wrong menu for
`/v1/messages`. Upstream rejects an illegal role, and `anthropicRole`'s
everything-non-assistant-becomes-user mapping means ANY typo'd or invented role silently
restructures the conversation today — `developer` was just the instance that got caught. Role
validation kills the class; the pin above is its recorded before-state. Not blocking Tier 0.

**Second exhibit for E2's lesson.** E2's four "demotion judgments" turned out to be two engineering
bugs that only a released checkpoint could reveal: a generated fixture cannot disagree with the
loader it gates. This is the same shape in different clothes — a predicted branch cannot disagree
with the predictor, so "structurally impossible" typed where a check belonged was, with mechanical
inevitability, the half that was wrong. If `docs/parity-coverage-policy.md` picks up the
fixture rule on its next edit as E2 suggested, both belong in that paragraph.

## Sequencing — release BEFORE G2

**Revised, and D3 is OUT of the release — no rebase attempted.**

> **cut the release → G2 → D3 design read → B1, B2 → mac batch**

The README change in this release is a **retraction**: the workaround language goes away because the
cap holds. D3, if it survives its design read, is an *addition to adjacent text later* — not the same
edit made twice, which was the argument for including it.

A repo-wide mechanical diff immediately before a tag costs bisectability and reasoning room and buys
the modernizers nothing. G2 is not urgent and never was; it is cleared, which is different from being
next.


## Freshness sweep — C, D, E, G (2026-08-12)

F was **fifteen for fifteen already fixed**, because it was seeded from `docs/completed/`. These
groups were seeded from conversation, and **the rate is much lower**, which is the useful result:

| entry | state | evidence |
|---|---|---|
| C1 drain fix — CUDA verification | **open** | no CUDA unload/drain test found |
| C2 out-of-tree consumer audit | **open** | needs a fresh no-repo session by design |
| C3 Metal consumer window | **RUN + CLOSED 2026-08-19 (metal/v0.14.0)** | v0.13.0 run (2026-08-15) found `go install …@v0.13.0` BROKEN by the committed `replace`; replaces removed from all 4 submodule go.mods, RELEASING.md updated. **v0.14.0 re-run (2026-08-19) verifies the fix reached consumers for real:** `go install …/metal/cmd/serve@v0.14.0` builds (13.7 MB, clean GOPATH); resident decode via `Load(Backend:"metal")`+blank-import at **65.4 tok/s out-of-tree** (was 65.2 at v0.13.0 — no regression); cgo-free otool-confirmed both paths. Full note: `docs/measurements/c3-metal-consumer-window-v0.14.0.md` (supersedes the v0.13.0 note, kept as history). |
| C4 soak testing | **open** | `internal/serveapp/fuzz_test.go` and `internal/serveapp/chaos_test.go` exist; neither is an hours-long soak |
| D1 trace tap + launch-site table | **open** | no coverage table in `docs/` |
| D2 launch-wrapper commit 1 | **open** | no `cuda/internal/gen` |
| D3 parked flag-pair | **open**, design read done above | — |
| E1 v1.0 gate as written criteria | **open** | prose item, no tree anchor |
| E2 four per-family demotions | **DONE 2026-08-15 `c3e43c8`** — and NO demotions | all four validated at T3, so the judgment the entry anticipated never had to be made: gpt2 `full-forward-oracle`, granitemoehybrid + nemotron_h `real-model-oracle`, kimi_k2 `shared-path (via deepseek_v3)`. **The tripwire now enforces 23/23, up from 19/23.** The two hybrids needed loader fixes first — neither could load its RELEASED checkpoint, and granite was roping a NoPE model (see the commit; the demotion rule's "unfinished does not qualify" is what forced fixing over demoting) |
| E3 freeze re-declaration | **DONE `cda8cfe`** | re-declared as a proof requirement, with decider and date |
| E4 `scripts/bench_compare.sh` fix or retire | **FIXED** | it now opens with *"goinfer's OWN numbers only. NOT a peer comparison"* and points at `scripts/bench_peer.py`, which drives both sides |
| E5 promo drafts | **unverifiable** | held in conversation, nothing in the tree to check |
| E6 aikit release | **CLOSED 2026-08-12** — superseded by events, not by reversal | aikit cut `v1.17.0`/`gpu/v0.28.0` (`ada417e`); goinfer is on it (`f33fcaf`). The release met E6's own "a reason a consumer can receive" test |
| G1 LFM2.5 family | **open** | no LFM2 code in the tree |
| G2 `go fix` modernizers | **DONE `3d6ae1e`** | — |

**Rate: 1 of 13 previously-open entries was silently already fixed (E4), against F's 15 of 15.**
Two more (E3, G2) were closed by this campaign and were already recorded.

**That difference is the finding.** F was seeded from a *filed audit* — work done elsewhere, reported
once, never propagated back. C/D/E/G were seeded from *conversation*, where the person who did the
work was the person holding the list. **The burial folder is what produced the 15/15, not the passage
of time.** So the sweep paid for itself once, on the group that came from a document, and should not
be assumed to pay again on groups that did not.

E5 is recorded as **unverifiable** rather than open: nothing in the tree can confirm or deny it, which
is a different state and should read as one.


## Description sweep (2026-08-12) — does each entry match its source?

The status sweep found 1 of 13. **D3 shows description can be wrong while status is right, and
description is what someone acts on.** So: for every open entry with a source outside conversation —
a branch, a commit, an audit line, a script — does the entry describe it correctly?

*(A goldens run in a fresh `git worktree` is fixture-less: the same commit proved 33 goldens in the
main checkout and 7 in the worktree, because the fixture checkpoints are gitignored. The refresh
script now says so when skips outnumber passes — `scripts/refresh_parity_hashes.sh`. Found by running
D3's refresh in the rebase worktree and getting `goldens=7`.)*

| entry | source | description matches? |
|---|---|---|
| D3 | branch `flag-pair-moe-cache` + `BRANCH-NOTE.md` | **NO — corrected `2d28358`.** Called a "parked flag-pair" on a workaround premise; it is an API-surface promotion following `KVPrecision` |
| B4 | a stash that does not exist here | **unverifiable** — the description is all that survives, and it names a file that resolves nowhere |
| C1 | `588052b` (the drain fix) | matches — Metal-verified, CUDA arm untested |
| D2 | design recorded in-entry, no branch | matches; no external source to drift from |
| E2 | `testdata/parity_manifest.json` | matched when written; **resolved 2026-08-15** — all four are now `validated`. What the description got wrong was not its status but its FRAMING: it called them "demotion judgments", inheriting the campaign doc's "validation + recording, NOT engineering". Two of the four were engineering, and only a released checkpoint could say so |
| E4 | `scripts/bench_compare.sh` | **stale** — the entry says "status unconfirmed, may still measure the two sides differently"; the script now refuses that use and points at `scripts/bench_peer.py`. Corrected in the status sweep above |
| E6 | aikit tree + `gpu/v0.27.0` | **now stale by design** — the tag it pinned is superseded by `gpu/v0.28.0`. Re-checked at the bump: `be049df` is an ancestor of `gpu/v0.27.0`, and across `gpu/v0.27.0..gpu/v0.28.0` the quantized GEMV PTX is byte-identical |
| G1 | `docs/scoping-lfm2.md` | matches |
| P1, P2, P4, P5, P8 | audit lines + the cited source | match; each carries a measured figure or an explicit ESTIMATE label |

**Split: 9 entries had an external source and were checkable; 2 of those 9 were wrong (D3, E4).
4 entries — C2, C3, E1, E5 — have no source outside conversation and are recorded as unverifiable
rather than checked.**

**THE TRIGGER, not a cadence.** An entry's description **and the specific details inside it** — counts,
file names, measurements that no lint covers — are re-read against their source **at the moment the
item is picked up for work** — nothing schedules this and nothing lints it. That is when the
description matters, when someone is already loading the context anyway, and it is exactly what caught
D3: the read happened because the item was next, not because a sweep came due. The cost falls at the
only point where the drift would have changed what someone did.

**That rate (2 of 9) is higher than the status sweep's (1 of 13), and the two are not the same
population.** A description drifts silently because nothing re-reads it against its source; a status
drifts only when work lands elsewhere. The queue's SHA and path citations are now linted, but **no
lint reads an entry's prose against the branch note or audit line it describes** — that remains a
person's job, and this sweep is its baseline.

## Sequencing

**D3 (loaded and bounded) → the mac batch as one session → B1, B2.**

Within the mac batch, **C3 goes FIRST**, not last: it is the largest completely uncovered surface and
it sank once already. Batching it behind two chores is precisely how that happened. Then
`metal-rope-merge`'s push, then B4's stash check.

**C3 DONE 2026-08-15** (auto-pickup fired on the `v0.13.0` tag). cgo-free/no-Xcode build verified via
the require path; two real gaps found — `go install …/metal/cmd/serve@v0.13.0` is broken by the
committed `replace github.com/townsendmerino/goinfer => ../` in the tagged `metal/go.mod`, and
resident-metal decode isn't drivable via the public API (needs the serve binary, which `go install`
can't build). Full findings + the resolved dep set in `docs/measurements/c3-metal-consumer-window.md`.
**Follow-up worth queuing:** tag `metal/` from a replace-free tree so `go install` works.

**C3 RE-RUN 2026-08-19, CLOSED for real** (auto-pickup fired again — aikit v1.17.1 → v1.21.0 — on
the `v0.14.0`/`metal/v0.14.0` tags). The follow-up above is done: `go install
…/metal/cmd/serve@v0.14.0` now **WORKS** from a clean GOPATH (13.7 MB binary, no checkout) — the
replace-free fix reaching consumers for the first time. Resident-metal decode via the public API
also verified out-of-tree: 65.4 tok/s decode-only, consistent with v0.13.0's 65.2 (no regression
across the aikit bump). RELEASING.md Step 2's two Mac gates (`GOWORK=off go build -tags metal ./...`
standalone build; the device gate — then a bash script, which needed Homebrew bash because macOS's
stock `/bin/bash` 3.2 cannot run `declare -A`; that dependency went away with the script when E8
replaced it with `go run ./cmd/gate gpu`) both passed clean on a committed tree; the gate's only finding was
the pre-existing, already-documented `c8b65ba` local-reflog artifact (allowlisted 2026-08-12,
CI-invisible), unrelated to this release. Full findings in
`docs/measurements/c3-metal-consumer-window-v0.14.0.md`; gate log archived at
`docs/measurements/gpu_gate_metal_v0.14.0_4d91858_FAIL-c8b65ba-only.log`.

## Draft: contents of the next release

**Not a version number** — that is a separate call. This is what has accumulated since
`demo/agent/v0.11.0` (93 commits) that a user would notice, and **none of it depends on the freeze
decision**.

### The headline: the 26B expert cache sizes itself correctly

The defect that opened this campaign was live in the product and is fixed. On an 8 GB card the
runtime auto-capped the MoE expert cache to **34 slots/layer, which allocates and then cannot
launch** — the forward produced **zero tokens**.

- **A5 (`6091e7a`)** — the cap is a **search over the granularity form**, not a division. The driver
  charges each of four buffers per layer its own whole 2 MiB quantum, so the requirement is a step
  function; at 34 all four tip at once, putting it 203,816,960 B over free. Verified through the
  shipping auto-cap path: `capping to 34` → 0 tokens becomes `capping to 33` → coherent output.
- **A9-FIX (`0103b49`)** — the deferred first-launch reservation (`moe_route` takes 138,412,032 B of
  local memory the first time it runs) is now paid **before** the free reading that sizes the cache,
  so the cap is correct by construction rather than covered by a margin. Costs two slots, and that is
  the point: 384 MiB now means 384 MiB.
- **A3 (`e42e83e`)** — a launch OOM now names the kernel and **both** the requested and effective
  slot counts, instead of a bare `cuLaunchKernel: CUDA_ERROR_OUT_OF_MEMORY`.
- **README** — the manual-workaround section is retracted and replaced with what the cap now
  accounts for, plus a version test (`capping to 33` has the fix, `34` does not).

### Performance, all bit-identical

- **P3 (`4c26a58`)** — Gemma's final-logit softcap parallelised: **1.43 ms → 640 µs** per sampled
  token at 262,144 vocab. Sampling path only; greedy never paid it.
- **P6 (`eea7f29`)** — MoE experts share one gate/up buffer pair per token instead of one per expert:
  **16 allocations → 2** at top-k 8.
- **P7 (`91f359f`)** — W4A8 reaches the per-stream `Workspace` it was silently excluded from, ending
  a fresh allocation per projection per token.

### Verification a user can check

- **int4 forward goldens** (`1d0d1ed`) — 23 fixtures across 16 architectures. int4 is the documented
  default quantization and **nothing gated it** before this.
- The goldens refresh went from **19 passed / 0 quantized** to **33 passed / 14 quantized**, and now
  prints its composition rather than a bare count.

### Known-unfixed, disclosed

- **A10** — a ~150 MiB driver allocation floor: memory `cuMemGetInfo` reports as free and
  `cuMemAlloc` will not hand out, at any request size down to 1 MiB. Measured, unattributed. It is
  why the margin cannot simply be lowered to recover the two slots.

<!-- sha-lint: allow c8b65ba UNPUSHED — the PRE-REBASE D3 flag-pair commit, on the local-only branch `flag-pair-moe-cache`; never pushed, so CI cannot resolve it (it failed there 2026-08-12). DELIBERATELY NOT re-pointed to its rebased successor `bacc04c` (on main via the D3 merge): the two have DIFFERENT patch-ids, and the passage citing this one is historical — it describes the branch as it stood BEFORE the rebase ("its branch predates the fix it completes"), which `bacc04c` no longer illustrates. Re-pointing would keep the lint green by making the surrounding sentence false. Allowlisted rather than laundered. Flagged 2026-08-12 -->
<!-- sha-lint: allow d682315 UNPUSHED — Metal branch `metal-rope-merge`, mac-local; not on origin and not in any clone here. Owner: whoever cited it. P4's "already implemented, snapshot-golden byte-exact" rests on a commit only that machine can see; push the branch or the claim stays unverifiable from anywhere else. Flagged 2026-08-12 -->

<!-- CITATION-INDEX: generated by scripts/queue_citation_lint.py --update; do not edit by hand -->

## SHA index

Generated. Every commit id cited above, with the subject it resolved to at the time
of generation. Regenerate with `scripts/queue_sha_lint.py --update`.

| sha | subject |
|---|---|
| `0103b49` | fix(cuda): pay the deferred reservation before sizing the cache (A9-FIX) |
| `0221d32` | docs: the developer-role task is NOT a blocker -- it is silent-wrong, which is worse |
| `1d0d1ed` | test(decoder): int4 forward goldens — 23 fixtures, 16 architectures (Q1c) |
| `2d28358` | docs(branch-note): re-derive against the corrected cap (D3 design read) |
| `3d6ae1e` | chore: go fix modernizers, one deterministic pass (G2) |
| `4c26a58` | perf(cuda): parallelise the Gemma final-logit softcap, bit-identical (P3) |
| `4ca19e9` | fix(serve): accept `role: "developer"` as an alias for `system` (G12) |
| `4da116d` | perf(decoder): P10 — reuse Sampler's full-vocab scratch buffer across draws |
| `588052b` | serve: drain in-flight requests before freeing an unloaded model (fixes the leak safely) |
| `6091e7a` | fix(cuda): size the expert cache by SEARCH over the granularity form (A5) |
| `61b1e03` | bench: add temp1.0_notrunc, the config §B5's temp-only rows actually used |
| `6a4e0ae` | decoder: optimistic next-token forward for sampled decode (Metal-verified, CUDA untested) |
| `8f003f2` | parity: v0.15.0 sweep GREEN at bd085de; qwen3_next validated by real oracle |
| `91f359f` | fix(decoder): matmulInto dispatches on the property, not on W8A8 (P7) |
| `9a9594c` | docs(prompts): task brief for `role: "developer"` compat on the serve surface |
| `ada417e` | [aikit] scripts: ptx-repro is n/a on darwin, keyed on the PLATFORM not on NVRTC's absence |
| `bacc04c` | feat(serve): --moe-cache-experts / --moe-cache-slots — PARKED on the freeze |
| `be049df` | [aikit] gpu(gemv): explicit __fmaf_rn in the quantized GEMV — the bit-identity contraction rule |
| `c3e43c8` | E2: the four pending families get real oracles — and two of them were decoding released checkpoints wrong |
| `c62f2b7` | test(decoder): give the real-model oracles a bar per precision |
| `ca29d6c` | cuda: resident context cap becomes configuration-derived (-ctx), VRAM-checked at load |
| `cda8cfe` | docs: re-declare the freeze as a proof requirement; clear G2 for amd64 alone |
| `e42e83e` | fix(cuda): name the kernel and both slot counts when a launch runs out of memory |
| `eea7f29` | perf(decoder): one gate/up pair per token in MoE, not one per expert (P6) |
| `f33fcaf` | chore(deps): aikit v1.16.0 -> v1.17.0, aikit/gpu v0.27.0 -> v0.28.0 |

## Path index

Generated. Every `file:line` cited above, the repo it resolved in, and the trimmed
content of that line. A line that MOVED is reported with its new number; content that
has VANISHED is red, because the citation then claims something the file no longer
supports.

| doc \| path:line | repo | line content |
|---|---|---|
| `docs/QUEUE.md|cuda/resident.go:2388` | goinfer | `// ForwardArgmax is the greedy fast path (decoder.ResidentGreedy): reduce the argmax on-` |
| `docs/QUEUE.md|cuda/softcap.go:25` | goinfer | `// This runs on the SAMPLING path only. ForwardArgmax reduces the argmax on-device and r` |
| `docs/audit-2026-09-02.md|chat/chat.go:1` | goinfer | `// Package chat renders a conversation into the exact prompt string a model's` |
| `docs/audit-2026-09-02.md|chat/chat.go:18` | goinfer | `"github.com/townsendmerino/goinfer/tokenizer"` |
| `docs/audit-2026-09-02.md|chat/gemma4_tools.go:28` | goinfer | `if s := strings.TrimSpace(system); s != "" {` |
| `docs/audit-2026-09-02.md|chat/templates.go:156` | goinfer | `func ChatML() *Template {` |
| `docs/audit-2026-09-02.md|chat/templates.go:197` | goinfer | `date := timeNow().Format("02 Jan 2006")` |
| `docs/audit-2026-09-02.md|chat/tools.go:56` | goinfer | `func (t *Template) RenderToolsSegments(system string, turns []Turn, tools []Tool) []Segm` |
| `docs/audit-2026-09-02.md|chat/tools_test.go:99` | goinfer | `// TestGemma4_declaration_byteExact pins the Gemma 4 declaration micro-language` |
| `docs/audit-2026-09-02.md|cmd/gate/configs.go:34` | goinfer | `Env: map[string]string{` |
| `docs/audit-2026-09-02.md|cmd/gate/gpu.go:1072` | goinfer | `func (g *gpuGate) metalModel() string {` |
| `docs/audit-2026-09-02.md|cmd/gate/gpu.go:1088` | goinfer | `_, cr, out := g.run(cell{` |
| `docs/audit-2026-09-02.md|cmd/gate/gpu.go:1122` | goinfer | `_, cr, out := g.run(cell{` |
| `docs/audit-2026-09-02.md|cmd/gate/gpu.go:1155` | goinfer | `lint := exec.Command("python3", "scripts/queue_citation_lint.py")` |
| `docs/audit-2026-09-02.md|cmd/gate/gpu.go:219` | goinfer | `func (g *gpuGate) noteIfEmpty(c cell, res *results) {` |
| `docs/audit-2026-09-02.md|cmd/gate/gpu.go:349` | goinfer | `func detectBackend() string {` |
| `docs/audit-2026-09-02.md|cmd/gate/gpu.go:374` | goinfer | `for _, v := range []string{"GOINFER_HEAVY_TESTS", "GOINFER_DRAIN_GROUP"} {` |
| `docs/audit-2026-09-02.md|cmd/gate/gpu.go:383` | goinfer | `switch g.backend {` |
| `docs/audit-2026-09-02.md|cmd/gate/gpu.go:386` | goinfer | `case "metal":` |
| `docs/audit-2026-09-02.md|cmd/gate/gpu.go:422` | goinfer | `switch g.backend {` |
| `docs/audit-2026-09-02.md|cmd/gate/gpu.go:508` | goinfer | `// The header used to read "CUDA kernels + parity" while running NEITHER the resident pa` |
| `docs/audit-2026-09-02.md|cmd/gate/gpu.go:509` | goinfer | `// NOR anything that asserts a forward. Every resident parity gate is behind `goinfer_te` |
| `docs/audit-2026-09-02.md|cmd/gate/gpu.go:543` | goinfer | `// ---- 2b. resident PARITY gates — the forward is asserted here ----` |
| `docs/audit-2026-09-02.md|cmd/gate/gpu.go:657` | goinfer | `Env: map[string]string{"CGO_ENABLED": "0", "GOINFER_HEAVY_TESTS": "1"},` |
| `docs/audit-2026-09-02.md|cmd/gate/gpu.go:758` | goinfer | `bin := filepath.Join(os.TempDir(), "gpu_gate_serve")` |
| `docs/audit-2026-09-02.md|cmd/gate/gpu.go:829` | goinfer | `if v := os.Getenv("GOINFER_NVRTC_DIRS"); v != "" {` |
| `docs/audit-2026-09-02.md|cmd/gate/gpu.go:903` | goinfer | `ptxFiles, _ := filepath.Glob(filepath.Join("cuda", "testdata", "*.ptx"))` |
| `docs/audit-2026-09-02.md|cmd/gate/gpu.go:980` | goinfer | `func (g *gpuGate) metalSuite() {` |
| `docs/audit-2026-09-02.md|cmd/gate/gpu.go:994` | goinfer | `func (g *gpuGate) metalCgoFree() {` |
| `docs/audit-2026-09-02.md|cmd/gate/parity.go:140` | goinfer | `// DERIVED FROM THE TAGGED FILES, not hand-written. The hand-written pattern could` |
| `docs/audit-2026-09-02.md|cmd/gate/parity.go:19` | goinfer | `// THIS IS A CHECKSET, NOT A TALLY, and that is the whole reason it needs its own decisi` |
| `docs/audit-2026-09-02.md|cmd/gate/parity.go:265` | goinfer | `cfg := &gateConfig{Name: "parity", Decision: "checkset", TopLevelOnly: true, RCIsFailure` |
| `docs/audit-2026-09-02.md|cmd/gate/parity.go:274` | goinfer | `for _, c := range cfg.Cells {` |
| `docs/audit-2026-09-02.md|cmd/gate/parity.go:296` | goinfer | `// Safety net: any OTHER parity/gate-shaped test that skipped — a family the lists forgo` |
| `docs/audit-2026-09-02.md|cmd/gate/parity.go:364` | goinfer | `// THE FOURTH OUTCOME (B14). A gate failing with no confirmed prior result is asserting ` |
| `docs/audit-2026-09-02.md|cmd/gate/parity.go:518` | goinfer | `// This exists because the distinction cost five weeks. TestQwen3NextReal_oracle was rep` |
| `docs/audit-2026-09-02.md|cmd/gate/parity.go:527` | goinfer | `func whyNoResult(test string, cells []cell) string {` |
| `docs/audit-2026-09-02.md|cmd/gate/parity_test.go:283` | goinfer | `func TestRealckptCellCanReachEveryGate(t *testing.T) {` |
| `docs/audit-2026-09-02.md|cmd/gate/parity_test.go:319` | goinfer | `func TestParity_missingGateSaysWhichCause(t *testing.T) {` |
| `docs/audit-2026-09-02.md|constrain/constrain.go:198` | goinfer | `func (m *Masker) Process(generated []int, logits []float32) {` |
| `docs/audit-2026-09-02.md|constrain/constrain.go:211` | goinfer | `// StopWhenComplete: at a completion point, only EOS survives.` |
| `docs/audit-2026-09-02.md|constrain/json.go:84` | goinfer | `func (g *jsonGrammar) CanEnd() bool {` |
| `docs/audit-2026-09-02.md|constrain/json.go:96` | goinfer | `func (g *jsonGrammar) TryBytes(bs []byte) bool {` |
| `docs/audit-2026-09-02.md|constrain/reflect.go:10` | goinfer | `// GrammarFromStruct derives a JSON Schema from a Go struct (via its json tags)` |
| `docs/audit-2026-09-02.md|constrain/reflect.go:103` | goinfer | `case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64,` |
| `docs/audit-2026-09-02.md|constrain/reflect.go:63` | goinfer | `for f := range t.Fields() {` |
| `docs/audit-2026-09-02.md|constrain/schema.go:187` | goinfer | `propsRaw, _ := s["properties"].(map[string]any)` |
| `docs/audit-2026-09-02.md|constrain/schema.go:191` | goinfer | `// An object with no declared properties and no `additionalProperties:false` is the` |
| `docs/audit-2026-09-02.md|constrain/schema.go:298` | goinfer | `// encodeLiteral renders an enum/const value to the compact JSON bytes the model` |
| `docs/audit-2026-09-02.md|constrain/schema_grammar.go:80` | goinfer | `// enum literal has no closing delimiter, so it's done as soon as it can't extend).` |
| `docs/audit-2026-09-02.md|constrain/schema_grammar.go:81` | goinfer | `func (g *schemaGrammar) CanEnd() bool {` |
| `docs/audit-2026-09-02.md|constrain/schema_grammar.go:98` | goinfer | `func (g *schemaGrammar) TryBytes(bs []byte) bool {` |
| `docs/audit-2026-09-02.md|constrain/schema_test.go:180` | goinfer | `out := genConstrained(t, g, int64(i)+1, 15) // cap digits so ints fit int64` |
| `docs/audit-2026-09-02.md|constrain/tool_grammar.go:29` | goinfer | `if len(paramSchema) == 0 {` |
| `docs/audit-2026-09-02.md|constrain/tool_grammar.go:32` | goinfer | `name, _ := json.Marshal(toolName)` |
| `docs/audit-2026-09-02.md|cuda/backend.go:204` | goinfer | `hls := make([]hlayer, nLayers)` |
| `docs/audit-2026-09-02.md|cuda/backend.go:225` | goinfer | `hl.isDeltaNet = true` |
| `docs/audit-2026-09-02.md|cuda/backend.go:455` | goinfer | `anchor: func (b *cudaBackend) BuildResident(m *decoder.Model) (rf decoder.ResidentForwar` |
| `docs/audit-2026-09-02.md|cuda/backend.go:494` | goinfer | `ctxCap:      resolveCtxCap(m.ResidentContextRequest(), m.Config().MaxPositions),` |
| `docs/audit-2026-09-02.md|cuda/backend.go:53` | goinfer | `func layerFusable(qkvInt4, moe, guInt4 bool) bool {` |
| `docs/audit-2026-09-02.md|cuda/backend.go:554` | goinfer | `r.cacheSlots = topK` |
| `docs/audit-2026-09-02.md|cuda/backend.go:598` | goinfer | `// THESE MODULE AND PIPELINE HANDLES DO NOT SURVIVE DEVICE EXHAUSTION. Read this before` |
| `docs/audit-2026-09-02.md|cuda/backend.go:684` | goinfer | `// fix didn't force a glue.ptx regen. (glue.ptx and all production PTX except moe.ptx ar` |
| `docs/audit-2026-09-02.md|cuda/backend.go:700` | goinfer | `if pbmod, e3 := r.dev.CompileLibrary(prefillBatchedPTX); e3 == nil {` |
| `docs/audit-2026-09-02.md|cuda/backend.go:786` | goinfer | `r.splitkvAttn = os.Getenv("GOINFER_SPLITKV_ATTN") != "0"` |
| `docs/audit-2026-09-02.md|cuda/backend.go:983` | goinfer | `L.hd, L.nKV, L.rhalf, L.qDim, L.kvDim = 0, 0, 0, 0, 0` |
| `docs/audit-2026-09-02.md|cuda/blockspec_test.go:63` | goinfer | `maxNew := 96` |
| `docs/audit-2026-09-02.md|cuda/blockspec_test.go:91` | goinfer | `ch, gen := mc.Generate(context.Background(), prompt, len(got), decoder.SamplingParams{})` |
| `docs/audit-2026-09-02.md|cuda/doc.go:10` | goinfer | `//     dlopen libcuda.so.1 at runtime, so `CGO_ENABLED=0` and the single-static-` |
| `docs/audit-2026-09-02.md|cuda/doc.go:19` | goinfer | `// Build with `-tags cuda`. Until the box wires the real kernel + a gocudrv-backed` |
| `docs/audit-2026-09-02.md|cuda/drafter.go:101` | goinfer | `d.fc = r.upW(fcw)` |
| `docs/audit-2026-09-02.md|cuda/drafter.go:173` | goinfer | `if n > d.ctxCap {` |
| `docs/audit-2026-09-02.md|cuda/drafter.go:251` | goinfer | `need := d.ctxLen + n` |
| `docs/audit-2026-09-02.md|cuda/drafter.go:271` | goinfer | `if n > d.extCap {` |
| `docs/audit-2026-09-02.md|cuda/drafter.go:382` | goinfer | `if d.ctxLen+M > d.kvCap {` |
| `docs/audit-2026-09-02.md|cuda/drafter.go:447` | goinfer | `if e := d.r.launch(d.attnBlock, LaunchConfig{GridX: uint32(nH), GridY: uint32(M), GridZ:` |
| `docs/audit-2026-09-02.md|cuda/drafter.go:451` | goinfer | `gpu.ArgValue(int32(d.ctxLen)), gpu.ArgValue(d.r.attnScale),` |
| `docs/audit-2026-09-02.md|cuda/drafter.go:563` | goinfer | `if M > d.headCap {` |
| `docs/audit-2026-09-02.md|cuda/drafter_loop_test.go:267` | goinfer | `const maxNew = 96` |
| `docs/audit-2026-09-02.md|cuda/drafter_vs_off_test.go:145` | goinfer | `maxNew := 96` |
| `docs/audit-2026-09-02.md|cuda/foreign_context_test.go:52` | goinfer | `func foreignCUDAContexts() (out []foreignCtx, ok bool) {` |
| `docs/audit-2026-09-02.md|cuda/gptoss_cache_ab_test.go:43` | goinfer | `opts.MoECacheSlots = 2 // FEWER slots than experts, so slot != expert id and the` |
| `docs/audit-2026-09-02.md|cuda/gptoss_real20b_test.go:36` | goinfer | `// Skips until CUDA declares the two features, exactly as metal/gptoss_real_test.go does` |
| `docs/audit-2026-09-02.md|cuda/gptoss_real20b_test.go:44` | goinfer | `// modelPath, NOT a direct environment read: the asset registry owns GOINFER_GPTOSS_GGUF` |
| `docs/audit-2026-09-02.md|cuda/graphs_safe.go:109` | goinfer | `// admitGraphs applies the safe-gate: it is the ONLY place r.graphs is promoted from "re` |
| `docs/audit-2026-09-02.md|cuda/kernel_fma_lint_test.go:15` | goinfer | `// moe.cu is the audited, FROZEN MoE PTX — reviewed separately (editing it needs its own` |
| `docs/audit-2026-09-02.md|cuda/kernels.go:107` | goinfer | `// this box's NVRTC 12.9.86, not 12.6; only moe.ptx + the bench kernels are the audited ` |
| `docs/audit-2026-09-02.md|cuda/kernels.go:178` | goinfer | `func f32tof16(f float32) uint16 {` |
| `docs/audit-2026-09-02.md|cuda/prefill.go:125` | goinfer | `func (r *cudaResident) prefillStaticDecline() error {` |
| `docs/audit-2026-09-02.md|cuda/prefill.go:156` | goinfer | `func nonBatchableKind(Ly *cudaLayer) string {` |
| `docs/audit-2026-09-02.md|cuda/prefill.go:204` | goinfer | `if e := r.checkCap(startPos, M); e != nil {` |
| `docs/audit-2026-09-02.md|cuda/prefill.go:226` | goinfer | `var scratch []Buffer` |
| `docs/audit-2026-09-02.md|cuda/prefill.go:319` | goinfer | `if e := r.launch(r.bAttn, LaunchConfig{GridX: uint32(r.nH), GridY: uint32(M), GridZ: 1, ` |
| `docs/audit-2026-09-02.md|cuda/prefill.go:323` | goinfer | `gpu.ArgValue(Ly.window), gpu.ArgValue(int32(M)), Arg(cctxB), ArgNull()); e != nil {` |
| `docs/audit-2026-09-02.md|cuda/prefill.go:417` | goinfer | `if e := r.stream.Sync(); e != nil {` |
| `docs/audit-2026-09-02.md|cuda/prefill.go:431` | goinfer | `for m := first; m < M; m++ {` |
| `docs/audit-2026-09-02.md|cuda/prefill.go:484` | goinfer | `func (r *cudaResident) batchedHeadArgmax(xB, aqB, aScB Buffer, M int, out *[]int) error ` |
| `docs/audit-2026-09-02.md|cuda/prefill.go:493` | goinfer | `// ONE head GEMV for all M rows: the weights are read once instead of M times.` |
| `docs/audit-2026-09-02.md|cuda/resident.go:1235` | goinfer | `if r.prefillReady && r.dnet == nil {` |
| `docs/audit-2026-09-02.md|cuda/resident.go:1242` | goinfer | `if startPos == 0 {` |
| `docs/audit-2026-09-02.md|cuda/resident.go:125` | goinfer | `func splitkvThreshold(nH, hd int) int {` |
| `docs/audit-2026-09-02.md|cuda/resident.go:1290` | goinfer | `_ = r.do(func() error { return r.resetState() })` |
| `docs/audit-2026-09-02.md|cuda/resident.go:1385` | goinfer | `r.reqCh = nil` |
| `docs/audit-2026-09-02.md|cuda/resident.go:1514` | goinfer | `func (r *cudaResident) capVec(src Buffer, dst [][]float32, l, n int) {` |
| `docs/audit-2026-09-02.md|cuda/resident.go:2227` | goinfer | `func (r *cudaResident) launchToken(emb []float32, pos int, head bool) error {` |
| `docs/audit-2026-09-02.md|cuda/resident.go:2301` | goinfer | `if r.splitkvAttn && r.skScores != (Pipeline{}) && nWin >= r.splitkvMin(Ly.hd) {` |
| `docs/audit-2026-09-02.md|cuda/resident.go:2320` | goinfer | `if err := r.launch(r.bAttn, LaunchConfig{GridX: uint32(r.nH), GridY: 1, GridZ: 1, BlockX` |
| `docs/audit-2026-09-02.md|cuda/resident.go:2324` | goinfer | `anchor: func (r *cudaResident) launchToken(emb []float32, pos int, head bool) error {` |
| `docs/audit-2026-09-02.md|cuda/resident.go:2325` | goinfer | `if err := r.launch(r.fAttn, LaunchConfig{GridX: uint32(r.nH), GridY: 1, GridZ: 1, BlockX` |
| `docs/audit-2026-09-02.md|cuda/resident.go:2636` | goinfer | `func (r *cudaResident) layerTail(Ly *cudaLayer, l int, gC bool) error {` |
| `docs/audit-2026-09-02.md|cuda/resident.go:310` | goinfer | `// hidCap is the PRODUCTION hidden-state seam (P10 / docs/spec/08): the resident` |
| `docs/audit-2026-09-02.md|cuda/resident.go:547` | goinfer | `func (r *cudaResident) cacheWQ(h hostW) cudaWQ {` |
| `docs/audit-2026-09-02.md|cuda/resident.go:700` | goinfer | `if decline {` |
| `docs/audit-2026-09-02.md|cuda/resident.go:889` | goinfer | `// Synchronize — right for per-request uploads, wrong here: a MoE decode token loads ~12` |
| `docs/audit-2026-09-02.md|cuda/resident.go:936` | goinfer | `if e := r.stream.Sync(); e != nil {` |
| `docs/audit-2026-09-02.md|cuda/resident.go:948` | goinfer | `for j := 0; j < r.topK; j++ {` |
| `docs/audit-2026-09-02.md|cuda/theta_probe_test.go:49` | goinfer | `for _, mdl := range []string{` |
| `docs/audit-2026-09-02.md|decoder/a3_fanout_test.go:190` | goinfer | `func TestAttendF32Fanout_bitIdentical(t *testing.T) {` |
| `docs/audit-2026-09-02.md|decoder/a3_moe_exclusion_test.go:15` | goinfer | `// forwardn.go excludes MoE from --cpu-fast-attention unconditionally, on a stated` |
| `docs/audit-2026-09-02.md|decoder/api_tiers_test.go:66` | goinfer | `bare := name` |
| `docs/audit-2026-09-02.md|decoder/arch.go:536` | goinfer | `func (a *Architecture) finalizeRoPE() {` |
| `docs/audit-2026-09-02.md|decoder/assets.go:49` | goinfer | `if m := os.Getenv("GOINFER_MODELS"); m != "" {` |
| `docs/audit-2026-09-02.md|decoder/attention.go:11` | goinfer | `func addBias(x, b []float32) {` |
| `docs/audit-2026-09-02.md|decoder/attention.go:114` | goinfer | `// 4. Append this position's K/V, then attend over the stored history. Route` |
| `docs/audit-2026-09-02.md|decoder/attention.go:129` | goinfer | `ctx := cache.scr.ctx[:nH*hd]` |
| `docs/audit-2026-09-02.md|decoder/attention.go:156` | goinfer | `pool := scr.headWorkerPool(nH, 1, nKeys, hd)` |
| `docs/audit-2026-09-02.md|decoder/blockspec.go:140` | goinfer | `// LOSSLESS BY CONSTRUCTION: every emitted token is one the TARGET's own argmax produced` |
| `docs/audit-2026-09-02.md|decoder/blockspec.go:175` | goinfer | `eos := map[int]bool{}` |
| `docs/audit-2026-09-02.md|decoder/blockspec.go:201` | goinfer | `ids, err := host.PrefillLastNArgmax(embs, 0)` |
| `docs/audit-2026-09-02.md|decoder/blockspec.go:209` | goinfer | `anchor := ids[len(ids)-1]` |
| `docs/audit-2026-09-02.md|decoder/blockspec.go:221` | goinfer | `for opt.MaxTokens <= 0 \|\| len(out) < opt.MaxTokens {` |
| `docs/audit-2026-09-02.md|decoder/blockspec.go:269` | goinfer | `blockIn := make([][]float32, width)` |
| `docs/audit-2026-09-02.md|decoder/blockspec.go:274` | goinfer | `trunk, e := rd.DraftBlock(blockIn)` |
| `docs/audit-2026-09-02.md|decoder/blockspec.go:301` | goinfer | `burst := make([]int, 0, accepted+1)` |
| `docs/audit-2026-09-02.md|decoder/blockspec.go:305` | goinfer | `// TRUNCATE BEFORE EOS INSIDE THE BURST. A round commits several tokens at once, so a` |
| `docs/audit-2026-09-02.md|decoder/blockspec.go:352` | goinfer | `func (s *BlockSpec) GenerateStream(ctx context.Context, prompt []int, maxTokens int,` |
| `docs/audit-2026-09-02.md|decoder/blockspec.go:385` | goinfer | `stats.Emitted = len(toks)` |
| `docs/audit-2026-09-02.md|decoder/blockspec_cpu_test.go:87` | goinfer | `n := min(len(base), len(got))` |
| `docs/audit-2026-09-02.md|decoder/config.go:1048` | goinfer | `// resolveEOSIDs returns the ids that end generation: config.json's` |
| `docs/audit-2026-09-02.md|decoder/config.go:1090` | goinfer | `func loadConfig(fsys fs.FS, name string) (*Config, error) {` |
| `docs/audit-2026-09-02.md|decoder/config.go:143` | goinfer | `AttentionChunkSize     int     `json:"attention_chunk_size"`` |
| `docs/audit-2026-09-02.md|decoder/config.go:491` | goinfer | `func (c *Config) validateGptOss() error {` |
| `docs/audit-2026-09-02.md|decoder/config.go:645` | goinfer | `func (c *Config) normalizeQwen3NextLayerTypes() error {` |
| `docs/audit-2026-09-02.md|decoder/deepseek_real_test.go:210` | goinfer | `prompt := "The capital of France is"` |
| `docs/audit-2026-09-02.md|decoder/deltanet.go:21` | goinfer | `var deltaNetTiming = os.Getenv("GOINFER_DELTANET_TIMING") != ""` |
| `docs/audit-2026-09-02.md|decoder/deltanet.go:226` | goinfer | `S := st.s[headV*hk*hv : (headV+1)*hk*hv] // [hk, hv]` |
| `docs/audit-2026-09-02.md|decoder/deltanet.go:77` | goinfer | `// THE THREE DOMINANT PROJECTIONS ARE QUANTIZABLE (2026-08-19). They were []float32 —` |
| `docs/audit-2026-09-02.md|decoder/deltanet_chunked.go:10` | goinfer | `// matmuls). It is NOT yet wired into the forward — the qwen3_5_moe path is still` |
| `docs/audit-2026-09-02.md|decoder/deltanet_chunked.go:137` | goinfer | `anchor: func scanChunk(core [][]float32, S []float32, conv, gt, beta [][]float32,` |
| `docs/audit-2026-09-02.md|decoder/deltanet_chunked_test.go:36` | goinfer | `aLog[i] = -2 * rng.Float32() // A_log in [-2,0] → gt = exp(g) in (0,1), stable` |
| `docs/audit-2026-09-02.md|decoder/dflash.go:651` | goinfer | `scale := 1 / math.Sqrt(float64(hd))` |
| `docs/audit-2026-09-02.md|decoder/embed.go:38` | goinfer | `if a.gemma4 != nil \|\| a.qwen35 != nil \|\| a.granite != nil \|\| a.nemotron != nil \|\| a.mla ` |
| `docs/audit-2026-09-02.md|decoder/embed.go:46` | goinfer | `cache := m.NewCache(len(ids))` |
| `docs/audit-2026-09-02.md|decoder/embed.go:48` | goinfer | `for _, id := range ids {` |
| `docs/audit-2026-09-02.md|decoder/features.go:261` | goinfer | `// residentBackendMoECap is the router-kernel capacity of each backend whose MoE scorebo` |
| `docs/audit-2026-09-02.md|decoder/features.go:273` | goinfer | `"webgpu": {experts: 512, groups: 32}, // gpu/moe.go: MAXE 512, array<f32,512> score/sel ` |
| `docs/audit-2026-09-02.md|decoder/features.go:419` | goinfer | `//   FeatAttnSink  the learned per-head softmax sink, the clamped interleaved-SwiGLU` |
| `docs/audit-2026-09-02.md|decoder/features_test.go:423` | goinfer | `// The cap was raised 256 -> 512 (MOE_MAX_E / MAXE) so Kimi-K2's 384 is now ADMITTED on ` |
| `docs/audit-2026-09-02.md|decoder/forward_gemma4.go:216` | goinfer | `if lw.LayerScalar != 0 {` |
| `docs/audit-2026-09-02.md|decoder/forward_gemma4.go:301` | goinfer | `for kvh := range nKV {` |
| `docs/audit-2026-09-02.md|decoder/forward_gemma4_moe.go:67` | goinfer | `if routerCapture {` |
| `docs/audit-2026-09-02.md|decoder/forward_gptoss.go:130` | goinfer | `sinkLogit := float64(lw.AttnSinks[qh])` |
| `docs/audit-2026-09-02.md|decoder/forward_lfm2.go:111` | goinfer | `pos := cache.Pos()` |
| `docs/audit-2026-09-02.md|decoder/forward_lfm2.go:126` | goinfer | `mix = shortConvStep(n, lw.shortConv, *g, hidden, cache.conv[l])` |
| `docs/audit-2026-09-02.md|decoder/forward_lfm2.go:32` | goinfer | `bcx := matvec(w.inProj, 3*cd, hidden, n)` |
| `docs/audit-2026-09-02.md|decoder/forward_llama4.go:16` | goinfer | `// Chunked (local) attention on the RoPE layers reduces to full causal for sequences bel` |
| `docs/audit-2026-09-02.md|decoder/forward_llama4.go:89` | goinfer | `attendQuery(q, ctx, cache.scr.scoresBuf(nKeys), cache, layer, pos, true /*full causal*/,` |
| `docs/audit-2026-09-02.md|decoder/forwardn.go:102` | goinfer | `return K > 1 && m.w.Embed.Rows() != 0 && !a.NonGatedMLP && !a.LearnedPosEmbed && a.gemma` |
| `docs/audit-2026-09-02.md|decoder/forwardn.go:1025` | goinfer | `// position ([K][VocabSize]) — used by the speculative verifier. Bit-identical to` |
| `docs/audit-2026-09-02.md|decoder/forwardn.go:1054` | goinfer | `h, err := m.forwardLayersN(reqCtx, ids, cache, false)` |
| `docs/audit-2026-09-02.md|decoder/forwardn.go:1080` | goinfer | `if cache.lora != nil \|\| !m.canBatchN(len(prompt)) {` |
| `docs/audit-2026-09-02.md|decoder/forwardn.go:1093` | goinfer | `h, err := m.forwardLayersN(ctx, prompt, cache, cpuFastAttention())` |
| `docs/audit-2026-09-02.md|decoder/forwardn.go:118` | goinfer | `if a.granite != nil \|\| a.nemotron != nil \|\| a.qwen35 != nil {` |
| `docs/audit-2026-09-02.md|decoder/forwardn.go:146` | goinfer | `// reused across the K rows (aikit's column-blocked W8A8 kernel); attention stays` |
| `docs/audit-2026-09-02.md|decoder/forwardn.go:192` | goinfer | `norm := make([]float32, K*hidden)` |
| `docs/audit-2026-09-02.md|decoder/forwardn.go:203` | goinfer | `maxKeys := startPos + K` |
| `docs/audit-2026-09-02.md|decoder/forwardn.go:273` | goinfer | `if fastAttn && K < fastAttnMinPrompt {` |
| `docs/audit-2026-09-02.md|decoder/forwardn.go:298` | goinfer | `attnPool := newHeadWorkerPool(prefillAttnWorkers(K, maxKeys, hd, arch.maxHeads()), K, ma` |
| `docs/audit-2026-09-02.md|decoder/forwardn.go:309` | goinfer | `var ws linalg.Workspace` |
| `docs/audit-2026-09-02.md|decoder/forwardn.go:32` | goinfer | `// WHAT IT GIVES UP IS BIGGER THAN "prefill != decode", and the help text understated it` |
| `docs/audit-2026-09-02.md|decoder/forwardn.go:346` | goinfer | `if err := reqCtx.Err(); err != nil {` |
| `docs/audit-2026-09-02.md|decoder/forwardn.go:371` | goinfer | `matmul(be, &lw.QProj, norm, q, K)` |
| `docs/audit-2026-09-02.md|decoder/forwardn.go:392` | goinfer | `if arch.QKNorm {` |
| `docs/audit-2026-09-02.md|decoder/forwardn.go:396` | goinfer | `anchor: func (m *Model) runLayersFromEmbedN(reqCtx context.Context, h []float32, cache *` |
| `docs/audit-2026-09-02.md|decoder/forwardn.go:414` | goinfer | `if isLocal {` |
| `docs/audit-2026-09-02.md|decoder/forwardn.go:415` | goinfer | `base, nRows := cache.batchReadLocal(l, startPos, K, k, v, alk, alv)` |
| `docs/audit-2026-09-02.md|decoder/forwardn.go:44` | goinfer | `// TestSessionFastAttnDivergence pins the new behaviour; the equality is still gated, un` |
| `docs/audit-2026-09-02.md|decoder/forwardn.go:50` | goinfer | `// MoE IS *NOT* EXCLUDED, and that is deliberate: 66d0a05 removed the exclusion after me` |
| `docs/audit-2026-09-02.md|decoder/forwardn.go:508` | goinfer | `if moeExpertMajor() {` |
| `docs/audit-2026-09-02.md|decoder/forwardn.go:509` | goinfer | `emOut = make([]float32, K*hidden)` |
| `docs/audit-2026-09-02.md|decoder/forwardn.go:56` | goinfer | `func cpuFastAttention() bool { return os.Getenv("GOINFER_CPU_FAST_ATTENTION") != "0" }` |
| `docs/audit-2026-09-02.md|decoder/forwardn.go:614` | goinfer | `cache.advanceTo(startPos + K)` |
| `docs/audit-2026-09-02.md|decoder/forwardn.go:656` | goinfer | `nKeys := len(keys) / kvDim` |
| `docs/audit-2026-09-02.md|decoder/forwardn.go:683` | goinfer | `fusedOK := !useAcc64 && cache.treeMask == nil` |
| `docs/audit-2026-09-02.md|decoder/forwardn.go:705` | goinfer | `tile := attnRowTile(K, nKeys)` |
| `docs/audit-2026-09-02.md|decoder/forwardn.go:710` | goinfer | `copy(qh[i*hd:i*hd+hd], q[b:b+hd])` |
| `docs/audit-2026-09-02.md|decoder/forwardn.go:873` | goinfer | `gatherKV := func(ws *headWorkerScratch, kvh int) {` |
| `docs/audit-2026-09-02.md|decoder/forwardn.go:98` | goinfer | `// Gemma 4 (per-layer head_dim, KV-sharing, PLE), qwen3_5_moe (Gated DeltaNet),` |
| `docs/audit-2026-09-02.md|decoder/forwardn_test.go:160` | goinfer | `// argmax-exact (above) is the hard bar; cosine ≥ 0.99 is the GPU-path` |
| `docs/audit-2026-09-02.md|decoder/forwardn_test.go:97` | goinfer | `// TestForwardN_matchesSequential checks the batched multi-position forward` |
| `docs/audit-2026-09-02.md|decoder/fp8.go:124` | goinfer | `// Shape is checked against the ARCHITECTURE (in/out from the config), not just against` |
| `docs/audit-2026-09-02.md|decoder/fusedattn.go:101` | goinfer | `a0, a1 := max(lo[i], k0), min(hi[i], k1-1)` |
| `docs/audit-2026-09-02.md|decoder/fusedattn.go:45` | goinfer | `func fusedAttention() bool { return os.Getenv("GOINFER_FUSED_ATTENTION") != "0" }` |
| `docs/audit-2026-09-02.md|decoder/fusedattn.go:85` | goinfer | `anchor: func attendTileFused(` |
| `docs/audit-2026-09-02.md|decoder/gguf.go:1238` | goinfer | `// Refuse families whose per-layer state the .giw writer can't express (MLA / Mamba-2 / ` |
| `docs/audit-2026-09-02.md|decoder/gguf.go:1253` | goinfer | `if arch.gemma4 != nil {` |
| `docs/audit-2026-09-02.md|decoder/gguf.go:1530` | goinfer | `if err := sink.writeHeadGlobals(w, id); err != nil {` |
| `docs/audit-2026-09-02.md|decoder/gguf.go:1755` | goinfer | `// expert weights are MXFP4 in the real checkpoint — stackedExperts routes them` |
| `docs/audit-2026-09-02.md|decoder/gguf.go:1851` | goinfer | `if err := parallelLayers(arch.NumLayers, loadGptOss); err != nil {` |
| `docs/audit-2026-09-02.md|decoder/gguf.go:27` | goinfer | `// Architectures: llama, qwen2, qwen3, gemma3, mellum. Quant types: F32/F16,` |
| `docs/audit-2026-09-02.md|decoder/gguf.go:41` | goinfer | `defer func() {` |
| `docs/audit-2026-09-02.md|decoder/gguf.go:483` | goinfer | `numLayers := u("block_count")` |
| `docs/audit-2026-09-02.md|decoder/gguf.go:556` | goinfer | `if k := u("leading_dense_block_count"); k > 0 {` |
| `docs/audit-2026-09-02.md|decoder/gptoss_real_test.go:61` | goinfer | `prompt := "The capital of France is"` |
| `docs/audit-2026-09-02.md|decoder/gptoss_safetensors.go:112` | goinfer | `{&l.QProj, &l.QBias, "q_proj", qDim, hidden},` |
| `docs/audit-2026-09-02.md|decoder/gptoss_safetensors.go:97` | goinfer | `for i := range arch.NumLayers {` |
| `docs/audit-2026-09-02.md|decoder/gptq.go:41` | goinfer | `func parseQuantConfig(raw json.RawMessage) (*quantConfig, error) {` |
| `docs/audit-2026-09-02.md|decoder/int4f16scales.go:10` | goinfer | `// GOINFER_INT4_F16_SCALES=1 is a DIAGNOSTIC (default-off, one env read per weight at LO` |
| `docs/audit-2026-09-02.md|decoder/int4f16scales.go:41` | goinfer | `func f32ToF16bits(f float32) uint16 {` |
| `docs/audit-2026-09-02.md|decoder/kvcache.go:235` | goinfer | `func (r *ring) truncate(p int) bool {` |
| `docs/audit-2026-09-02.md|decoder/kvcache.go:239` | goinfer | `exact := r.count <= r.w \|\| p >= r.count-1` |
| `docs/audit-2026-09-02.md|decoder/kvcache.go:389` | goinfer | `func (c *KVCache) resetRecurrent() {` |
| `docs/audit-2026-09-02.md|decoder/kvcache.go:439` | goinfer | `if c.mamba != nil \|\| c.delta != nil {` |
| `docs/audit-2026-09-02.md|decoder/kvcache.go:452` | goinfer | `for l := range c.numLayers {` |
| `docs/audit-2026-09-02.md|decoder/kvcache.go:497` | goinfer | `func (c *KVCache) batchReadLocal(layer, startPos, K int, newK, newV, dstK, dstV []float3` |
| `docs/audit-2026-09-02.md|decoder/kvcache.go:503` | goinfer | `base = max(startPos-r.w+1, 0)` |
| `docs/audit-2026-09-02.md|decoder/kvcache_recurrent_test.go:13` | goinfer | `c.mamba = []*mamba2State{{ssm: []float32{1, 2, 3}, convWin: [][]float32{{9}}}}` |
| `docs/audit-2026-09-02.md|decoder/kvsnapshot.go:210` | goinfer | `if numLayers < 0 \|\| numLayers > maxSerializedLayers \|\| kvDim < 0 \|\| kvDim > 1<<24 \|\|` |
| `docs/audit-2026-09-02.md|decoder/kvsnapshot.go:214` | goinfer | `if perPos := numLayers * kvDim; perPos > 0 && int64(pos) > int64(len(data))/int64(perPos` |
| `docs/audit-2026-09-02.md|decoder/kvsnapshot.go:250` | goinfer | `if st != kvDim \|\| rr.count < 0 \|\| nLive > rr.w \|\| rr.count < nLive {` |
| `docs/audit-2026-09-02.md|decoder/kvsnapshot.go:283` | goinfer | `} else if quant == kvI8 {` |
| `docs/audit-2026-09-02.md|decoder/kvsnapshot.go:67` | goinfer | `if c.delta != nil \|\| len(c.mamba) > 0 \|\| len(c.mlaLatent) > 0 {` |
| `docs/audit-2026-09-02.md|decoder/layerpaging.go:106` | goinfer | `const ahead = 1` |
| `docs/audit-2026-09-02.md|decoder/layerpaging.go:63` | goinfer | `if a := w.arch; a.gemma4 != nil \|\| a.qwen35 != nil \|\| a.granite != nil \|\| a.nemotron != ` |
| `docs/audit-2026-09-02.md|decoder/lfm2_test.go:178` | goinfer | `for _, id := range g.PromptIDs {` |
| `docs/audit-2026-09-02.md|decoder/llama4_real_test.go:52` | goinfer | `prompt := "The capital of France is"` |
| `docs/audit-2026-09-02.md|decoder/longprompt_golden_test.go:84` | goinfer | `os.Unsetenv("GOINFER_CPU_FAST_ATTENTION")` |
| `docs/audit-2026-09-02.md|decoder/mamba2.go:44` | goinfer | `inProj  []float32 // [projDim, hidden]` |
| `docs/audit-2026-09-02.md|decoder/mamba2.go:76` | goinfer | `if ssmQ8CPU { // confirmation seam: match the resident W8A8 — int8 weights AND int8 acti` |
| `docs/audit-2026-09-02.md|decoder/mamba2_chunked.go:91` | goinfer | `Pc := make([]float32, L)` |
| `docs/audit-2026-09-02.md|decoder/mlp.go:464` | goinfer | `func moeExpertMajor() bool { return os.Getenv("GOINFER_MOE_EXPERT_MAJOR") != "0" }` |
| `docs/audit-2026-09-02.md|decoder/mlp.go:484` | goinfer | `func moeMLPBatch(rows []float32, n int, lw *LayerWeights, arch *Architecture, be Backend` |
| `docs/audit-2026-09-02.md|decoder/mlp.go:504` | goinfer | `logits := make([]float32, n*nE)` |
| `docs/audit-2026-09-02.md|decoder/model.go:1001` | goinfer | `anchor: func (m *Model) generateInto(ctx context.Context, out chan<- int, g *Generation,` |
| `docs/audit-2026-09-02.md|decoder/model.go:1080` | goinfer | `if sp.Logprobs {` |
| `docs/audit-2026-09-02.md|decoder/model.go:1112` | goinfer | `fastNext, err = greedyRF.ForwardArgmax(emb, gpuPos)` |
| `docs/audit-2026-09-02.md|decoder/model.go:1142` | goinfer | `func (m *Model) isStop(id int, sp SamplingParams) bool {` |
| `docs/audit-2026-09-02.md|decoder/model.go:216` | goinfer | `weightsBlob, _, gerr := giw.Read(data)` |
| `docs/audit-2026-09-02.md|decoder/model.go:719` | goinfer | `// mixers whose "residual after layer l" needs deciding rather than assuming, mla and ll` |
| `docs/audit-2026-09-02.md|decoder/model.go:724` | goinfer | `if a.granite != nil \|\| a.nemotron != nil \|\| a.mla != nil \|\| a.llama4 != nil {` |
| `docs/audit-2026-09-02.md|decoder/model.go:750` | goinfer | `if a.gemma4 != nil \|\| a.qwen35 != nil \|\| a.granite != nil \|\| a.nemotron != nil \|\| a.mla ` |
| `docs/audit-2026-09-02.md|decoder/model.go:821` | goinfer | `anchor: func (m *Model) Generate(ctx context.Context, prompt []int, maxTokens int, sp Sa` |
| `docs/audit-2026-09-02.md|decoder/model.go:864` | goinfer | `if lg, perr := pf.PrefillLast(embs, 0); perr == nil {` |
| `docs/audit-2026-09-02.md|decoder/model.go:989` | goinfer | `optFwd := useGPU && !fastGreedy && m.optFwdEligible(sp) && os.Getenv("GOINFER_NO_OPTFWD"` |
| `docs/audit-2026-09-02.md|decoder/moe_expert_batch_test.go:220` | goinfer | `func loadMoEBitIdentModel(t *testing.T) (*Model, error) {` |
| `docs/audit-2026-09-02.md|decoder/moecap_kernel_pin_test.go:37` | goinfer | `// webgpu: array<f32, 256> score / array<f32, 32> gscore` |
| `docs/audit-2026-09-02.md|decoder/moepaging.go:61` | goinfer | `anchor: func newExpertPager(w *Weights, mapping []byte, budget int64) *expertPager {` |
| `docs/audit-2026-09-02.md|decoder/mtp.go:35` | goinfer | `anchor: type MTPHead struct {` |
| `docs/audit-2026-09-02.md|decoder/prefillattnpool_test.go:83` | goinfer | `os.Unsetenv("GOINFER_PREFILL_ATTN_WORKERS")` |
| `docs/audit-2026-09-02.md|decoder/registry.go:1198` | goinfer | `QKNorm:          true, // per-head RMSNorm over head_dim — see the note above` |
| `docs/audit-2026-09-02.md|decoder/registry.go:1446` | goinfer | `// DeepSeek's YaRN attention_factor (the cos/sin mscale) is NOT the generic` |
| `docs/audit-2026-09-02.md|decoder/registry.go:1526` | goinfer | `// Plain qk_head_dim^-0.5. ⚠️ Phase 3: the real V2-Lite/V3 fold YaRN's` |
| `docs/audit-2026-09-02.md|decoder/registry.go:1641` | goinfer | `scaling, err := parseRopeScaling(cfg.RopeScaling)` |
| `docs/audit-2026-09-02.md|decoder/registry.go:1697` | goinfer | `var base float64` |
| `docs/audit-2026-09-02.md|decoder/registry.go:1727` | goinfer | `useRope := make([]bool, cfg.NumLayers)` |
| `docs/audit-2026-09-02.md|decoder/registry.go:1792` | goinfer | `llama4: &llama4Params{` |
| `docs/audit-2026-09-02.md|decoder/registry.go:183` | goinfer | `AttnScale:         math.Pow(cfg.QueryPreAttnScalar, -0.5),` |
| `docs/audit-2026-09-02.md|decoder/registry.go:320` | goinfer | `// backfillFlatRope fills the flat rope_theta / rope_scaling fields from transformers >=` |
| `docs/audit-2026-09-02.md|decoder/registry.go:95` | goinfer | `func (a *Architecture) validateResolved() error {` |
| `docs/audit-2026-09-02.md|decoder/residency.go:129` | goinfer | `if m.w.arch.nemotron != nil && os.Getenv("GOINFER_SSM_RESIDENT") == "" {` |
| `docs/audit-2026-09-02.md|decoder/residency.go:225` | goinfer | `// Nemotron 3 Nano adds a FOURTH block kind (MoE FFN, arch.MoE != nil — plain Nemotron-H` |
| `docs/audit-2026-09-02.md|decoder/residency.go:60` | goinfer | `type ResidentGreedy interface {` |
| `docs/audit-2026-09-02.md|decoder/residency.go:726` | goinfer | `// staged/CPU path reports canBatchN, which excludes the families with their own sequent` |
| `docs/audit-2026-09-02.md|decoder/residency.go:730` | goinfer | `if !m.canBatchN(2) {` |
| `docs/audit-2026-09-02.md|decoder/residency.go:831` | goinfer | `out := make([]float32, len(lw.Experts)*2*inter)` |
| `docs/audit-2026-09-02.md|decoder/residency.go:854` | goinfer | `out := make([]float32, len(lw.Experts)*hidden)` |
| `docs/audit-2026-09-02.md|decoder/rmsnorm.go:32` | goinfer | `row[i] = (v * inv) * weight[i]` |
| `docs/audit-2026-09-02.md|decoder/rope.go:168` | goinfer | `if img >= len(grids) {` |
| `docs/audit-2026-09-02.md|decoder/rope.go:29` | goinfer | `half := len(invFreq) // == rotaryDim/2` |
| `docs/audit-2026-09-02.md|decoder/rope.go:31` | goinfer | `for d := range half {` |
| `docs/audit-2026-09-02.md|decoder/routercapture.go:23` | goinfer | `var routerCaptureBuf [][]int` |
| `docs/audit-2026-09-02.md|decoder/sampler.go:101` | goinfer | `//   - Temperature > 0: softmax at that temperature, optionally restricted to` |
| `docs/audit-2026-09-02.md|decoder/sampler.go:134` | goinfer | `// `top_p` / `min_p` at any value are safe alongside it: both cuts clamp at ≥1 retained ` |
| `docs/audit-2026-09-02.md|decoder/sampler.go:264` | goinfer | `func (s *Sampler) applyPenaltiesOver(logits []float32, window []int) {` |
| `docs/audit-2026-09-02.md|decoder/sampler.go:293` | goinfer | `func computeLogprobs(logits []float32, chosen int, temperature float64, topN int) (float` |
| `docs/audit-2026-09-02.md|decoder/sampler.go:320` | goinfer | `func (s *Sampler) drawFiltered(ips []indexedProb) int {` |
| `docs/audit-2026-09-02.md|decoder/sampler.go:342` | goinfer | `return len(probs) - 1` |
| `docs/audit-2026-09-02.md|decoder/sampler.go:437` | goinfer | `case minP > 0:` |
| `docs/audit-2026-09-02.md|decoder/sampler.go:496` | goinfer | `if topPActive {` |
| `docs/audit-2026-09-02.md|decoder/sampler.go:539` | goinfer | `func topKByLogit(logits []float32, k int) []int {` |
| `docs/audit-2026-09-02.md|decoder/sampler_chunked.go:110` | goinfer | `func forEachChunk(n int, fn func(c, lo, hi int)) {` |
| `docs/audit-2026-09-02.md|decoder/sampler_chunked.go:159` | goinfer | `return hi - 1 // rounding hair inside the chunk` |
| `docs/audit-2026-09-02.md|decoder/sampler_selection_test.go:296` | goinfer | `// RE-BOUNDED after P2b (2026-08-09). This gate compares temp+top_p against temp-only, a` |
| `docs/audit-2026-09-02.md|decoder/scratch.go:192` | goinfer | `func attnRowTile(K, nKeys int) int {` |
| `docs/audit-2026-09-02.md|decoder/scratch.go:222` | goinfer | `if v := os.Getenv("GOINFER_PREFILL_ATTN_WORKERS"); v != "" {` |
| `docs/audit-2026-09-02.md|decoder/scratch.go:230` | goinfer | `// Per slot, in bytes, at the TILED size (G20): scores (tile*nKeys) + kh + vt` |
| `docs/audit-2026-09-02.md|decoder/scratch.go:260` | goinfer | `for i := range s.headPool[:n] {` |
| `docs/audit-2026-09-02.md|decoder/scratch.go:294` | goinfer | `func newHeadWorkerPool(n, K, nKeys, hd int) []headWorkerScratch {` |
| `docs/audit-2026-09-02.md|decoder/scratch.go:304` | goinfer | `t := attnRowTile(K, nKeys)` |
| `docs/audit-2026-09-02.md|decoder/scratch.go:311` | goinfer | `kh:     make([]float32, nKeys*hd),` |
| `docs/audit-2026-09-02.md|decoder/scratch.go:84` | goinfer | `ws := &linalg.Workspace{}` |
| `docs/audit-2026-09-02.md|decoder/serialize.go:112` | goinfer | `anchor: func canSerialize(a *Architecture) *SerializeError {` |
| `docs/audit-2026-09-02.md|decoder/serialize.go:1174` | goinfer | `anchor: func (r *giwReader) weightMat() linalg.WeightMat {` |
| `docs/audit-2026-09-02.md|decoder/serialize.go:1201` | goinfer | `func (r *giwReader) layer(l *LayerWeights) {` |
| `docs/audit-2026-09-02.md|decoder/serialize.go:24` | goinfer | `// Discipline mirrors ken's index_serialize.go: magic + version + a config/quant` |
| `docs/audit-2026-09-02.md|decoder/serialize.go:264` | goinfer | `// LoadSerializedWeights reconstructs a *Weights from a SerializeWeights blob` |
| `docs/audit-2026-09-02.md|decoder/serialize.go:293` | goinfer | `// CRC: verify the whole payload (everything before the trailing crc word)` |
| `docs/audit-2026-09-02.md|decoder/serialize.go:303` | goinfer | `var cfg Config` |
| `docs/audit-2026-09-02.md|decoder/serialize.go:340` | goinfer | `n := int(r.u32())` |
| `docs/audit-2026-09-02.md|decoder/serialize.go:375` | goinfer | `func validateShapes(w *Weights, arch *Architecture) *SerializeError {` |
| `docs/audit-2026-09-02.md|decoder/serialize.go:860` | goinfer | `func (w *giwWriter) layer(l *LayerWeights) {` |
| `docs/audit-2026-09-02.md|decoder/serialize.go:98` | goinfer | `func canSerialize(a *Architecture) *SerializeError {` |
| `docs/audit-2026-09-02.md|decoder/serialize_census_test.go:58` | goinfer | `// "passed" while the new code went unexercised.` |
| `docs/audit-2026-09-02.md|decoder/session.go:73` | goinfer | `func (s *Session) rewindForReuse(prompt []int) int {` |
| `docs/audit-2026-09-02.md|decoder/session.go:86` | goinfer | `func (s *Session) reconcile(seq []int) {` |
| `docs/audit-2026-09-02.md|decoder/session.go:98` | goinfer | `if rolledBack && (s.cache.mamba != nil \|\| s.cache.delta != nil) {` |
| `docs/audit-2026-09-02.md|decoder/session_test.go:213` | goinfer | `bad := append([]byte(nil), blob...)` |
| `docs/audit-2026-09-02.md|decoder/spec_adaptive.go:163` | goinfer | `case "cuda":` |
| `docs/audit-2026-09-02.md|decoder/spec_adaptive.go:82` | goinfer | `if a.Theta >= 1 {` |
| `docs/audit-2026-09-02.md|decoder/spec_eagle.go:254` | goinfer | `logitsN, feats, err := captureN(prompt)` |
| `docs/audit-2026-09-02.md|decoder/spec_eagle.go:82` | goinfer | `logitsN, feats, err := captureN(prompt)` |
| `docs/audit-2026-09-02.md|decoder/spec_hitrate_probe_test.go:37` | goinfer | `const giw = "/Users/francistownsend-merino/models/gemma4-26b-int4.giw"` |
| `docs/audit-2026-09-02.md|decoder/spec_ngram.go:334` | goinfer | `lookupCtx := append(slices.Clone(hist), cur)` |
| `docs/audit-2026-09-02.md|decoder/spec_ngram.go:381` | goinfer | `p := dist(logitsN[i], ph)` |
| `docs/audit-2026-09-02.md|decoder/spec_ngram_test.go:83` | goinfer | `recurrent := map[string]*Model{` |
| `docs/audit-2026-09-02.md|decoder/spec_optfwd.go:30` | goinfer | `// GOINFER_OPTFWD_MAX_TEMP overrides it, for MEASUREMENT rather than tuning: moving this` |
| `docs/audit-2026-09-02.md|decoder/spec_sample.go:111` | goinfer | `for i, v := range slices.Backward(p) { // float-rounding guard: last token with mass` |
| `docs/audit-2026-09-02.md|decoder/spec_sample.go:33` | goinfer | `return softmaxStable(logits, s.p.Temperature) // drawFull draws from this directly` |
| `docs/audit-2026-09-02.md|decoder/spec_sample.go:76` | goinfer | `func (s *Sampler) specStep(p []float64, x int) (int, bool) {` |
| `docs/audit-2026-09-02.md|decoder/weightbytes.go:47` | goinfer | `func (m *Model) ResidentWeightBytes() int64 {` |
| `docs/audit-2026-09-02.md|decoder/weightmat.go:305` | goinfer | `var w4a8SplitHalfRepackEnabled = os.Getenv("GOINFER_W4A8_SPLITHALF") != ""` |
| `docs/audit-2026-09-02.md|decoder/weights.go:1071` | goinfer | `func loadFusedExperts(st *embed.SafetensorsFile, gateUpName, downName string, nExpert, i` |
| `docs/audit-2026-09-02.md|decoder/weights.go:114` | goinfer | `gemma4moe *gemma4MoEWeights` |
| `docs/audit-2026-09-02.md|decoder/weights.go:360` | goinfer | `// P13: release the SOURCE mapping now when nothing can alias it, instead of holding it ` |
| `docs/audit-2026-09-02.md|decoder/weights.go:694` | goinfer | `if arch.lfm2 != nil && arch.isConvLayer(i) {` |
| `docs/audit-2026-09-02.md|decoder/weights.go:90` | goinfer | `delta *deltaNetWeights` |
| `docs/audit-2026-09-02.md|decoder/weights.go:96` | goinfer | `shortConv *shortConvWeights` |
| `docs/audit-2026-09-02.md|demo/agent/agent/agent.go:330` | goinfer | `ids, err := s.tk.Encode(s.buildPrompt(visionSystem, turns), s.tmpl == nil)` |
| `docs/audit-2026-09-02.md|demo/agent/agent/agent.go:397` | goinfer | `flush := func(final bool) {` |
| `docs/audit-2026-09-02.md|demo/agent/agent/agent.go:462` | goinfer | `return constrain.NewMasker(g, constrain.TokenBytes(s.vocab, s.tk.TokenText), eos).StopWh` |
| `docs/audit-2026-09-02.md|gpu/attention.go:793` | goinfer | `func (c *Context) ensureAttnWide() error {` |
| `docs/audit-2026-09-02.md|gpu/decode_staged_prize_test.go:23` | goinfer | `func TestDecodeStaged_prize(t *testing.T) {` |
| `docs/audit-2026-09-02.md|gpu/decoderunner.go:1126` | goinfer | `fq, fs := rmsQuant(r.xd, m.finalNorm, hidden)` |
| `docs/audit-2026-09-02.md|gpu/decoderunner.go:1176` | goinfer | `func (r *DecodeRunner) ReadMambaCap(projN, convN, dInner int) (proj, conv, y, gated []fl` |
| `docs/audit-2026-09-02.md|gpu/decoderunner.go:1231` | goinfer | `enc.CopyBufferToBuffer(r.lastLogits, 0, r.stag, 0, uint64(r.vocab*4))` |
| `docs/audit-2026-09-02.md|gpu/decoderunner.go:294` | goinfer | `ssmStopLayer := -1 // GOINFER_SSM_STOP_LAYER debug (resident SSM bring-up): truncate the` |
| `docs/audit-2026-09-02.md|gpu/decoderunner.go:303` | goinfer | `w8a16 := os.Getenv("GOINFER_SSM_W8A16") != ""` |
| `docs/audit-2026-09-02.md|gpu/decoderunner.go:467` | goinfer | `gemv := func(aq, as *wgpu.Buffer, w decodeWeight) *wgpu.Buffer {` |
| `docs/audit-2026-09-02.md|gpu/decoderunner.go:745` | goinfer | `anchor: func (c *Context) newDecodeRunner(m runModel, hidden, nH, nKV, hd, inter, start ` |
| `docs/audit-2026-09-02.md|gpu/decodetoken_batched.go:11` | goinfer | `// DecodeTokenFusedBatched is the Stage-B (docs/spec/07) batched verify forward: it` |
| `docs/audit-2026-09-02.md|gpu/deltanet.go:13` | goinfer | `// This is the mixer that makes every DeltaNet hybrid CPU-only on every backend today: 4` |
| `docs/audit-2026-09-02.md|gpu/doc.go:19` | goinfer | `// Status: FOUNDATION cut. A single `dst = a·bᵀ` GEMM offloaded to the GPU` |
| `docs/audit-2026-09-02.md|gpu/doc.go:6` | goinfer | `// wgpu-native Rust library) is allowed to appear. Every file except this doc` |
| `docs/audit-2026-09-02.md|gpu/gemv_w4a8.go:119` | goinfer | `func f32to16(f float32) uint16 {` |
| `docs/audit-2026-09-02.md|gpu/gemv_w4a8.go:25` | goinfer | `@group(0) @binding(0) var<storage, read>       aq:      array<vec4<u32>>;  // [kp/16] in` |
| `docs/audit-2026-09-02.md|gpu/gemv_w8a16.go:21` | goinfer | `@group(0) @binding(0) var<storage, read>       act:     array<f32>;        // [kp] f32 a` |
| `docs/audit-2026-09-02.md|gpu/gpu.go:50` | goinfer | `// not. Not safe for concurrent use by multiple goroutines; wrap in your` |
| `docs/audit-2026-09-02.md|gpu/kv_longctx_test.go:24` | goinfer | `func TestKVLongCtx(t *testing.T) {` |
| `docs/audit-2026-09-02.md|gpu/mamba_f16.go:21` | goinfer | `func f32ToF16(f float32) uint16 {` |
| `docs/audit-2026-09-02.md|gpu/moe.go:25` | goinfer | `const MAXE: u32 = 512u;` |
| `docs/audit-2026-09-02.md|gpu/moe_w4a8.go:146` | goinfer | `func (c *Context) UploadStackedExpertsInt4Packed(q4 [][]byte, scales [][]float32, nE, N,` |
| `docs/audit-2026-09-02.md|gpu/moe_w4a8_expert_test.go:37` | goinfer | `stack, err := ctx.UploadStackedExpertsInt4(nib, sc, nE, N, K)` |
| `docs/audit-2026-09-02.md|gpu/qwen35_resident_parity_test.go:26` | goinfer | `if os.Getenv("GOINFER_DNET_PARITY") == "" {` |
| `docs/audit-2026-09-02.md|gpu/residency.go:104` | goinfer | `_, _, _, _, _, _, _, _, _, granOK := m.GraniteResidentParams()` |
| `docs/audit-2026-09-02.md|gpu/residency.go:126` | goinfer | `if nE, _, _, _, _, _, _, _, nGroup, _, moeOK := m.MoEResidentParams(); moeOK && (nE > 25` |
| `docs/audit-2026-09-02.md|gpu/residency.go:138` | goinfer | `kvF16 := m.KVCacheF16()` |
| `docs/audit-2026-09-02.md|gpu/residency.go:140` | goinfer | `ctxCap := 16384` |
| `docs/audit-2026-09-02.md|gpu/residency.go:382` | goinfer | `if K%w4a8GroupSize == 0 && !int4SlowPath {` |
| `docs/audit-2026-09-02.md|gpu/residency.go:628` | goinfer | `if os.Getenv("GOINFER_SSM_F16MAMBA") != "" { // f16 (no quality gain; kept for experimen` |
| `docs/audit-2026-09-02.md|gpu/residency.go:743` | goinfer | `anchor: func (b *webgpuBackend) BuildResident(m *decoder.Model) (decoder.ResidentForward` |
| `docs/audit-2026-09-02.md|gpu/residency.go:892` | goinfer | `func (rd *residentDecoder) Forward(embedding []float32, pos int) ([]float32, error) {` |
| `docs/audit-2026-09-02.md|gpu/residency.go:906` | goinfer | `func (rd *residentDecoder) ForwardN(embeddings [][]float32, startPos int) ([][]float32, ` |
| `docs/audit-2026-09-02.md|gpu/residency.go:997` | goinfer | `func (rd *residentDecoder) Reset() {` |
| `docs/audit-2026-09-02.md|gpu/residency_c01_reset_test.go:24` | goinfer | `requireHeavyModel(t)` |
| `docs/audit-2026-09-02.md|gpu/resident_pack_bench_test.go:18` | goinfer | `func TestResidentPackCost(t *testing.T) {` |
| `docs/audit-2026-09-02.md|gpu/testhooks_gen.go:1` | goinfer | `//go:build goinfer_testhooks` |
| `docs/audit-2026-09-02.md|gpu/vision.go:52` | goinfer | `let mean = smean[0] / f32(p.h);` |
| `docs/audit-2026-09-02.md|gpu/vision_encoder.go:202` | goinfer | `for head := range nH {` |
| `docs/audit-2026-09-02.md|internal/chatapp/main.go:189` | goinfer | `case strings.HasSuffix(path, ".giw"):` |
| `docs/audit-2026-09-02.md|internal/chatapp/main.go:192` | goinfer | `if tk, err = tokenizer.LoadGGUFBytes(raw); err != nil {` |
| `docs/audit-2026-09-02.md|internal/chatapp/main.go:332` | goinfer | `flush := func(final bool) {` |
| `docs/audit-2026-09-02.md|internal/chatapp/main.go:485` | goinfer | `m := constrain.NewMasker(g, constrain.TokenBytes(s.vocab, s.tk.TokenText), eos).StopWhen` |
| `docs/audit-2026-09-02.md|internal/chatapp/prequant.go:45` | goinfer | `tk, err := tokenizer.LoadGGUFBytes(tokGGUF)` |
| `docs/audit-2026-09-02.md|internal/gemmaapp/main.go:113` | goinfer | `// 4) Generate + stream. Decode the whole running sequence (prompt +` |
| `docs/audit-2026-09-02.md|internal/gemmaapp/main.go:131` | goinfer | `flush := func(final bool) error {` |
| `docs/audit-2026-09-02.md|internal/gemmaapp/main.go:175` | goinfer | `m := constrain.NewMasker(constrain.JSON(), constrain.TokenBytes(vocab, tk.TokenText), eo` |
| `docs/audit-2026-09-02.md|internal/giw/bundle.go:44` | goinfer | `func WriteStream(f *os.File, tok []byte, writeWeights func(io.Writer) (int64, error)) er` |
| `docs/audit-2026-09-02.md|internal/prequant/prequant.go:101` | goinfer | `func transcodeDir(ctx context.Context, dir, out, quant string, embedInt4, row4 bool) err` |
| `docs/audit-2026-09-02.md|internal/prequant/prequant.go:154` | goinfer | `func EnsureCachedGIW(ctx context.Context, ggufPath, quant string) (string, error) {` |
| `docs/audit-2026-09-02.md|internal/prequant/prequant.go:180` | goinfer | `func cacheFresh(cache, src string) bool {` |
| `docs/audit-2026-09-02.md|internal/prequant/prequant.go:193` | goinfer | `// selfCheck verifies a freshly written bundle loads through the real mmap path` |
| `docs/audit-2026-09-02.md|internal/prequant/prequant.go:70` | goinfer | `f, err := os.Create(out)` |
| `docs/audit-2026-09-02.md|internal/serveapp/admin.go:64` | goinfer | `anchor: func (s *server) handleAdminLoad(w http.ResponseWriter, r *http.Request) {` |
| `docs/audit-2026-09-02.md|internal/serveapp/admin.go:71` | goinfer | `lm, err := loadDecoder(r.Context(), modelSpec{name: name, path: req.Path}, c)` |
| `docs/audit-2026-09-02.md|internal/serveapp/admin_test.go:80` | goinfer | `// old contract was "busy → 409"; unload now DRAINS instead of refusing — the in-flight-` |
| `docs/audit-2026-09-02.md|internal/serveapp/anthropic.go:127` | goinfer | `writeAnthropicErr(w, http.StatusBadRequest, "invalid_request_error", "invalid request bo` |
| `docs/audit-2026-09-02.md|internal/serveapp/anthropic.go:354` | goinfer | `func anthropicForcedTool(mode, name string, tools []chat.Tool) *chat.Tool {` |
| `docs/audit-2026-09-02.md|internal/serveapp/anthropic.go:417` | goinfer | `func (s *server) handleMessages(w http.ResponseWriter, r *http.Request) {` |
| `docs/audit-2026-09-02.md|internal/serveapp/anthropic.go:433` | goinfer | `// O(n) tokenize. The OpenAI routes got this guard; /v1/messages did not, so a body unde` |
| `docs/audit-2026-09-02.md|internal/serveapp/anthropic.go:438` | goinfer | `if err := lm.promptTooLargeForContext(anthropicInputBytes(&req)); err != nil {` |
| `docs/audit-2026-09-02.md|internal/serveapp/anthropic.go:561` | goinfer | `func (s *server) serveCountTokensWith(w http.ResponseWriter, req anthropicReq, lm *loade` |
| `docs/audit-2026-09-02.md|internal/serveapp/anthropic_stream.go:94` | goinfer | `func (s *server) streamMessagesTools(w http.ResponseWriter, r *http.Request, f http.Flus` |
| `docs/audit-2026-09-02.md|internal/serveapp/cpufastattn_test.go:12` | goinfer | `//  1. It is OFF unless asked for. A speed flag that turns itself on is how a user` |
| `docs/audit-2026-09-02.md|internal/serveapp/decoder_embedder.go:170` | goinfer | `func (e *decoderEmbedder) encodeLocked(text string, isQuery bool) ([]float32, error) {` |
| `docs/audit-2026-09-02.md|internal/serveapp/decoder_embedder.go:181` | goinfer | `func (e *decoderEmbedder) tokenize(text string, isQuery bool) ([]int, error) {` |
| `docs/audit-2026-09-02.md|internal/serveapp/decoder_embedder.go:84` | goinfer | `maxTokens: 0,` |
| `docs/audit-2026-09-02.md|internal/serveapp/embeddings.go:122` | goinfer | `promptTokens := s.countEmbedTokens(inputs, isQuery)` |
| `docs/audit-2026-09-02.md|internal/serveapp/embeddings.go:31` | goinfer | `// positions. maxEmbedInputs matches OpenAI's per-request batch cap; maxEmbedInputBytes ` |
| `docs/audit-2026-09-02.md|internal/serveapp/embeddings.go:34` | goinfer | `maxEmbedInputs     = 2048` |
| `docs/audit-2026-09-02.md|internal/serveapp/heartbeat_test.go:161` | goinfer | `// G21 end-to-end. What this asserts is bounded by the model available: the 1.5B` |
| `docs/audit-2026-09-02.md|internal/serveapp/helpers.go:129` | goinfer | `// Don't echo the raw json error: UnmarshalTypeError's default string leaks the Go struc` |
| `docs/audit-2026-09-02.md|internal/serveapp/helpers.go:171` | goinfer | `func sseSend(w http.ResponseWriter, f http.Flusher, v any) {` |
| `docs/audit-2026-09-02.md|internal/serveapp/helpers.go:203` | goinfer | `anchor: var sseHeartbeatInterval = 10 * time.Second` |
| `docs/audit-2026-09-02.md|internal/serveapp/helpers.go:207` | goinfer | `func sseHeartbeat(w http.ResponseWriter, f http.Flusher) (stop func()) {` |
| `docs/audit-2026-09-02.md|internal/serveapp/helpers.go:392` | goinfer | `var reqCounter atomic.Uint64` |
| `docs/audit-2026-09-02.md|internal/serveapp/liveness.go:10` | goinfer | `// Model liveness + the drain-based unload path. See docs/completed/task-admin-unload-dr` |
| `docs/audit-2026-09-02.md|internal/serveapp/liveness.go:159` | goinfer | `if s.cfg.sessionDir != "" && s.cfg.kvSessions > 0 {` |
| `docs/audit-2026-09-02.md|internal/serveapp/main.go:1088` | goinfer | `for _, lm := range srv.modelList() {` |
| `docs/audit-2026-09-02.md|internal/serveapp/main.go:156` | goinfer | `func (s modelSpec) explicitQuant(cfg config) string {` |
| `docs/audit-2026-09-02.md|internal/serveapp/main.go:359` | goinfer | `flag.IntVar(&cfg.ctxSize, "ctx", 0, "GPU-resident KV capacity in positions (per-model ov` |
| `docs/audit-2026-09-02.md|internal/serveapp/main.go:393` | goinfer | `// discoverable only by reading the source. The MoE refusal and the` |
| `docs/audit-2026-09-02.md|internal/serveapp/main.go:404` | goinfer | `if cfg.cpuExactPrefill \|\| !cfg.cpuFastAttention {` |
| `docs/audit-2026-09-02.md|internal/serveapp/main.go:405` | goinfer | `os.Setenv("GOINFER_CPU_FAST_ATTENTION", "0")` |
| `docs/audit-2026-09-02.md|internal/serveapp/main.go:496` | goinfer | `mux.HandleFunc("GET /health", auth(srv.handleHealth))` |
| `docs/audit-2026-09-02.md|internal/serveapp/main.go:514` | goinfer | `// ReadHeaderTimeout + ReadTimeout + IdleTimeout bound slow-header (slowloris), slow-bod` |
| `docs/audit-2026-09-02.md|internal/serveapp/main.go:563` | goinfer | `srvCancel()       // cancel in-flight generations (via BaseContext) so they release lm.m` |
| `docs/audit-2026-09-02.md|internal/serveapp/main.go:728` | goinfer | `if visionModelType(cand) == "qwen2_5_vl" {` |
| `docs/audit-2026-09-02.md|internal/serveapp/main.go:846` | goinfer | `giwPath, err := prequant.EnsureCachedGIW(ctx, spec.path, opts.Quant)` |
| `docs/audit-2026-09-02.md|internal/serveapp/openai.go:1036` | goinfer | `for id := range stream {` |
| `docs/audit-2026-09-02.md|internal/serveapp/openai.go:1041` | goinfer | `text, _ := lm.tk.Decode(ids)` |
| `docs/audit-2026-09-02.md|internal/serveapp/openai.go:1042` | goinfer | `if cut, which, hit := firstStop(text, gr.stopStrings); hit {` |
| `docs/audit-2026-09-02.md|internal/serveapp/openai.go:1095` | goinfer | `func (lm *loadedModel) logprobs(lps []decoder.SampleInfo) map[string]any {` |
| `docs/audit-2026-09-02.md|internal/serveapp/openai.go:417` | goinfer | `type completionReq struct {` |
| `docs/audit-2026-09-02.md|internal/serveapp/openai.go:498` | goinfer | `if err := lm.promptTooLargeForContext(chatInputBytes(req.Messages)); err != nil {` |
| `docs/audit-2026-09-02.md|internal/serveapp/openai.go:516` | goinfer | `if !lm.enter(w) {` |
| `docs/audit-2026-09-02.md|internal/serveapp/openai.go:539` | goinfer | `if req.StreamOptions != nil && req.StreamOptions.IncludeUsage {` |
| `docs/audit-2026-09-02.md|internal/serveapp/openai.go:661` | goinfer | `StopIDs:     lm.stopIDs,` |
| `docs/audit-2026-09-02.md|internal/serveapp/openai.go:663` | goinfer | `TopLogprobs: deref(sm.TopLogprobs, 0),` |
| `docs/audit-2026-09-02.md|internal/serveapp/openai.go:742` | goinfer | `gr.maxTokens = clampMaxTokens(gr.maxTokens, len(promptIDs), ctx)` |
| `docs/audit-2026-09-02.md|internal/serveapp/openai.go:744` | goinfer | `g, err := grammarFor(sm.ResponseFormat)` |
| `docs/audit-2026-09-02.md|internal/serveapp/openai.go:750` | goinfer | `m := constrain.NewMasker(g, lm.cachedTokenBytes(), eos).StopWhenComplete()` |
| `docs/audit-2026-09-02.md|internal/serveapp/openai.go:881` | goinfer | `func (lm *loadedModel) promptFor(system string, turns []chat.Turn) ([]int, error) {` |
| `docs/audit-2026-09-02.md|internal/serveapp/openai.go:895` | goinfer | `func genErr(err error) error {` |
| `docs/audit-2026-09-02.md|internal/serveapp/openai.go:914` | goinfer | `// GPU-resident models take the STATELESS path. decoder.Generate only engages the reside` |
| `docs/audit-2026-09-02.md|internal/serveapp/responses.go:130` | goinfer | `ids, err := lm.chatPrompt(messages)` |
| `docs/audit-2026-09-02.md|internal/serveapp/responses.go:227` | goinfer | `var stopBeat func()` |
| `docs/audit-2026-09-02.md|internal/serveapp/responses.go:273` | goinfer | `// Tool-call continuations round-trip via the next request's input; store the` |
| `docs/audit-2026-09-02.md|internal/serveapp/responses.go:290` | goinfer | `func responseInputToMessages(raw json.RawMessage) ([]chatMessage, error) {` |
| `docs/audit-2026-09-02.md|internal/serveapp/responses.go:94` | goinfer | `if req.PreviousResponseID != "" {` |
| `docs/audit-2026-09-02.md|internal/serveapp/responses_test.go:133` | goinfer | `// 4. Tool round-trip: forced function call → a function_call output item.` |
| `docs/audit-2026-09-02.md|internal/serveapp/sessions.go:126` | goinfer | `func (l *sessionLRU) acquire(prompt []int) *decoder.Session {` |
| `docs/audit-2026-09-02.md|internal/serveapp/sessions.go:155` | goinfer | `func bestExtend(sessions [][]int, prompt []int) int {` |
| `docs/audit-2026-09-02.md|internal/serveapp/sessions.go:20` | goinfer | `UNKEYABLE` |
| `docs/audit-2026-09-02.md|internal/serveapp/sessions.go:236` | goinfer | `if err := os.MkdirAll(l.dir, 0o755); err != nil {` |
| `docs/audit-2026-09-02.md|internal/serveapp/sessions.go:375` | goinfer | `continue // sliding-window (ring) cache: not yet persistable (Inc 3)` |
| `docs/audit-2026-09-02.md|internal/serveapp/tools.go:104` | goinfer | `var stopBeat func()` |
| `docs/audit-2026-09-02.md|internal/serveapp/tools.go:105` | goinfer | `if f != nil {` |
| `docs/audit-2026-09-02.md|internal/serveapp/tools.go:109` | goinfer | `stopBeat = sseHeartbeat(w, f)` |
| `docs/audit-2026-09-02.md|internal/serveapp/tools.go:111` | goinfer | `finish, nComp, _, _, gerr := lm.drive(r.Context(), gr, func(t string) {` |
| `docs/audit-2026-09-02.md|internal/serveapp/tools.go:118` | goinfer | `sseSend(w, f, chatChunk(id, created, lm.name, delta{Content: out}, nil))` |
| `docs/audit-2026-09-02.md|internal/serveapp/tools.go:122` | goinfer | `stopBeat() // joins the ticker goroutine before anything else writes to w` |
| `docs/audit-2026-09-02.md|internal/serveapp/tools.go:147` | goinfer | `usagev := usage{len(gr.promptIDs), nComp, len(gr.promptIDs) + nComp}` |
| `docs/audit-2026-09-02.md|internal/serveapp/tools.go:149` | goinfer | `if req.Stream {` |
| `docs/audit-2026-09-02.md|internal/serveapp/tools.go:215` | goinfer | `g, gerr := constrain.ToolCallGrammar(prefix, suffix, argsKey, forced.Name, array, forced` |
| `docs/audit-2026-09-02.md|internal/serveapp/tools.go:223` | goinfer | `m := constrain.NewMasker(g, constrain.TokenBytes(lm.vocab, lm.tk.TokenText), eos).StopWh` |
| `docs/audit-2026-09-02.md|internal/serveapp/tools.go:23` | goinfer | `func (s *server) serveChatToolsWith(w http.ResponseWriter, r *http.Request, req chatReq,` |
| `docs/audit-2026-09-02.md|internal/serveapp/tools.go:257` | goinfer | `// toolChoiceMode returns "auto" (default), "none", or "required"/"function" from` |
| `docs/audit-2026-09-02.md|internal/serveapp/tools.go:272` | goinfer | `func forcedTool(toolChoice json.RawMessage, tools []chat.Tool) *chat.Tool {` |
| `docs/audit-2026-09-02.md|internal/serveapp/vision_serve.go:133` | goinfer | `system, turns := messagesToTurns(req.Messages)` |
| `docs/audit-2026-09-02.md|internal/serveapp/vision_serve.go:151` | goinfer | `if req.Stream {` |
| `docs/audit-2026-09-02.md|internal/serveapp/vision_serve.go:58` | goinfer | `pv, err := vision.Preprocess(img.data, lm.vcfg)` |
| `docs/audit-2026-09-02.md|internal/serveapp/vision_serve.go:72` | goinfer | `ids, err := lm.encode(lm.tmpl.Render(system, turns))` |
| `docs/audit-2026-09-02.md|metal/attn_shape_test.go:149` | goinfer | `enc.Dispatch(pAttn, nH*128, 128, qB, kc, vc, out, uNH, uNKV, uHd, uNKeys, uScale, uWin)` |
| `docs/audit-2026-09-02.md|metal/backend.go:116` | goinfer | `if os.Getenv("GOINFER_NO_RESIDENT_MEM_GUARD") != "" {` |
| `docs/audit-2026-09-02.md|metal/backend.go:119` | goinfer | `need := m.ResidentWeightBytes()` |
| `docs/audit-2026-09-02.md|metal/backend.go:80` | goinfer | `if !residentFitsMemory(m) {` |
| `docs/audit-2026-09-02.md|metal/batched_verify_test.go:139` | goinfer | `e.Dispatch(r.pRope, r.nH*g.half, 64, qkv.At(m*qkvRows*4), L.invf, g.uHd, uPos, g.uQtotal` |
| `docs/audit-2026-09-02.md|metal/close_leak_test.go:182` | goinfer | `requireHeavyModel(t)` |
| `docs/audit-2026-09-02.md|metal/close_leak_test.go:45` | goinfer | `requireHeavyModel(t)` |
| `docs/audit-2026-09-02.md|metal/cmdbuf_status_test.go:94` | goinfer | `// This exercises the ceil-sized reduction on device for a %8 vocab (tmVocab=64) — a com` |
| `docs/audit-2026-09-02.md|metal/deltanet.go:139` | goinfer | `e.Dispatch(r.pDnNorm, dp.nk*128, 128, r.dnConvOut, r.dnQn, r.dnKn, r.uDnNk, r.uDnHk, r.u` |
| `docs/audit-2026-09-02.md|metal/deltanet.go:149` | goinfer | `e.DispatchTG(r.pSAResid, r.H*32, 256, dp.valueDim*2, D.outW, D.outS, r.dnGq, r.dnGSc, r.` |
| `docs/audit-2026-09-02.md|metal/gemma4_moe.go:333` | goinfer | `panic(fmt.Sprintf("metal gemma4 MoE pread gate\|up expert %d: %v", ei, err))` |
| `docs/audit-2026-09-02.md|metal/gemma4_moe.go:387` | goinfer | `e.Dispatch(r.pRms, 256, 256, r.x, ml.preFFN, r.mq, r.mSc, r.uH, r.uEps, r.uAddOne)` |
| `docs/audit-2026-09-02.md|metal/gemma4_moe.go:408` | goinfer | `for j := 0; j < g.topK; j++ {` |
| `docs/audit-2026-09-02.md|metal/gptoss_real20b_test.go:42` | goinfer | `// modelPath, NOT a direct environment read: the asset registry owns GOINFER_GPTOSS_GGUF` |
| `docs/audit-2026-09-02.md|metal/heavytest_test.go:20` | goinfer | `func requireHeavyModel(t *testing.T) {` |
| `docs/audit-2026-09-02.md|metal/kernels.go:511` | goinfer | `kernel void rope(device float* x[[buffer(0)]], device const float* invf[[buffer(1)]],` |
| `docs/audit-2026-09-02.md|metal/kernels.go:515` | goinfer | `float th=float(pos)*invf[dd]; float c=cos(th)*scale,s=sin(th)*scale;` |
| `docs/audit-2026-09-02.md|metal/kernels.go:606` | goinfer | `kernel void attention(device const float* q[[buffer(0)]], device const half* kc[[buffer(` |
| `docs/audit-2026-09-02.md|metal/layer_test.go:145` | goinfer | `enc.Dispatch(pAttn, nH*128, 128, qB, kc, vc, ctx, uNH, uNKV, uHd, uNKeys, uScale, uWindo` |
| `docs/audit-2026-09-02.md|metal/model.go:1457` | goinfer | `e.Dispatch(r.pRope, r.nH*g.half, 64, r.qkv, L.invf, g.uHd, r.uPos, g.uQtotal, g.uHalf, L` |
| `docs/audit-2026-09-02.md|metal/model.go:330` | goinfer | `func maxThreadgroupStageBytes(hidden, qWidth, moeInter, g4moeInter int) int {` |
| `docs/audit-2026-09-02.md|metal/model.go:342` | goinfer | `if words, scales, ok := int4DirectWords(w); ok {` |
| `docs/audit-2026-09-02.md|metal/model.go:440` | goinfer | `if preciseMathCompile \|\| os.Getenv("GOINFER_PRECISE_MATH") != "" {` |
| `docs/audit-2026-09-02.md|metal/model.go:485` | goinfer | `r.kvF32 = false` |
| `docs/audit-2026-09-02.md|metal/model.go:49` | goinfer | `anchor: const (` |
| `docs/audit-2026-09-02.md|metal/model.go:772` | goinfer | `// Vocab is NOT checked here (2026-08-18): it never routes through an SA-family kernel —` |
| `docs/audit-2026-09-02.md|metal/model.go:887` | goinfer | `if r.g4moe != nil && r.g4moe.paged && os.Getenv("GOINFER_MOE_RESIDENCY") != "0" && Resid` |
| `docs/audit-2026-09-02.md|metal/moe.go:106` | goinfer | `for (uint j=0u;j<k;j++) {` |
| `docs/audit-2026-09-02.md|metal/moe.go:203` | goinfer | `if (lane==0) out[row] += wgt[slot]*(acc*asc[0] + bias[idx[slot]*rowsPerExpert + row]);` |
| `docs/audit-2026-09-02.md|metal/moe.go:236` | goinfer | `uint biasOff = hasBias != 0u ? idx[slot]*2u*I : 0u;` |
| `docs/audit-2026-09-02.md|metal/moe.go:404` | goinfer | `if s := os.Getenv("GOINFER_METAL_MOE_SLOTS"); s != "" {` |
| `docs/audit-2026-09-02.md|metal/moe.go:517` | goinfer | `panic(fmt.Sprintf("metal MoE pread gate expert %d: %v", ei, err))` |
| `docs/audit-2026-09-02.md|metal/moe.go:589` | goinfer | `e.Dispatch(r.pRms, 256, 256, r.x, L.postNorm, r.mq, r.mSc, r.uH, r.uEps, r.uAddOne)` |
| `docs/audit-2026-09-02.md|metal/moe.go:613` | goinfer | `for j := 0; j < mo.k; j++ {` |
| `docs/audit-2026-09-02.md|metal/moe.go:638` | goinfer | `e.DispatchTG(mo.pGU, (2*mo.inter)*32, 256, r.H*2, s.guW, s.guS, r.mq, r.mSc, r.gu, r.uH,` |
| `docs/audit-2026-09-02.md|metal/moe.go:639` | goinfer | `if mo.isGptOss {` |
| `docs/audit-2026-09-02.md|metal/moe.go:679` | goinfer | `func (r *resident) forwardLogitsMoEPaged(pos int) (logits []float32) {` |
| `docs/audit-2026-09-02.md|metal/prefill_nan_test.go:20` | goinfer | `requireHeavyModel(t)` |
| `docs/audit-2026-09-02.md|metal/prefill_parity_test.go:20` | goinfer | `requireHeavyModel(t)` |
| `docs/audit-2026-09-02.md|metal/qwen35_35b_paged_test.go:95` | goinfer | `r, err := buildResident(m)` |
| `docs/audit-2026-09-02.md|metal/snapshot_golden_test.go:40` | goinfer | `// ⚠ RE-BAKE OWED (audit G-02, fixed on Linux where this suite cannot run). The checkpoi` |
| `docs/audit-2026-09-02.md|metal/snapshot_golden_test.go:52` | goinfer | `// no heavy-model dependency). Coverage: mixtral-tiny is full-causal (attention softmax ` |
| `docs/audit-2026-09-02.md|multimodal/qwen_preprocess.go:213` | goinfer | `func qwenExtractRGB(img image.Image, h, w int) []float32 {` |
| `docs/audit-2026-09-02.md|multimodal/qwen_preprocess.go:24` | goinfer | `// patch-row, patch-col). PIL-bicubic resize parity is a separate refinement — the` |
| `docs/audit-2026-09-02.md|multimodal/qwen_preprocess_test.go:12` | goinfer | `// TestQwenPreprocess_exact gates the Qwen2.5-VL preprocessing (normalize +` |
| `docs/audit-2026-09-02.md|scripts/apidiff_check.sh:11` | goinfer | `#   scripts/apidiff_check.sh                 # baseline v0.13.0 (the last released tag)` |
| `docs/audit-2026-09-02.md|scripts/asset_registry.py:45` | goinfer | `def models_root():` |
| `docs/audit-2026-09-02.md|scripts/ci_checks.py:51` | goinfer | `HYGIENE = re.compile(r"gofmt\|staticcheck\|^vet\|^build\|cleanliness\|lint", re.I)` |
| `docs/audit-2026-09-02.md|scripts/gate_ledger.py:151` | goinfer | `if e is None:` |
| `docs/audit-2026-09-02.md|scripts/gptoss_tiny_golden.py:32` | goinfer | `UNKEYABLE` |
| `docs/audit-2026-09-02.md|scripts/queue_citation_lint.py:574` | goinfer | `update = "--update" in sys.argv` |
| `docs/audit-2026-09-02.md|tokenizer/bytelevel.go:185` | goinfer | `// decodeByteLevel inverts the byte-level map: render each piece (special tokens` |
| `docs/audit-2026-09-02.md|tokenizer/bytelevel.go:206` | goinfer | `// splitGPT2 reproduces the Qwen/Llama-3 pretokenizer split — the GPT-2 regex` |
| `docs/audit-2026-09-02.md|tokenizer/bytelevel.go:231` | goinfer | `for i := 0; i < n; {` |
| `docs/audit-2026-09-02.md|tokenizer/bytelevel.go:27` | goinfer | `func (t *Tokenizer) initByteLevel(tj *tokenizerJSON, dir string) error {` |
| `docs/audit-2026-09-02.md|tokenizer/bytelevel.go:350` | goinfer | `func normalizerForm(raw json.RawMessage) (norm.Form, bool) {` |
| `docs/audit-2026-09-02.md|tokenizer/bytelevel_test.go:18` | goinfer | `//	.venv/bin/python scripts/pin_qwen3_tokenizer.py` |
| `docs/audit-2026-09-02.md|tokenizer/doc.go:26` | goinfer | `// Golden parity against HF `tokenizers` is the gate for every family (M2 /` |
| `docs/audit-2026-09-02.md|tokenizer/gguf.go:118` | goinfer | `for i, tok := range tokens {` |
| `docs/audit-2026-09-02.md|tokenizer/gguf.go:262` | goinfer | `// byteLevelKnobs maps a tokenizer.ggml.pre identifier to the byte-level pipeline` |
| `docs/audit-2026-09-02.md|tokenizer/gguf.go:278` | goinfer | `default: // "gpt-2", "default", "", and unrecognized` |
| `docs/audit-2026-09-02.md|tokenizer/gguf_test.go:19` | goinfer | `//	.venv/bin/python scripts/pin_tinyllama_tokenizer.py` |
| `docs/audit-2026-09-02.md|tokenizer/sentencepiece.go:412` | goinfer | `// Encode turns text into token ids. If addBOS, prepend the BOS token (the` |
| `docs/audit-2026-09-02.md|tokenizer/sentencepiece.go:723` | goinfer | `func (t *Tokenizer) Decode(ids []int) (string, error) {` |
| `docs/audit-2026-09-02.md|tokenizer/sentencepiece.go:791` | goinfer | `if t.mode == modeByteLevel {` |
| `docs/audit-2026-09-02.md|tokenizer/sentencepiece_test.go:51` | goinfer | `//	.venv/bin/python scripts/pin_gemma_tokenizer.py` |
| `docs/audit-2026-09-02.md|tokenizer/tokentext_test.go:30` | goinfer | `if _, err := os.Stat(c.dir); errors.Is(err, fs.ErrNotExist) {` |
| `docs/book/04-the-loop-and-the-kv-cache.md|decoder/deltanet.go:145` | goinfer | `// last K-1 conv inputs (so the causal conv has its left context at decode) and` |
| `docs/book/09-guessing-ahead.md|decoder/deltanet.go:145` | goinfer | `// last K-1 conv inputs (so the causal conv has its left context at decode) and` |
| `docs/book/09-guessing-ahead.md|decoder/speculative.go:89` | goinfer | `// rolls back the rejected tail. A recurrent (Mamba-2 / Gated DeltaNet) or staged` |
| `docs/cuda-megakernel-spec.md|gpu/attention.go:17` | goinfer | `// uses f64 accumulation; the GPU f32 — cosine ~1.0, not bit-exact).` |
| `docs/cuda-megakernel-spec.md|gpu/decoderunner.go:807` | goinfer | `// moeExpert records one indexed sparse-expert GEMV: dst[n] = expert[idx[slot]]·aq` |
| `docs/cuda-megakernel-spec.md|gpu/decoderunner.go:912` | goinfer | `// relu²→int8 → down + residual into xd. The other kinds fall through to the mixer.` |
| `docs/cuda-megakernel-spec.md|gpu/forward_parity_test.go:36` | goinfer | `func TestWebGPU_forwardParity(t *testing.T) {` |
| `docs/cuda-megakernel-spec.md|gpu/gemv.go:41` | goinfer | `@compute @workgroup_size(64)` |
| `docs/gpu-residency-coverage.md|decoder/registry.go:175` | goinfer | `IntermediateDim:   cfg.IntermediateDim,` |
| `docs/how-inference-works.md|decoder/attention.go:107` | goinfer | `if !arch.LearnedPosEmbed && !arch.isNoPELayer(layer) {` |
| `docs/how-inference-works.md|decoder/attention.go:147` | goinfer | `cache.Append(layer, k, v)` |
| `docs/how-inference-works.md|decoder/attention.go:59` | goinfer | `nH, nKV, hd := arch.headsAt(layer), arch.NumKVHeads, arch.HeadDim` |
| `docs/how-inference-works.md|decoder/kvcache.go:132` | goinfer | `subCapture bool` |
| `docs/how-inference-works.md|decoder/kvcache.go:20` | goinfer | `func quantizeHeads(src []float32, q []int8, scales []float32, nKV, headDim int) {` |
| `docs/how-inference-works.md|decoder/model.go:545` | goinfer | `h[i] *= scale` |
| `docs/how-inference-works.md|decoder/model.go:586` | goinfer | `lw := &m.w.Layers[l]` |
| `docs/how-inference-works.md|decoder/registry.go:19` | goinfer | `var registry = map[string]archAdapter{` |
| `docs/how-inference-works.md|decoder/sampler.go:122` | goinfer | `// can never silently diverge. They are separate predicates, not one widened one, so tha` |
| `docs/how-inference-works.md|decoder/sampler.go:129` | goinfer | `// though a temperature is set — the `top_k=1` shape. It is TRUE at any temperature, whi` |
| `docs/how-inference-works.md|decoder/sampler.go:131` | goinfer | `// distribution restricted to ONE token is deterministic regardless of that token's prob` |
| `docs/how-inference-works.md|decoder/session.go:71` | goinfer | `// stale history. Callers must skip it (and reconcile) for an empty prompt, so a rejecte` |
| `docs/ideas-weight-memory.md|decoder/mlp.go:69` | goinfer | `anchor: func mlp(h, out []float32, lw *LayerWeights, arch *Architecture, be Backend, scr` |
| `docs/measurements/c3-metal-consumer-window-v0.14.0.md|metal/gemma_parity_test.go:84` | goinfer | `t.Fatal("metal resident DECLINED — admission says it should be admitted")` |
| `docs/measurements/c3-metal-consumer-window.md|decoder/model.go:301` | goinfer | `switch o.Backend {` |
| `docs/measurements/c3-metal-consumer-window.md|decoder/residency.go:580` | goinfer | `func (m *Model) withResidency() *Model {` |
| `docs/measurements/c3-metal-consumer-window.md|metal/gemma_parity_test.go:84` | goinfer | `t.Fatal("metal resident DECLINED — admission says it should be admitted")` |
| `docs/measurements/demo-chat-gemma4e2b-blocked-2026-08-22.md|decoder/config.go:252` | goinfer | `SharedKVLayers          int   `json:"num_kv_shared_layers"`` |
| `docs/measurements/demo-chat-gemma4e2b-blocked-2026-08-22.md|decoder/gguf.go:2252` | goinfer | `firstShared := arch.NumLayers - g4.SharedKVLayers` |
| `docs/measurements/demo-chat-gemma4e2b-blocked-2026-08-22.md|decoder/registry.go:275` | goinfer | `SharedKVLayers:          cfg.SharedKVLayers,` |
| `docs/measurements/demo-chat-tier2-gates-2026-08-22.md|decoder/config.go:1101` | goinfer | `// under "text_config" rather than at the top level. Flatten it: decode` |
| `docs/measurements/demo-chat-tier2-gates-2026-08-22.md|decoder/weights.go:549` | goinfer | `if have["model.language_model.embed_tokens.weight"] {` |
| `docs/measurements/demo-chat-tier2-gates-2026-08-22.md|decoder/weights.go:962` | goinfer | `if d.inProjQKV, err = mkQ(nm("linear_attn.in_proj_qkv.weight"), convDim, hidden); err !=` |
| `docs/measurements/moe-expert-batching-m1-vs-mn-2026-09-01.md|decoder/forwardn.go:537` | goinfer | `ff, err = moeMLP(row(norm, i, hidden), lw, arch, be, moePrefillScr, m.pager)` |
| `docs/measurements/moe-expert-batching-m1-vs-mn-2026-09-01.md|decoder/mlp.go:292` | goinfer | `matmul(be, &ex.Gate, h, gate, 1)` |
| `docs/measurements/theta-per-backend-2026-09-01.md|metal/backend.go:253` | goinfer | `func (a *metalResident) ForwardN(embeddings [][]float32, startPos int) ([][]float32, err` |
| `docs/multimodal.md|decoder/config.go:1110` | goinfer | `if json.Unmarshal(b, &nest) == nil && len(nest.TextConfig) > 0 {` |
| `docs/multimodal.md|decoder/gguf_qwen35.go:77` | goinfer | `anchor: func ggufQwen35Config(g *embed.GGUFFile) (*Config, error) {` |
| `docs/multimodal.md|decoder/weights.go:410` | goinfer | `const shardIndexFile = "model.safetensors.index.json"` |
| `docs/ollama-chase.md|cuda/resident.go:1340` | goinfer | `// All of it runs ON the executor thread — that thread made the context current — and th` |
| `docs/ollama-chase.md|cuda/resident.go:42` | goinfer | `// resolveCtxCap turns a request into the effective resident KV capacity:` |
| `docs/ollama-chase.md|cuda/resident.go:436` | goinfer | `g4x1, g4x2, g4rn Buffer` |
| `docs/ollama-chase.md|cuda/resident.go:685` | goinfer | `// declined to the staged/CPU path upstream.` |
| `docs/ollama-chase.md|decoder/gguf.go:631` | goinfer | `numLayers := u("block_count") - u("nextn_predict_layers")` |
| `docs/ollama-chase.md|decoder/gguf_qwen35.go:33` | goinfer | `numLayers := blocks - u("nextn_predict_layers") // drop the NextN/MTP block(s)` |
| `docs/ollama-chase.md|decoder/model.go:1037` | goinfer | `// sample. Identical to the logits path — guarded by ArgmaxEquivalent/GreedyEquivalent.` |
| `docs/ollama-chase.md|decoder/model.go:916` | goinfer | `// logits. On the batched archs this runs the layers at M=len in one pass (each` |
| `docs/ollama-chase.md|decoder/registry.go:1056` | goinfer | `// num_nextn_predict_layers MTP head is dropped (only num_hidden_layers load). The` |
| `docs/ollama-chase.md|decoder/residency.go:747` | goinfer | `return false, "sequential — this backend has no batched prefill (per-token resident forw` |
| `docs/ollama-chase.md|decoder/weightmat.go:336` | goinfer | `var matmulWSPool = sync.Pool{New: func() any { return new(linalg.Workspace) }}` |
| `docs/ollama-chase.md|decoder/weights.go:541` | goinfer | `// index so one loader serves both — the vision tower (model.visual.*) and MTP` |
| `docs/parity-coverage-policy.md|cuda/resident.go:1132` | goinfer | `// always been allocated without one, and a hard failure here would regress every driver` |
| `docs/parity-coverage-policy.md|linalg/dot.go:25` | aikit | `sum += a[k] * b[k]` |
| `docs/plan-cpubrrr-steal-and-bindings.md|decoder/registry.go:58` | goinfer | `"gpt_oss":          gptOssArchitecture,      // gpt-oss (20b/120b): sparse MoE + per-hea` |
| `docs/plan-cpubrrr-steal-and-bindings.md|linalg/quant.go:446` | aikit | `func QuantizeGroupInt4Row(row []float32, cols, group int, packed []byte, scales []float3` |
| `docs/post-v1.0-models.md|decoder/registry.go:1308` | goinfer | `func nemotronhArchitecture(cfg *Config) (*Architecture, *tensorSchema, error) {` |
| `docs/post-v1.0-models.md|decoder/registry.go:1424` | goinfer | `func deepseekArchitecture(cfg *Config) (*Architecture, *tensorSchema, error) {` |
| `docs/post-v1.0-models.md|decoder/registry.go:447` | goinfer | `// cohere2Architecture expresses Cohere2 / Command-R7B (model_type "cohere2":` |
| `docs/post-v1.0-models.md|decoder/registry.go:58` | goinfer | `"gpt_oss":          gptOssArchitecture,      // gpt-oss (20b/120b): sparse MoE + per-hea` |
| `docs/post-v1.0-models.md|decoder/registry.go:757` | goinfer | `Name:            "qwen3_5_moe",` |
| `docs/prompts/attention-a1-bit-identical-restructure.md|decoder/forwardn.go:632` | goinfer | `// of the next matmul); then ctx_head[K,hd] = scores·V_head, expressed as` |
| `docs/prompts/attention-a1-bit-identical-restructure.md|decoder/forwardn.go:840` | goinfer | `// stride kvDim (vt's column index steps by a whole KV row) — skipping` |
| `docs/prompts/goinfer-w4a8-opsperbyte-citations.md|linalg/quant.go:206` | aikit | `func QuantizeActivationsInto(aq []int8, scales []float32, a []float32, M, K int) {` |
| `docs/prompts/mac-cpu-decode-vs-ollama.md|decoder/sampler_chunked.go:111` | goinfer | `workers := min(runtime.GOMAXPROCS(0), numChunks)` |
| `docs/prompts/mac-cpu-decode-vs-ollama.md|decoder/weightmat.go:365` | goinfer | `w.MatmulBTW4A8Into(ws, a, dst, M)` |
| `docs/prompts/mac-cpu-decode-vs-ollama.md|decoder/weights.go:286` | goinfer | `workers := min(runtime.GOMAXPROCS(0), n)` |
| `docs/prompts/mac-demo-finish.md|internal/chatapp/main.go:358` | goinfer | `fmt.Fprintf(os.Stderr, "\033[2m[%d tok, %.1f tok/s]\033[0m", len(out), float64(len(out))` |
| `docs/queue-engineering.md|cmd/gate/configs.go:14` | goinfer | `models := env("GOINFER_GATE_MODELS", filepath.Join(home(), "models"))` |
| `docs/queue-engineering.md|cmd/gate/gpu.go:365` | goinfer | `g.models = env("GOINFER_GATE_MODELS", filepath.Join(home(), "models"))` |
| `docs/queue-engineering.md|cuda/argmax_tiebreak_test.go:19` | goinfer | `func TestArgmaxTieBreak(t *testing.T) {` |
| `docs/queue-engineering.md|cuda/backend.go:1144` | goinfer | `// cache, so the cap is correct by construction rather than covered by a margin.` |
| `docs/queue-engineering.md|cuda/prefill.go:227` | goinfer | `defer func() {` |
| `docs/queue-engineering.md|cuda/resident.go:274` | goinfer | `// backend.go locals; the per-layer KV cache and UploadKV read r.layers[l].kvDim.` |
| `docs/queue-engineering.md|cuda/resident.go:499` | goinfer | `func (r *cudaResident) recordUpload(e error) {` |
| `docs/queue-engineering.md|decoder/forwardn.go:1007` | goinfer | `logits[j] = sc * float32(math.Tanh(float64(val/sc)))` |
| `docs/queue-engineering.md|decoder/kvsnapshot_gemma4_test.go:10` | goinfer | `func TestSnapshot_refusesNonUniformKVWidth_C05(t *testing.T) {` |
| `docs/queue-engineering.md|decoder/layerpaging.go:42` | goinfer | `// mu guards the mutable paging state below (audit C-30). The pager lives on *Model, sha` |
| `docs/queue-engineering.md|decoder/model.go:746` | goinfer | `// Diagnostic — same byte-identical-output contract as ForwardCapture. Not wired for own` |
| `docs/queue-engineering.md|decoder/serialize.go:638` | goinfer | `func (w *Weights) hasPopulatedLayers() bool {` |
| `docs/queue-engineering.md|decoder/serialize_shapecheck_test.go:15` | goinfer | `func TestValidateShapes_catchesArchMismatch(t *testing.T) {` |
| `docs/queue-engineering.md|decoder/serialize_test.go:436` | goinfer | `t.Fatalf("streamed length %d != buffered %d", n, len(want))` |
| `docs/queue-engineering.md|internal/giw/bundle.go:114` | goinfer | `if avail := fi.Size() - (tokOff + 4); tokLen > avail {` |
| `docs/queue-engineering.md|internal/serveapp/embeddings.go:26` | goinfer | `// Embedding request bounds (audit C-21). /v1/embeddings is deliberately un-queued (the ` |
| `docs/queue-engineering.md|internal/serveapp/main.go:555` | goinfer | `// A SECOND signal during the drain force-exits instead of being swallowed by the buffer` |
| `docs/queue-engineering.md|linalg/quant.go:138` | aikit | `dequantRowInt8(deq, bq, 1.0)` |
| `docs/queue-engineering.md|metal/model.go:1048` | goinfer | `r.logitsHost[j] = sc * float32(math.Tanh(float64(v/sc)))` |
| `docs/queue-engineering.md|scripts/bench_peer.py:407` | goinfer | `def gate_cell_idle():` |
| `docs/scoping-lfm2.md|decoder/arch.go:177` | goinfer | `type nemotronParams struct {` |
| `docs/scoping-lfm2.md|decoder/attention.go:97` | goinfer | `if arch.QKNorm {` |
| `docs/scoping-lfm2.md|decoder/config.go:903` | goinfer | `case c.UseQKNorm:` |
| `docs/scoping-lfm2.md|decoder/deltanet.go:176` | goinfer | `// 1. Projection + depthwise causal conv (+ SiLU). Taps t-K+1..t: the last K-1` |
| `docs/scoping-lfm2.md|decoder/forward_qwen35.go:30` | goinfer | `if arch.isLinearLayer(l) {` |
| `docs/scoping-lfm2.md|decoder/kvcache.go:50` | goinfer | `type KVCache struct {` |
| `docs/scoping-lfm2.md|decoder/mamba2.go:89` | goinfer | `// 2. Depthwise causal conv over xBC (+ bias, + SiLU). Taps t-K+1..t: the last` |
| `docs/scoping-lfm2.md|decoder/mamba2_chunked.go:60` | goinfer | `// Depthwise causal conv over xBC (+bias, +SiLU), then split into x/B/C.` |
| `docs/scoping-lfm2.md|decoder/rmsnorm.go:49` | goinfer | `func layerNorm(x, weight, bias []float32, rows, dim int, eps float64) {` |
| `docs/scoping-qwen38-flash-next.md|decoder/registry.go:1976` | goinfer | `// qwen35DenseArchitecture expresses Qwen3.8 (model_type qwen3_5): the SAME Gated-DeltaN` |
| `docs/scoping-qwen38-flash-next.md|decoder/registry.go:44` | goinfer | `"qwen3_5_moe_text": qwen35Architecture,      // the text-only checkpoint's model_type` |
| `docs/spec/09-mtp-heads.md|cuda/resident.go:241` | goinfer | `// owns a contiguous row. dnWin is the causal-conv ring, [(K-1)*convDim]. Both COMPOUND,` |
| `docs/spec/09-mtp-heads.md|cuda/resident.go:248` | goinfer | `dnWin, dnState               Buffer // persistent: conv ring, recurrent matrix state` |
| `docs/spec/09-mtp-heads.md|decoder/blockspec.go:399` | goinfer | `// breakEvenTokensPerRound is the acceptance below which block drafting LOSES.` |
| `docs/spec/09-mtp-heads.md|decoder/deltanet.go:147` | goinfer | `// head). Fixed size — independent of sequence length, and NOT position-` |
| `docs/spec/09-mtp-heads.md|decoder/deltanet.go:150` | goinfer | `type deltaState struct {` |
| `docs/spec/09-mtp-heads.md|decoder/deltanet.go:184` | goinfer | `win := st.convWin` |
| `docs/spec/09-mtp-heads.md|decoder/forwardn.go:116` | goinfer | `func (m *Model) specRollbackSafe() bool {` |
| `docs/spec/09-mtp-heads.md|decoder/gguf.go:631` | goinfer | `numLayers := u("block_count") - u("nextn_predict_layers")` |
| `docs/spec/09-mtp-heads.md|decoder/gguf_qwen35.go:33` | goinfer | `numLayers := blocks - u("nextn_predict_layers") // drop the NextN/MTP block(s)` |
| `docs/spec/09-mtp-heads.md|decoder/model.go:724` | goinfer | `if a.granite != nil \|\| a.nemotron != nil \|\| a.mla != nil \|\| a.llama4 != nil {` |
| `docs/spec/09-mtp-heads.md|decoder/registry.go:1056` | goinfer | `// num_nextn_predict_layers MTP head is dropped (only num_hidden_layers load). The` |
| `docs/spec/09-mtp-heads.md|decoder/speculative.go:92` | goinfer | `if !target.specRollbackSafe() {` |
| `docs/spec/09-mtp-heads.md|decoder/weights.go:541` | goinfer | `// index so one loader serves both — the vision tower (model.visual.*) and MTP` |
| `docs/spec/README.md|decoder/forwardn.go:116` | goinfer | `func (m *Model) specRollbackSafe() bool {` |
| `docs/task-attention-decode-cost.md|decoder/forwardn.go:632` | goinfer | `// of the next matmul); then ctx_head[K,hd] = scores·V_head, expressed as` |
| `docs/task-attention-decode-cost.md|decoder/forwardn.go:740` | goinfer | `// MatmulBTAcc64Strided runs the SAME sequential f64 reduction as` |
| `docs/task-attention-decode-cost.md|decoder/forwardn.go:840` | goinfer | `// stride kvDim (vt's column index steps by a whole KV row) — skipping` |
| `docs/task-attention-decode-cost.md|internal/serveapp/main.go:353` | goinfer | `flag.BoolVar(&cfg.moeCacheExperts, "moe-cache-experts", false, "run a MoE model whose ex` |
| `docs/task-attention-decode-cost.md|linalg/linalg.go:58` | aikit | `var parThreshold = 1 << 24 // 16.78M MACs` |
| `docs/task-attention-decode-cost.md|linalg/matmul_strided.go:30` | aikit | `func MatmulBTAcc64Strided(a, bMat, dst []float32, M, K, N, bOff, bRowStride, bElemStride` |
| `docs/task-freetoken-techniques.md|decoder/model.go:163` | goinfer | `MoECacheSlots int` |
| `docs/task-freetoken-techniques.md|internal/serveapp/main.go:275` | goinfer | `moeCacheSlots    int    // per-layer expert slot REQUEST (--moe-cache-slots); an upper b` |
| `docs/task-gpu-batched-prefill.md|decoder/residency.go:54` | goinfer | `// ResidentGreedy is an optional capability on a ResidentForward: compute the token's gr` |
| `docs/task-int4-int8-exact-mma.md|metal/kernels.go:220` | goinfer | `#define W4A8_BODY \` |
| `docs/task-int4-int8-exact-mma.md|metal/kernels.go:223` | goinfer | `device const half*  srow = bsc + (uint)gid*(K/32u); \` |
| `docs/task-int4-int8-exact-mma.md|metal/kernels.go:231` | goinfer | `acc += float(gi) * float(srow[wi>>2]); \` |
| `docs/task-int4-int8-exact-mma.md|metal/kernels.go:237` | goinfer | `kernel void gemv_w4a8_coal(device const uint* bq[[buffer(0)]], device const half* bsc[[b` |
| `docs/task-int4-int8-exact-mma.md|metal/kernels.go:241` | goinfer | `if (lid == 0) out[gid] = acc * asc[0];` |
| `docs/task-int4-int8-exact-mma.md|metal/kernels.go:287` | goinfer | `#define SA_BODY \` |
| `docs/task-int4-int8-exact-mma.md|metal/kernels.go:290` | goinfer | `uint G = K>>5u; \` |
| `docs/task-int4-int8-exact-mma.md|metal/kernels.go:293` | goinfer | `device const half*  sr = sct + (uint)row*G; \` |
| `docs/task-int4-int8-exact-mma.md|metal/kernels.go:301` | goinfer | `kernel void gemv_w4a8_sa(device const uint4* wq[[buffer(0)]], device const half* sct[[bu` |
| `docs/task-int4-int8-exact-mma.md|metal/kernels.go:307` | goinfer | `if (lane==0) out[row] = acc*asc[0];` |
| `docs/task-int4-int8-exact-mma.md|metal/kernels.go:51` | goinfer | `float sc=red[0]/127.0f; if(sc==0)sc=1; if(tid==0)asc[0]=sc; float inv=1/sc;` |
| `docs/task-int4-int8-exact-mma.md|metal/model.go:1327` | goinfer | `e.DispatchTG(r.pSABias, qkvRows*32, 256, r.H*2, L.qkvW, L.qkvS, r.aq, r.aSc, r.qkv, L.qk` |
| `docs/task-int4-int8-exact-mma.md|metal/model.go:1356` | goinfer | `e.Dispatch(r.pGemv, r.H*32, 32, L.dW, L.dS, r.dq, r.dSc, r.dO, r.uI)` |
| `docs/task-int4-int8-exact-mma.md|metal/model.go:456` | goinfer | `r.pRms, r.pQv, r.pGemv = pipe("rmsnorm_quant"), pipe("quant_vec"), pipe("gemv_w4a8_coal"` |
| `docs/task-int4-int8-exact-mma.md|metal/model.go:458` | goinfer | `r.pSA, r.pSABias, r.pSAResid = pipe("gemv_w4a8_sa"), pipe("gemv_w4a8_sa_bias"), pipe("ge` |
| `docs/task-metal-batched-verify-kernel.md|metal/kernels.go:220` | goinfer | `#define W4A8_BODY \` |
| `docs/task-metal-batched-verify-kernel.md|metal/kernels.go:287` | goinfer | `#define SA_BODY \` |
| `docs/task-metal-batched-verify-kernel.md|metal/model.go:330` | goinfer | `func maxThreadgroupStageBytes(hidden, qWidth, moeInter, g4moeInter int) int {` |
| `docs/task-moe-streaming.md|decoder/forwardn.go:455` | goinfer | `// Sequential: add the attention residual, then re-norm the updated stream for the MLP.` |
| `docs/task-moe-streaming.md|decoder/forwardn.go:94` | goinfer | `// MoE FFN itself stays per-row (router picks different experts per token).` |
| `docs/task-moe-streaming.md|decoder/mlp.go:83` | goinfer | `// Only the chosen experts are evaluated — the point of MoE.` |
| `docs/task-moe-streaming.md|decoder/moepaging.go:15` | goinfer | `// only K·L per token; the router's top-k selection is the demand signal. The` |
| `docs/task-moe-streaming.md|decoder/moepaging_test.go:13` | goinfer | `// it with the frequency-aware policy (TestSpanCache_evictsLeastRecentWithPolicy),` |
| `docs/task-moe-streaming.md|decoder/residency.go:130` | goinfer | `return m.residentProjsInt4()` |
| `docs/task-verification-surface-audit.md|decoder/blockspec.go:399` | goinfer | `// breakEvenTokensPerRound is the acceptance below which block drafting LOSES.` |
| `docs/task-zeno-compare.md|decoder/gguf.go:1245` | goinfer | `// gemma4's fused PLE/MoE tail can't stream incrementally; it falls back to a` |
| `docs/task-zeno-compare.md|decoder/gguf.go:1253` | goinfer | `if arch.gemma4 != nil {` |
| `docs/task-zeno-compare.md|decoder/gguf.go:1414` | goinfer | `embMat := func(name string, out, in int) (linalg.WeightMat, error) {` |
| `docs/task-zeno-compare.md|decoder/gguf.go:1517` | goinfer | `if g.Has("output.weight") {` |
| `docs/task-zeno-compare.md|decoder/serialize.go:170` | goinfer | `anchor: func SerializeWeightsToRow4(out io.Writer, w *Weights, id string) (int64, error)` |
| `docs/task-zeno-compare.md|decoder/weightmat.go:125` | goinfer | `func streamQuantized(rows, cols int, mode quantMode, rowInto func(r int, dst []float32) ` |
| `docs/task-zeno-compare.md|internal/prequant/prequant.go:65` | goinfer | `// 2) Weights half: transcode the GGUF straight into the bundle, ONE LAYER at a` |

## Bare file index

Generated. Every file referenced WITHOUT a line number, and the repo it resolves in.
Existence only — there is no line to key content against, which is recorded rather
than papered over.

| file | repo |
|---|---|
| `decoder/g26_sampler_bench_test.go` | goinfer |
| `decoder/real_oracle_test.go` | goinfer |
| `internal/serveapp/chaos_test.go` | goinfer |
| `internal/serveapp/fuzz_test.go` | goinfer |
| `internal/serveapp/openai.go` | goinfer |
| `scripts/bench_compare.sh` | goinfer |
| `scripts/bench_peer.py` | goinfer |
| `scripts/refresh_parity_hashes.sh` | goinfer |

<!-- /CITATION-INDEX -->
## RETRACTION, 2026-08-27 — the "152k sampler crossover" was a microbenchmark artifact

Raw: `docs/measurements/g26-152k-{anchor,head}.json`, log `g26-152k_run.log`.

I filed a finding that HEAD's sampler regressed +23.4% at 152k vocab, from
`BenchmarkG26SampleTemp1_152k`. **It does not reproduce end-to-end, and the sign is opposite.**
Tested on qwen2.5-coder-1.5B (vocab **151936** — the standard bench model already sits at the vocab
in question), `GOINFER_NO_OPTFWD=1` in BOTH arms so the optFwd effect cannot contaminate it, n=8:

| | anchor `ca29d6c` | HEAD | |
|---|---|---|---|
| greedy (control) | 219.81 ± 0.91 | 220.51 ± 1.36 | +0.3% |
| temp1.0_notrunc | 166.22 ± 2.31 | 180.37 ± 2.21 | **+8.5% HEAD FASTER** |
| **production sampling step** | **1467 us** | **1009 us** | **−31%** |
| *microbenchmark claimed* | *1358 us* | *1676 us* | *+23%* |

**In production HEAD's sampler is 31% FASTER at 152k. P10 delivered what it claimed.** There is no
sampler regression at any vocab measured: −35% at phi3-mini's 32064 (0.703 → 0.457 ms, from the
optFwd-OFF arms) and −31% at 151936.

**THE METHODOLOGICAL POINT, which is the durable part.** A tight-loop microbenchmark of the sampler
is not representative of decode, and this is the second time in ONE investigation that a sampler
microbenchmark produced a wrong conclusion — the first was `BenchmarkExpChunked`'s ~96 us
under-bounding the real tail by 7–10x. In a tight loop the scratch buffer stays hot and the
allocator serves from a warm free-list; in decode the same buffer is cold behind an ~8 ms forward.
The loop does not merely mis-scale the number, it **inverted the comparison**.

**Use the in-situ measurement instead, which is available and costs nothing extra: the same-build
greedy-vs-temp1.0 end-to-end difference, with `GOINFER_NO_OPTFWD=1` so overlap cannot confound it.**
Greedy's on-device argmax means that difference is the whole sampled tail rather than the sampler
alone — but it is measured where the code actually runs, and both times it disagreed with a
microbenchmark it was the microbenchmark that was wrong.

**Consequently the tail decomposition recorded above is DOWNGRADED.** Its "remainder 75 → 397 us"
split subtracted microbenchmark `Sample` figures from in-situ tails, and those figures are now known
to be unrepresentative — at 32k the microbenchmark says 553 us where production says 457 us. **The
split is not reliable and should not be quoted.** What is unaffected: the optFwd conclusion, which
rests entirely on an end-to-end A/B (`GOINFER_NO_OPTFWD=1`, n=15, plus the five-point ladder) and
never depended on the microbenchmark at all. `decoder/g26_sampler_bench_test.go` is kept, but its
doc comment should be read with this retraction attached.

## G27 · optFwdGate's hysteresis is a one-way latch, and its thresholds are calibrated on one model

Found 2026-08-28 while shipping the T ≤ 0.2 cap. Two independent defects in `decoder/spec_optfwd.go`,
neither of which the shipped cap depends on — it makes both moot in the losing regime — but both of
which matter if anyone re-enables the overlap more widely.

**1. The re-enable branch is unreachable.** `Observe` is called only from `optFwdStep`; the caller
invokes `optFwdStep` only when `Should()` is true. Once α drops below DisableAt the gate turns off,
no further outcomes are observed, α freezes, and `α ≥ EnableAt` can never be reached again within
that Generate. The documented dead band is one-way in practice.
`TestOptFwdGate_hysteresis` does not catch it because it calls `Observe` in an unconditional loop —
a unit test that is correct about the component while the composition defeats it. **A test at the
wiring level would have caught what a test at the component level could not.**

**2. EnableAt/DisableAt (0.90/0.75) are model-independent constants fitted on one model.** The
comment names it: the 90.9% worst-case break-even was measured on qwen2.5-coder-0.5b, a 152k-vocab
model. Break-even rises as the sampler share falls, so on phi3-mini (5.4% share vs 18.2%) the true
break-even is above 0.90 and the whole dead band sits below it — the gate cannot turn off in the
regime where the feature loses 2.8-6.8%.

**ESCALATED 2026-08-28 — G27 is no longer latent, it is blocking.** The latch stopped a measurement,
not just a code path: `TestOptFwd_hitRateLadder` cannot estimate low hit rates because the gate turns
off and stops observing (guess counts collapse to 6-18 on all three models). Break-even α is the
input the binary load-time gate needs (spec/10), so the latch now sits on the critical path. Fixing
it also needs the test's degenerate `[]int{1, 7, 42}` prompt replaced with a realistic one at depth
128 — measured on that prompt, phi3-mini reports α=1.00 where its throughput says optFwd loses 6.8%,
and the 1.5B reports α=0.61 where its throughput says it wins 5.1%; neither reconciles.

**Not fixed, deliberately.** Fixing (2) properly means knowing whether break-even tracks a static
property like sampler share, which is the open question the third-model run addresses. Fixing (1) is
small and self-contained but pointless while the overlap is capped at T ≤ 0.2, where the gate barely
runs. Both should be revisited together if the cap is ever raised.

## G28 · every bench prompt has four unique words, and speculation numbers were measured on them

Found 2026-08-28 while re-checking optFwd. `scripts/prompts.json` is calibrated for token DEPTH —
correct and deliberate for throughput, where decode cost is content-independent — but every entry is
`"Continue this text. the the the ... the"`. Anything whose value depends on what gets GENERATED was
measured on a regime the engine never serves.

**Demonstrated cost on the one case re-run:** the optFwd A/B moved up to **9.4 points** on a single
cell between filler and a 127-token prose paragraph, converting a 5.1% win into a 4.3% loss and
moving a model's break-even temperature from 0.95 to 0.37. That artifact was the entire case for an
adaptive gate (spec/10), which it retired.

**Unaudited and in scope**, all content-dependent:
1. **optFwd's `EnableAt = 0.90`** — sourced to a "90.9% worst-case break-even on qwen2.5-coder-0.5b",
   plausibly measured on this prompt set, since it is the only one in the repo (see G27).
2. **02's n-gram drafter, "wins on copy-heavy traffic"** — a claim about content, tested on a prompt
   that is maximally copy-heavy. Direction of the bias is obvious and favourable to the drafter.
3. **05/06 EAGLE acceptance (~1.6 tok/verify)**, and anything derived from it — including 09's Gate 1
   comparison, though 09 ran its own suite prompts rather than this file.

**The rule, cheap to apply going forward: a prompt file calibrated for DEPTH may be used for
THROUGHPUT, never for ACCEPTANCE.** Now recorded in benchmarks.md Methodology.

**Not chased here.** Re-measuring (2) and (3) is real work and neither currently gates a decision;
optFwd was re-measured only because it had just shipped a behaviour change.

## G29 · gate-runner reachability — AUDITED AND CLOSED 2026-08-28, no work needed

Carried from the untracked July audit as "census, heavy, composition, selector, mutation are
unchecked for reachability; mechanical, ~1 hour, the pattern is already written
(`TestRealckptCellCanReachEveryGate`)". **Audited instead of implemented, and the premise does not
hold.**

**The specific bug — a `-run` filter that cannot match a required gate — needs a filter to exist.**
Only three non-test files in `cmd/gate` carry `Run:` at all: `parity.go`, `gpu.go`, `configs.go`.
Census, composition, selector and mutation narrow nothing, so nothing can be unreachable by filter
there. Copying the parity test to them would assert a property they cannot violate.

**The GENERAL failure — "ran nothing, reported green" — is already covered everywhere, by five
different mechanisms. That is why it reads as unaudited:**

| runner | guard |
|---|---|
| parity | static: `TestRealckptCellCanReachEveryGate`, `TestBaseCellIsUnfiltered` |
| gpu | runtime: `noteIfEmpty`, with `TestGPUGate_emptyFilteredCellIsNotAPass` |
| heavy | `ZeroPolicy: "no-pass"` — zero PASSES is red even if tests ran and skipped |
| census | `ZeroPolicy: "no-tests"` — zero test events is red |
| composition | `len(parityGates) == 0` → "composition derived from nothing" |
| selector | `len(selS) == 0`, `len(rows) == 0` |
| mutation | baseline-must-be-green, plus asserting the mutation actually CHANGED something |

**`mutation.go`'s header is the best statement of this bug class in the repo** and records two prior
instances *inside the mechanism built to prevent it*: a `command -v staticcheck && staticcheck …`
whose `&&` short-circuited so the check "evaluated to nothing, and it was reported as clean", and a
`… | head -3; echo "exit=$?"` that read `head`'s status instead of the lint's. Its own words:
**"G-01 inside the mechanism built to prevent G-01."**

**A third instance landed the same day, on me** — `~/go/bin/staticcheck` was version-skewed against
the Go 1.27 toolchain, analysed NOTHING, and exited; a U1000 then held CI red across three pushes.
Same shape as that first bullet, years apart, in a different tool. The countermeasure is in CLAUDE.md
(run CI's pinned `go run …@v0.8.0`, and check CI after pushing).

**The real gap was legibility, not coverage: nobody could tell the property was covered without
redoing this audit.** That is what this entry fixes. **Closed — no code change.**

## G30 · the g4x2 clear's sync is REDUNDANT, not expensive — negative result, change NOT shipped

`docs/ollama-chase.md` lists, as a verified small win: *"CUDA g4x2 accumulator clear: H2D per MoE
layer per token … An on-stream memset/zero kernel removes an H2D (and its implicit null-stream sync)
per MoE layer per token."* **Built the cheapest form of that and measured it. It buys nothing.**

**What was built** (kept only in a scratchpad; reverted, not committed): replace the
`r.stream.Sync()` + `gpu.Upload(r.g4x2, r.g4zero)` pair in `layerTail` with a single
`r.stream.UploadAsync` from a pinned zero buffer. Same bytes, issued ON `r.stream`, so stream order
supplies the guarantee the Sync was buying and the drain becomes unnecessary.

**Measured on the 26B (`TestGemma4_26B_cache_B`), paired same-session, load-sampled every 15s:**

| arm | slots | ms/tok | tok/s | hits/misses | continuation |
|---|---|---|---|---|---|
| baseline | 48→30 | 65 | 15.44 | 16611/5229 | — |
| g4zero | 48→30 | 64 | **15.51 (+0.45%)** | 16611/5229 | **identical** |
| baseline | 16 | 86 | 11.61 | 12524/9316 | — |
| g4zero | 16 | 86 | **11.67 (+0.52%)** | 12524/9316 | **identical** |

**WHY IT BUYS NOTHING, which is the useful part.** In the cached configuration the clear is
immediately followed by `loadRoutedExperts`, which performs its own **Sync → D2H routing indices →
H2D expert misses**. Removing the clear's drain only relocates the stall into the next one. The
audit item is factually correct — there IS an H2D and an implicit sync per MoE layer per token — and
economically empty, because a second drain follows it unconditionally.

**Not shipped.** It trades away an `r.stream.Sync()` that guards a documented data race (audit R-03:
the null-stream DMA landing mid-`segC(l-1)` and zeroing the previous layer's expert contribution) for
0.5%, which is noise. Correctness risk for no gain.

**A CAUTION ABOUT THE FIRST MEASUREMENT, because it nearly shipped this.** An earlier unpaired-ish run
showed **+6.2%** (15.27 → 16.22) and I began writing it up as a win. It did not reproduce: two clean
load-audited pairs give +0.45% and +0.52%. The outlier was the g4zero arm, not a contaminated
baseline as I first guessed. A `go test` records no machine state — unlike `bench_peer` — so the
first run was unauditable after the fact, which is why the rerun added a 15-second load sampler.
**Two paired runs and per-arm load sampling turned a 6.2% "win" into noise.**

**WHERE THE EFFORT SHOULD GO INSTEAD.** The same audit entry names the real target one bullet down:
`loadRoutedExperts`'s Sync → D2H → H2D round trip, which is the drain that absorbs everything. That
is the item the standing Metal verdict addresses — *synchronous MoE paging is dead and speculative
prefetch is the path* — and it is a substantially larger piece of work than an on-stream memset.

## G31 · the C′ round trip PRICED: the sync is 0.4%, the expert DMA is 45–60% of decode

Measured 2026-08-28 on the 26B (`TestGemma4_26B_cache_B`, `GOINFER_MOE_CACHE_PROF=1`, load-checked
starts). **`cacheProf` had existed and been read by NOTHING** — `CacheProfForTest()` had no callers,
so the instrument was built and never wired. Raw `docs/measurements/g31-cprime-roundtrip.log`.

| | 30 slots (48 req) | 16 slots | ratio |
|---|---|---|---|
| **stall** (the `Sync` at `resident.go:887`) | 15 ms (**0.4%**) | 15 ms (**0.3%**) | 1.00 |
| **host** (slot bookkeeping) | 107 ms (2.7%) | 108 ms (2.0%) | 1.01 |
| **dma** (expert transfers) | 1.808 s (**45.1%**) | 3.227 s (**59.9%**) | **1.785** |
| misses | 5229 | 9316 | **1.782** |
| per token | 30.14 ms of 62.60 | 52.35 ms of 84.17 | |

**THE SYNC IS 0.4% OF DECODE.** That is the whole explanation for G30's null result, and it retires
the standing audit item: *"an on-stream memset/zero kernel removes an H2D (and its implicit
null-stream sync) per MoE layer per token"* targets **0.4%**, so no implementation of it — memset,
`UploadAsync`, or the new `CopyDevice` from aikit v0.31.0 — can pay. **Stop trying to remove syncs
from this path.**

**THE COST IS THE DMA, AND IT OBEYS AN EXACT LAW.** DMA time scales 1.785× against a 1.782× miss
increase: **346 µs per expert miss in both configurations**, to three digits. Stall and host are
constant because calls (2730 = ~43 MoE layers × 64 tokens) do not depend on slot count.

    per-token ms  ≈  32.1 (compute)  +  1.9 (round-trip overhead)  +  0.346 × misses_per_token

**Compute is constant at 32.5 / 31.8 ms across the two configurations, as it must be — so EVERY
tok/s difference between slot counts is DMA, nothing else.** That also re-explains the ctx-vs-slots
result: more slots raise the hit rate, which buys throughput purely by removing 346 µs transfers.

### What this says about speculative prefetch

**The prize is now quantified rather than inherited.** If the expert DMA were fully hidden behind
compute, token time would fall to `max(compute, dma)`:

| config | today | perfect overlap | ceiling |
|---|---|---|---|
| 30 slots | 15.97 tok/s | 30.81 tok/s | **+93%** |
| 16 slots | 11.88 tok/s | 19.10 tok/s | **+61%** |

**Perfect overlap is unattainable** — routing for layer L+1 is not known until layer L computes,
which is the serial dependency that makes the paging synchronous in the first place, and the reason
the Metal verdict says *speculative* prefetch rather than plain async. But the gap between 30 ms of
transfer and 32 ms of compute per token is close to ideal for hiding: there is almost exactly enough
compute to cover the transfers, if the transfers can be started early enough.

**This is the first quantitative case for that work.** Everything before it was a standing verdict
carried from Metal. The scoping should now be judged against a measured ~1.9× ceiling at 30 slots
and a 346 µs-per-miss cost model, not against "misses are on the critical path".

## G32 · Phase 0: the expert DMA is BANDWIDTH-bound at line rate — batching is dead, overlap is the only lever

Measured 2026-08-28, 26B at 30 slots, `GOINFER_MOE_CACHE_PROF=1`. Raw
`docs/measurements/g31-phase0-upload-split.log`. Each miss issues **four** blocking null-stream
uploads (weights + scales, for `expGU` and `expDown`), so the aggregate rate hides two populations:

| | calls | bytes | time | rate | per call |
|---|---|---|---|---|---|
| **weights** | 10458 | 15.55 GB | 1.427 s | **10.90 GB/s** | 136 µs over 1.49 MB |
| **scales** | 10458 | 1.94 GB | 0.384 s | 5.06 GB/s | 37 µs over 186 KB |

**THE WEIGHT COPIES RUN AT LINE RATE.** 10.90 GB/s is PCIe 3.0 x16 practical throughput on this
board. They are not overhead-bound, and no batching, pinning or stream trick makes them faster.

**This refutes the Phase 0 hypothesis, which was mine.** The aggregate figure from G31 — 346 µs per
miss over 2.40 MB = 6.95 GB/s — looked like a ~40% overhead gap. It is not: it is a weighted average
of large efficient copies (89% of bytes, at line rate) and small inefficient ones (11% of bytes, at
half rate). **An aggregate rate over a mixed size distribution is not a rate.**

**Batching is therefore nearly worthless.** Bringing the scale copies up to the weight copies' rate
saves 206 ms of 1814 ms DMA — **11.3% of DMA, 5.2% of decode**, and that is the ceiling, not an
estimate. Phase 1 as proposed is dropped.

**THE PATH MOVES 273 MB PER TOKEN**, sustained 4.44 GB/s across the whole decode, at a link that
tops out near 11. The volume is the cost, and it is irreducible at this hit rate.

### What survives, and the constraint that changes the prefetch design

Two levers remain, and only one is large:

1. **Move less** — a higher hit rate. Slots are VRAM-capped at 30 of a requested 48 on an 8 GB card,
   and §B4.1's slot sweep was already saturating. Little headroom.
2. **Overlap** — hide ~30 ms of transfer behind ~32 ms of compute. This is the whole prize (G31's
   ~1.9x ceiling), and it is now the ONLY large lever, because the transfers cannot be made faster
   and the volume cannot be much reduced.

**A CONSTRAINT ON SPECULATION THAT THE METAL VERDICT DOES NOT CARRY: THE LINK IS SATURATED.** A
mispredicted prefetch consumes the exact resource that is the bottleneck. That is unlike speculative
COMPUTE, where a wrong guess wastes idle cycles — here a wrong guess displaces a transfer that real
work is waiting on. **Prediction accuracy is therefore a kill gate, not a tuning parameter**, and a
prefetch scheme that is right 60% of the time may be a net LOSS rather than a smaller win.

**And the obvious predictor is already spent.** The Metal note proposes prefetching *last token's
experts*; a 30-slot LRU cache retaining 8 routed experts per layer already IS a last-token predictor,
and its hit rate is the 76%. **The 24% that miss are precisely the experts a last-token predictor
gets wrong.** Any CUDA prefetch must predict NEW routing — e.g. running layer L's router early on an
approximate residual — which is a different and harder proposition than the verdict implies.

**Next step, before any prefetch code: measure whether routing is predictable at all.** Offline, from
captured routing indices — how often does a router run on the pre-MoE residual agree with the true
top-8? That number, against the saturated-link constraint above, decides whether Phase 2 exists.

## G33 · SCOPE — is expert routing predictable at all? (decides whether prefetch exists)

Scoped 2026-08-28 off G31/G32. **No code written.** G32 left overlap as the only large lever
(~1.9x ceiling) and named two constraints the inherited Metal verdict does not carry. This scopes
the measurement that decides whether any prefetch scheme can work on CUDA.

### The structural fact that shapes everything

**Each layer owns its own bank of 128 experts** (`expGU`/`expDown` are fields on `cudaLayer`).
Expert 37 at layer 5 is a different tensor from expert 37 at layer 6. **Therefore cross-layer
prediction is meaningless** — there is no "the same expert" to prefetch — and the only predictors
that can exist are TEMPORAL: same layer, across tokens.

**But the 30-slot LRU cache IS a temporal predictor**, and its accuracy is the measured 76.1% hit
rate. So a prefetcher built on temporal correlation is competing with the cache using the same
signal, and can only win where the cache's policy — not its signal — falls short.

### Tier 1 — trace-only, and it may end the whole item

**The trace is free.** `GOINFER_G4_CAPTURE=1` already fills `g4capIdx [][]uint32` in "APPEND order
(token-outer, layer-inner)". Only a dump-to-file is missing (~10 lines in the test). One capture run
on the box, then all analysis is offline Python.

**Step 1 — validate the simulator before trusting it.** Replay the trace through a 30-slot per-layer
LRU and check it reproduces the measured **16611 hits / 5229 misses / 76.1%**. If the simulation does
not match, the model of the cache is wrong and nothing downstream means anything. This is the
do-nothing arm for the analysis itself.

**Step 2 — decompose the 5229 misses into COLD vs EVICTED.**

- **EVICTED**: the expert WAS used at that layer within the last K tokens, and the cache pushed it
  out. A better policy (or more slots) recovers it. Prefetch is then cache-policy work.
- **COLD**: the expert has not been used at that layer recently at all. **No temporal predictor can
  see it coming**, because there is no signal to predict from.

**PRE-REGISTERED READING, fixed before the run:**

| result | conclusion |
|---|---|
| mostly EVICTED | The signal exists and the policy wastes it. Try policy first (LFU, frequency-aware admission, layer-aware slot allocation) — cheaper and safer than speculation. |
| mostly COLD | **No temporal scheme can help, and that includes every prefetcher built on the routing history.** Prefetch as the Metal verdict describes it is dead on CUDA. Only Tier 2 remains. |
| split | Report the split and size each half against the 346 µs/miss cost model before choosing. |

### Tier 2 — early-router speculation, only if Tier 1 says COLD dominates

The only way to anticipate a cold expert is to COMPUTE the routing early, not predict it: run
layer L's router on an approximation of its input (the residual before layer L-1's MoE contribution
is joined), start the transfers, and correct on mismatch. This needs new instrumentation — the
router is small (hidden x 128), so the compute is cheap; the question is purely whether the
approximate input yields the same top-8.

**THE PRECISION GATE, and it is unusually strict because the link is SATURATED (G32).** A mispredicted
prefetch spends the bottleneck resource itself. With gain ≈ p·(transfer hidden) and cost ≈
(1−p)·(transfer wasted), and both at the same 346 µs, break-even sits near **p = 0.5 before any
discount for imperfect overlap** — so the real bar is higher, call it **p > 0.6**. A 60%-accurate
scheme that would be a clear win for speculative COMPUTE can be a net LOSS here.

**Lead-time requirement, for sizing:** a layer's transfers are ~1.9 misses x 346 µs ≈ 660 µs against
~755 µs of per-layer compute (32.5 ms / 43 layers). **Roughly one full layer of lead is needed to
hide a layer's transfers** — which is exactly the horizon the early-router approach would buy, and
no more. There is no slack for a two-layer pipeline.

### Cost

Tier 1: ~10 lines of dump code, one ~5 min capture run, then offline analysis. **Tier 1 is cheap
enough that it should happen before any further CUDA work on this path**, because a COLD-dominated
answer retires the largest remaining item on the page.

## G33 RESULT · prefetch is aimed at the wrong 10% — the misses are CAPACITY, not prediction

Tier 1 ran 2026-08-28. Trace `docs/measurements/g33-routing-trace.json` (2730 decisions, 30 MoE
layers x 91 positions = 27 prompt + 64 generated), replay `scripts/g33_replay.py`.

**VALIDATION GATE PASSED EXACTLY.** The offline per-layer LRU replay reproduces **16611 hits / 5229
misses / 76.1%** — the measured hardware figures, to the token. The cache is a plain per-layer LRU
and the model of it is right, so everything below is trustworthy. (Positive-controlled first on
synthetic traces: constant routing → all-cold, no evictions; 40 experts through 16 slots → 87.5%
evicted at age 5.)

**COLD IS A STARTUP ARTIFACT.** The headline split — cold 2239 (42.8%) vs evicted 2990 (57.2%) —
inverts once read against position:

| positions | cold | evicted | cold % |
|---|---|---|---|
| 0–12 | 1232 | 34 | **97.3%** |
| 26–38 | 233 | 481 | 32.6% |
| 52–64 | 106 | 470 | 18.4% |
| 78–90 | 65 | 698 | **8.5%** |

Cold misses are FIRST TOUCH of a (layer, expert) pair. In a 91-position window they cannot amortize;
by the last bucket they are 8.5%. **In steady state ~90% of misses are EVICTIONS.**

**THE SLOT CURVE, and it saturates at the cold floor:**

| slots | hit rate | misses | DMA/token |
|---|---|---|---|
| 16 | 57.3% | 9316 | 35.4 ms |
| **30 (today, VRAM-capped)** | **76.1%** | **5229** | **19.9 ms** |
| 48 (requested) | 85.5% | 3170 | 12.1 ms |
| 64 | 88.7% | 2466 | 9.4 ms |
| 96 / 128 | 89.7% | **2239** | 8.5 ms |

**2239 is exactly the cold count.** With every expert resident the only misses left are unavoidable
first touch; everything above that floor is capacity.

### The conclusion, which retires Tier 2

**Speculation can only attack COLD misses — and those are the floor, already the small half in steady
state.** Capacity attacks the ~90% that are evictions. **Prefetch is aimed at the wrong 10%**, and
the early-router scheme scoped in Tier 2 is not worth building: even perfect prediction of new
routing cannot beat simply keeping the expert resident, and it would spend saturated bandwidth to do
worse.

**The lever is VRAM, and it is now priced exactly** at 346 µs per miss avoided:

- 30 → 48 slots: **7.8 ms/token saved, +14% throughput**
- 30 → 64 slots: **10.5 ms/token, +20%**
- beyond 96: nothing, the floor is reached

That makes the existing **ctx-vs-slots trade** (`GOINFER_26B_CTX`) the actionable lever rather than a
curiosity: it converts resident KV into expert slots, and this curve gives the exchange rate. **Cost
a context reduction against the hit-rate curve before spending any effort on prediction.**

**Tier 2 is CLOSED, not parked.** The saturated-link precision gate that worried G32 turns out not to
matter — the question it guarded was never the binding one.
## G34 RESULT · block verify pays, but the SLOT BUDGET caps the width — not acceptance

Scoped and measured 2026-08-31 off the rev-9 framing: *block verify attacks the expert-DMA term by
batching real, already-decided work across tokens rather than guessing routing ahead, so G33's kill
of routing-PREDICTION prefetch does not rule it out.* That framing is correct, and the mechanism is
genuinely different — but the answer is not the one the framing implies.

**Method.** Offline replay of `docs/measurements/g33-routing-trace.json` (91 positions × 30 MoE
layers, topK 8, 30 slots) through G33's per-layer LRU, which reproduces the measured 76.1% hit rate
— that validation gate is why these numbers mean anything. A block verify of width K runs the target
over K draft positions in ONE pass, so at layer L all K positions' routing is computed together and
the **union** is requested at once. Script: `scripts/g34_blockverify_replay.py`.

**Result 1 — batching does NOT reduce misses. It increases them.**

| K | union per layer | fits 30 slots | misses vs serial |
|---|---|---|---|
| 2 | 16 | yes | 1.007× |
| 3 | 24 | yes | 1.017× |
| 4 | **32** | **NO** | 1.026× |
| 6 | 48 | NO | 1.069× |
| 8 | 64 | NO | 1.187× |

**The reason is the same shape as G33's: the predictor is already spent.** A 30-slot LRU already
retains recent experts, so the cross-token reuse block verify would batch is reuse the cache was
*already* converting into hits. Unioning captures nothing new and perturbs eviction. Worse, a block
needs K×topK experts resident **simultaneously** where serial decode needs 8 — so past K=3 the
block's working set exceeds the slot budget and the cache thrashes.

**Result 2 — block verify still pays, via compute amortization, and the width cap is the finding.**
Under G31's model (`32.1 compute + 1.9 overhead + 0.346 × misses`), with compute amortized over α
accepted tokens and DMA scaling with *verified* positions:

| K | α=1.6 | α=2.0 | α=2.5 | α=3.0 |
|---|---|---|---|---|
| serial | | **18.56 tok/s** | | |
| 2 | 21.62 | **27.02** | — | — |
| 3 | 16.90 | 21.13 | **26.41** | 31.69 |
| 4 | 13.84 | 17.30 | 21.63 | 25.96 |
| 6 | 9.90 | 12.38 | 15.47 | 18.57 |

**K=4 at α=2.0 is 17.30 — BELOW the 18.56 serial baseline.** The cliff sits exactly where
K×topK = 32 crosses the 30-slot budget. So the verify width here is capped by **VRAM slots, not by
acceptance** — the opposite of the usual speculative-decoding constraint, and not something the
Metal-derived framing anticipated.

**Where that lands against measured α.** MTP Gate 1 gave 2.02–2.91 (`docs/spec/09-mtp-heads.md`);
EAGLE-3 gave 1.60. So the operating point is **K=2–3**, and both pay: +46% at K=2/α=2.0, +42% at
K=3/α=2.5. Note the DMA per *accepted* token is flat when α ≈ K and rises as α falls below K — the
win is entirely compute amortization, and the DMA term is a drag that grows with the gap between K
and α.

**Limits, stated.** One trace, 64 generated tokens, gemma-4-26b at 30 slots on the 2070S; G31's
constants come from the same box and configuration. Rollback cost on rejection is not modelled. And
the slot budget is the binding constraint, so **a card with more VRAM moves the cliff** — 30 slots
was itself a cap on a requested 48 (G32), which makes this a finding about an 8 GB card rather than
about block verify in general.

**Result 2's compute-amortization case is unverified against a real kernel — there isn't one yet.**
G31's 32.1 ms/token compute figure was measured on ordinary serial decode, one token at a time, and
Result 2 treats it as roughly flat per verify-pass regardless of K. But `cuda/moe.cu`'s kernels are
GEMV, architecturally: `moe_route` is one thread per token by design ("runs ONE thread once per
layer per token"), `gemv_w4a8_moe` takes a single `slot`, not an array, and `prefill_batched.cu`
says outright that moe.ptx is untouched by the batched-prefill work. There is no fused, batched MoE
kernel anywhere in this codebase to run a K-wide verify pass through — and G34 itself is an offline
trace replay, not a CUDA measurement — so the flat-compute assumption behind +46%/+42% has not been
tested against real hardware. External data point, same day: [llama.cpp#27621](https://github.com/ggml-org/llama.cpp/pull/27621)
fused exactly this — MoE router and GLU/up-projection fusion, from nrows==1 to nrows-up-to-8 — for
the identical reason, speculative-decode verify batches on MoE, and it took dedicated kernel work to
get there: 2–22% E2E, gains tapering past width 4. Real, but bounded, not free. If block verify is
ever built, `mmvq.cu` / `topk-moe.cu` in that PR is a working reference for the fusion shape; until
then, Result 2 is a projection, not a measurement.
