# Pre-registration — does speculative verify break the expert pager?

**Written before any arm was run.** Box taken 2026-09-02 16:12 (nobara-pc idle: 16 MiB
GPU, load 0.02). Decision rules, thresholds and the ambiguous→parked band below are
fixed as of this commit; the results file records outcomes against them without
editing them.

## Origin of the hypothesis

A Reddit report of a 176B MoE in a hybrid split (experts on CPU, attention on GPU)
claims enabling MTP/n-gram drafting collapsed throughput, with a stated mechanism of a
pipeline stall forcing CPU threads to re-fetch expert matrices from RAM when control
returns from a GPU verify. That post is the origin of the question and nothing else:
n=1, one machine, LLM-narrated, and containing several confabulated technical claims.
**No number from it is used, quoted, or compared against here.** Only the mechanism is
under test.

The specific split it describes does not exist in this tree — Lead 5 (bandwidth-adaptive
CPU/GPU co-execution, `docs/task-freetoken-techniques.md`) is PROPOSED, NOT BUILT — so
what is measured is the generalized mechanism on a path that does exist: a width-K
speculative verify against a model whose routed experts are paged.

## The two hypotheses, which are NOT the same

**H-paging** (the field report's mechanism, generalized). A verify of width K asks for
K positions' routing at once. Different positions route to different experts, so a
verify presents several times as many distinct experts per staging event as decode
ever does. If that exceeds the resident slot budget the pager thrashes. The slot budget
was tuned on decode traffic; nobody has checked it against a verify.

**H-noamort** (raised by reading the code while scoping this, and it must be
distinguished, not assumed away). `cuda/prefill.go:162` declines the batched
weight-stationary path for MoE. If a width-K verify therefore walks position by
position, the pager sees exactly decode-shaped traffic — the slot budget is never
stressed — but the verify pays K full decode steps to commit at most K tokens, so the
amortization speculation depends on is absent. This predicts a regression too, with the
same sign and a similar size, and **wall-clock alone cannot tell it from H-paging.**

Distinguishing them is the point of the demand instrument added for this work
(`cuda/testhooks_gen.go: PagerStageStatsForTest`), which reports staging events and the
distinct experts each one requested. No pre-existing hook reported that quantity;
`CacheStatsForTest` reports hits/misses, which says how much DMA reuse SAVED and not
how much the caller ASKED FOR.

## Venue

| venue | status |
|---|---|
| CUDA expert staging (C′), RTX 2070 SUPER 8 GB, driver 595.91.07 | **measured here** |
| Metal expert streaming, 16 GB MacBook | **NOT RUN** — `ssh franciss-macbook-pro` refused on port 22 (host up on Tailscale, remote login not accepting). Not measured, not inferred. |

Model: `~/models/qwen3.6-35b-a3b-int4.giw` (local NVMe — the archive is never a read
path for a timed row). 40 layers, 256 experts, top-8, ~20 GB of int4 experts against
8 GB of VRAM, so it runs only via C′ paging. Drafter: `~/models/qwen36-35b-dflash`,
pairing verified against the target (vocab 248320 = 248320, num_target_layers 40 = 40,
hidden 2048 = 2048, tap ids max 37 < 40). Greedy, temperature 0.

## Arms

`off` is mandatory and is not a formality: this repo has already found a speculation
suite where no verify width beat running no drafter at all.

1. `off` — no drafter, plain resident greedy decode
2. `K=4` — block drafter, VerifyWidth 4
3. `K=7` — block drafter, VerifyWidth 7
4. slot ladder at K=7 and at `off`, over per-layer slot depths
5. same ladder, lowered end

Arms 4–5 are the diagnostic. **Note a venue asymmetry that changes their shape from the
brief's:** on CUDA the default slot request is "all experts", capped by `capSlots` to
measured free VRAM, so the default IS the ceiling and there is no "raised" rung above
it. The ladder therefore runs `{8 (=topK, the no-reuse degenerate), 16, auto-cap}` and
the question is put as: does the K=7-vs-off penalty DEPEND on slot depth? Each rung
reports the depth actually built (`CacheSlotsForTest`), not the depth requested.

## Prompts

Realistic traffic, 3 prompts spanning prose / code / math, chat-templated.
**`scripts/prompts.json` is not used and must not be:** it is four-unique-word filler
(`"the the the the …"`), and on a MoE every such position routes to the same experts,
so the pager never stages anything and the effect under test cannot appear. Using it
would manufacture a null result. The same confound already produced one wrong profile
in this tree (mellum2 prefill split, no MoE frames at all).

## Denominator

Speculation changes the decode step it would be measured against, so a ratio whose
denominator moves with the arm is contaminated — the failure caught this week, where a
PCIe arm's 7.92 ms decode step was divided by a D2D arm's 5.69 ms with no overlap
across three runs.

**Primary metric is denominator-free: wall-clock seconds to generate a fixed 64 new
tokens, per arm, per prompt.** Where a ratio is unavoidable it is formed PAIRED per
(prompt, repeat) round and then aggregated — never ratio-of-medians, which disagreed
with the paired form by 7.6 pp in one run and under 1 pp in others, and is therefore
uncorrectable after the fact.

3 repeats per (arm, prompt). Spread is reported, not just central tendency: thrashing
announces itself in variance first — that is how the Metal N=128 cliff was found, at a
20.8% spread against ≤3% elsewhere.

## Decision rules — fixed now

**R1 — is the regression real?** Paired per (prompt, repeat) ratio of wall-clock
`K=7 / off`, aggregated as the median of the paired ratios.

| median paired ratio | verdict |
|---|---|
| ≥ 1.10 | REAL REGRESSION (K=7 is ≥10% slower than no drafter) |
| 1.02 – 1.10 | **AMBIGUOUS → PARKED** |
| ≤ 1.02 | no regression; speculation is neutral or a win here |

Sign must agree across all 3 repeats for any verdict other than PARKED.

**R2 — is it H-paging?** Mean distinct experts per staging event, K=7 vs off. The router
picks top-8 distinct, so `off` must read 8.00; a reading materially off 8.00 in the
`off` arm invalidates the instrument rather than the hypothesis, and the run stops.

| distinct/stage at K=7 | verdict |
|---|---|
| ≥ 10.0 (a ≥25% rise over 8.00) | H-paging CONFIRMED: a verify does present multiple positions at once |
| 7.84 – 8.16 (±2% of topK) | H-paging REFUTED on this venue: the pager sees decode-shaped traffic at every K |
| 8.16 – 10.0 | **AMBIGUOUS → PARKED** |

**R3 — is it H-noamort (i.e. "this is alpha/amortization, not paging")?** Realized alpha
is captured per round from `BlockSpecOptions.OnRound(width, committed)`. If verify walks
position by position, a width-K round costs K positions and commits `mean(committed)`
tokens, so the predicted wall-clock ratio is `K / mean(committed)`.

The item CLOSES as "alpha/amortization, not paging" when **all three** hold:
- R2 returns REFUTED (distinct/stage pinned at topK), and
- the measured paired ratio from R1 falls within ±15% of `K / mean(committed)`, and
- pager hit rate and misses-per-staging-event at K=7 are within ±5 pp of `off`.

That conjunction is what makes it a positive finding rather than an absence: the
regression is then quantitatively accounted for by a mechanism that has nothing to do
with the slot budget.

**R4 — config fix or structural?** Relative change in the K=7/off paired ratio across
the slot ladder.

| variation across rungs | verdict |
|---|---|
| ≥ 10% | config fix: N must be tuned per speculation width |
| < 5% | structural: belongs in `docs/spec/` as a venue restriction |
| 5 – 10% | **AMBIGUOUS → PARKED** |

**Two pre-registrations that can disagree, on purpose.** R1 (does it regress?) and R3
(is the regression fully explained without paging?) answer different questions and can
contradict — a real regression whose size does NOT match the amortization prediction
would leave paging live even with R2 refuted. That is the intended check: the Metal slot
sweep is on record as a case where a stopping rule and a full ladder disagreed and the
ladder was right, and the lesson recorded there was to pre-register a second thing that
can contradict the first rather than to write a better single rule.

## Where the result goes either way

Present or absent, the finding is written into `docs/task-freetoken-techniques.md`
Lead 5 as a pre-registered risk: Lead 5 and the speculation program are currently
planned independently and nothing on record says they may be antagonistic. If the
effect is real it additionally goes to `docs/spec/` as a venue restriction beside the
existing rollback-safety refusals.

---

## Addendum, written during the run — design corrections, thresholds untouched

Two corrections were made after the first launch. Both change the APPARATUS, not a
decision rule; R1–R4 above are exactly as first written. They are recorded here rather
than folded silently into the method, because a pre-registration that gets quietly
edited is worth less than no pre-registration.

**1. The slot ladder as first written could not have been read.** The rungs were
`{8, 16, auto}`. `cuda/backend.go` adopts `GOINFER_MOE_CACHE_SLOTS` only under
`req > topK`, so with topK=8 a request of 8 — and an unset value — both leave the
`8*topK = 64` default in place. The ladder would have BUILT `{64, 16, 64}`: two rungs on
one configuration, reported as three, with the collapsed pair sitting at the two ends
where R4 reads its variation. Rungs are now `{16, 32, 64}`, all above topK.

This was caught by the instrument rather than by inspection: `CacheSlotsForTest` reports
the depth actually built, and the first rung logged `64 slots/layer built (requested
"8")`. That accessor exists for exactly this reason and it earned itself on the first
run. A ladder reporting its REQUEST would have produced a clean-looking three-point
result with two points secretly identical. The harness now also logs a loud NOTE
whenever built ≠ requested.

Note a consequence worth keeping: **a slot depth at or below topK is not reachable
through the env var at all.** The topK floor is deliberate (one token's routed set must
be simultaneously resident), so "does the cliff arrive sooner at a lower N" can only be
asked down to topK+1, not below it.

**2. The spread metric pooled across prompts.** `specPagerSpread` was applied to every
timing in an arm at once, which folds between-prompt variance — different lengths,
different routing — into a figure that reads as run-to-run noise, and the prompts here
differ by design. It now reports WITHIN-prompt spread, averaged over prompts. This is
the same rule as differencing matched observations instead of pooling them; the pooled
form would have been dominated by the thing the arms hold constant.

**3. A process collision invalidated the first launch, and its numbers are discarded.**
The first run was not fully killed before the second started, so two processes each held
a ~20 GB pinned allocation on a 62 GB box. Nothing from that launch is used here. It did
produce one artifact that is kept and independently re-measured by the clean run: the
per-request decline text quoted in the results.
