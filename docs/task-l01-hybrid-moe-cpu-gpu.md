# Task: L-01 — hybrid CPU/GPU expert execution for the 8 GB CUDA MoE path (design pass, 2026-09-10)

> **Status: SCOPING ONLY, no code.** This is the "dedicated future pass" the audit
> (`docs/audit-2026-09-02.md` L-01) called for, done as a design pass per explicit instruction —
> size q⋆ from the already-captured trace, work out the split/merge design, re-check the
> speculation-antagonism risk, write it up, stop before any CUDA. **The headline finding
> reverses the audit's optimistic framing at the MEAN: at today's measured numbers, computing
> even one missed expert on the CPU (using the existing kernel as-is) is slower than that
> layer's own DMA-fetch-everything baseline, so the audit's uniform `q⋆ ≈ m·(B_PCIe/B_host)`
> ratio is the wrong shape.** A same-session follow-up (§8) pulled the miss-count DISTRIBUTION,
> not just the mean, from the already-captured trace: **half of all decisions (m≤1) have nothing
> to gain, but the top ~6.5% (m≥5) are exactly where a saturated PCIe bus (G32) meets a real
> CPU-compute opportunity — so a threshold-gated mechanism, sized against that tail, is the live
> candidate, not a uniform per-miss ratio.** §7's speculation-antagonism risk was re-run on real
> hardware today (not re-derived): the four-gate refusal still holds exactly as documented. This
> is not a rejection of L-01; it is why the audit said "dedicated future pass" rather than "next
> PR," and it now has a sharper target than when that disposition was written.

## 0. What L-01 is, verbatim from the audit

`docs/audit-2026-09-02.md` L-01: *"per layer, the m experts missing from the device cache are
split — q⋆ ≈ m·(B_PCIe/B_host) fetched into slots and run on the GPU, the rest computed on the
CPU from the pinned host copy with the existing W4A8/W8A8 NEON/AVX2 expert kernels, partial sums
merged exactly."* Rationale: `cuda/resident.go`'s C′ path already holds every expert pinned on
the host, already has the CPU expert kernels (P18 made them expert-major), and already pays a
host-visible routing readback per layer — so the architectural objection recorded in
`docs/task-freetoken-techniques.md` Lead 5 is smaller than when written. The lever: with misses
on CPU, token time becomes `max(GPU hits, CPU misses)` instead of `hits + DMA`, and G32 already
found the DMA bandwidth-saturated at line rate — exactly the case where moving work off the bus
beats overlapping it.

**Pre-registered decision rule** (unchanged, restated for this pass): Qwen3.6-35B-A3B int4 on
the 8 GB card, paired and interleaved against the shipped C′ path and against `off`; fund at
**≥1.3× end-to-end**, park below **1.15×**, 1.15–1.3× ambiguous → second mechanism. "The number
to beat is 39.3" — FreeToken's own published tok/s on an RTX 4060 Laptop (a different card, per
`docs/task-freetoken-techniques.md:7`), an external sanity check, not the primary bar.

## 1. What the trace already answers (G33, 2026-08-28 — no new hardware run needed)

`docs/measurements/g33-routing-trace.json` (2730 decisions = 30 MoE layers × 91 positions,
gemma-4-26b, 30 slots) + `scripts/g33_replay.py`, validation gate passed exactly (offline replay
reproduces the measured 16611/5229 hits/misses to the token).

- **In steady state, ~90% of misses are evictions (capacity), not cold/unpredictable.** This
  directly supports L-01's premise — the reason speculative *prefetch* was closed (G33 RESULT)
  is a DIFFERENT reason than why hybrid *compute* stays viable: prefetch fails because it can
  only attack the small cold floor; compute-on-CPU doesn't care whether a miss was predictable,
  it just needs the miss to happen.
- **m (missed experts per layer per token) ≈ 1.915 on average at 30 slots**
  (5229 misses / 2730 decisions, topK=8 requested per decision).
- **DMA cost is an exact law: 346 µs per miss**, cross-validated three ways in this pass (G31's
  original figure, G33's 19.9 ms/token at 30 slots = 1.915 × 30 layers × 0.346 ms, and G34's
  serial baseline 18.56 tok/s = 1/(32.1+1.9+19.9 ms)) — I did not re-measure this; I re-derived
  it from existing numbers and it lines up exactly, which is itself a useful cross-check that
  G31–G34's arithmetic is internally consistent.
- **"Perfect overlap" (hide DMA behind existing GPU compute, no CPU involved) is separately
  already closed as unattainable** (`docs/QUEUE.md` G32 section, "Perfect overlap is
  unattainable — routing for layer L+1 is not known until layer L computes"). This matters for
  scoping L-01: the simpler, cheaper lever is not sitting there unexploited waiting to be taken
  instead of L-01 — it hit a hard serial-dependency wall. L-01 is a genuinely different
  mechanism (WHO computes a miss, not WHEN it's known), so it isn't just reinventing overlap.

## 2. New measurement this pass: gemma-4-26B CPU-only decode rate, real hardware

Used `fit -measure` (`internal/fitcmd/fit.go`, shipped this session, Phase 4) exactly as built —
no new code for this. `~/models/gemma4-26b-int4.giw`, CPU-only build (`demo/chat`), nobara-pc
(16 cores, 62 GB RAM, model fully resident — no paging, unlike the Mac's catastrophic case for
this same model class):

```
(measure) cpu    2.1 tok/s over 32/32 decode steps (includes a 64-token prompt prefill)
```

**Derived B_host (rough, NOT an isolated microbenchmark):** 1/2.1 tok/s ≈ 476 ms/token total;
gemma-4-26b has 30 MoE layers × topK=8 = 240 expert-FFN computes per token. Dividing (ignoring
attention/router/shared-expert share, which is real but a minority of FLOPs for a MoE model)
gives **≈1.98 ms per expert, at full machine width** — i.e. this already reflects whatever
parallelism the CPU kernel gets, not a single-core number. **This estimate needs a real isolated
microbenchmark before it's trusted for a funding decision** (see §5) — it's good enough to
motivate or kill the naive design, not good enough to size a shipped q⋆.

## 3. The finding that reverses the framing: today's CPU expert path has no spare capacity to give

Read `decoder/mlp.go`'s `moeMLP` (the CPU MoE decode path) directly, not assumed:

- **The expert loop is sequential across experts, by design and by comment**: *"The experts run
  sequentially, so k pairs were never simultaneously live"* — the 2-buffer (not 2k-buffer)
  scratch reuse actively depends on this and would need restructuring (per-goroutine scratch) to
  parallelize across experts.
- **But each expert's OWN matmul already parallelizes across cores** — `swiGLUExpert` → `matmul`
  → aikit's `linalg.MatmulBT`, which calls `parallelCols(M*N*K, N, ...)`: a work-stealing pool
  across output columns, gated on a MAC-count threshold. At decode shape (M=1, gemma-4-26b's
  expert intermediate dim), this MAC count is well above any reasonable threshold, so one
  expert's own compute already claims most/all 16 cores.

**Consequence: there is little genuinely idle CPU capacity for a second, concurrently-computed
expert to exploit "for free."** Adding cross-expert parallelism (compute two missed experts on
different cores simultaneously) means dividing the SAME 16-core pool between them, not finding
spare cores — a throughput/latency trade that needs its own measurement, not an assumption.

**The per-LAYER arithmetic (the real constraint, since layers are strictly sequential — layer
L+1 needs layer L's fully-merged output) is unfavorable at today's numbers:**

| quantity | value | source |
|---|---|---|
| per-layer GPU compute (hits) | ≈1.07 ms | 32.1 ms ÷ 30 layers (G31) |
| per-layer DMA cost at m≈1.915 | ≈0.66 ms | 1.915 × 0.346 ms (G33) |
| **one CPU expert, full-width, sequential** | **≈1.98 ms** | this pass, §2 |

**One CPU-computed expert already costs more than that entire layer's current total (GPU
compute + DMA for ALL its misses) combined (1.07+0.66=1.73 ms < 1.98 ms).** At the AVERAGE miss
count (1.915), routing even a single miss to CPU — with the kernel as it exists today — likely
makes that layer slower, not faster, because CPU compute doesn't overlap with same-layer GPU
compute the way the mechanism's name suggests: the layer's output needs BOTH paths done and
merged before the next layer can start, and 1.98 ms already exceeds what GPU alone was going to
cost.

**This is a rough estimate from a derived B_host, not a rejection — it is the reason to measure
before building**, exactly the discipline `CLAUDE.md`'s "measurement discipline" section asks
for ("A kernel win is not an end-to-end win until Amdahl has been paid" — here, the opposite
risk: a plausible-sounding mechanism that a derived estimate says may not even be a kernel win).

## 4. What would have to be true for L-01 to work anyway

Three ways the numbers above could still support a real win, each naming what it would need:

1. **B_host is much lower than 1.98 ms in an isolated, single-expert measurement.** My estimate
   folds in attention/router/shared-expert time as if it were expert compute, which
   over-counts. If those are, say, 30% of the 476 ms/token, real per-expert cost could be closer
   to ~1.4 ms — still above the 0.66 ms per-layer DMA baseline, not a game-changer, but the gap
   matters for exactly how narrow a trigger condition is needed. **First concrete step: an
   isolated Go benchmark of `swiGLUExpert`/`moeMLP` at gemma-4-26b's exact expert geometry,
   synthetic weights (same pattern as `metal/attention_prefill_bench_test.go` used real
   architecture geometry with random data) — no real checkpoint load, seconds not minutes.**
2. **A genuinely faster, lower-latency single-expert CPU kernel** (different from today's
   `parallelCols`-parallel-but-still-~2ms path) — e.g. one tuned for minimum latency at M=1
   rather than the throughput-oriented blocked kernel the decode path shares with everything
   else. This is real, scoped kernel work, not a wiring change, and was not budgeted by the
   audit's one-paragraph mechanism description.
3. **A narrower trigger than "every miss splits":** only route to CPU when a layer's m exceeds
   some threshold where GPU-side DMA queueing/backpressure already dominates (bursty misses,
   not the steady 1.915 average) — meaning most tokens never touch the CPU path at all, and it
   only helps the tail. This changes q⋆'s formula from a smooth ratio to a threshold gate, and
   needs the trace's own miss-count DISTRIBUTION (not just its mean) to size — `g33-routing-trace.json`
   already has this; it was not pulled in this pass and is the next thing to read off it if this
   direction is pursued.

## 5. q⋆ sizing: what the formula needs and what's still missing

`q⋆ ≈ m·(B_PCIe/B_host)` is the low-traffic-CPU-share approximation of the balance point
`q⋆ = m·B_PCIe/(B_host+B_PCIe)` (time-to-fetch q⋆ on GPU ≈ time-to-compute the rest on CPU,
so the two paths, run concurrently, finish together). With this pass's numbers
(B_PCIe⁻¹≈0.346 ms/expert, B_host⁻¹≈1.98 ms/expert, both as PER-EXPERT COSTS not rates — so
B_PCIe/B_host in the formula's own terms is (1/0.346)/(1/1.98) ≈ 5.72):

- q⋆ ≈ m × 5.72/(1+5.72) ≈ 0.85m — i.e. **the balance point already says ~85% of misses should
  stay on GPU/DMA, only ~15% go to CPU** — consistent with §3's per-layer arithmetic (CPU is the
  slower per-unit path here, so the balance naturally leans away from it), and a much more
  modest mechanism than "split roughly half."
- This is NOT the same as §3's per-layer finding that even ONE CPU expert may lose outright —
  the balance-point formula assumes the two paths run perfectly concurrently and finish
  together; §3's finding is that CPU's OWN latency for one expert already exceeds the layer's
  total current budget, i.e. concurrency does not save it because there's nothing on the GPU
  side happening at the same time worth waiting out. **These two views need to be reconciled
  with a real measurement, not two different approximations talking past each other** — which is
  exactly why this pass stops here rather than picking one and building against it.

## 6. Partial-sum merge — the "exactly" in the audit's mechanism

Not designed in detail this pass (gated on §4/§5 resolving first), but named so it isn't a
surprise later: `moeMLP`'s existing weighted-sum accumulation (`out[i] += w * expOut[i]` per
expert, §3 above) already merges an arbitrary subset of experts into one output — GPU-computed
and CPU-computed expert outputs would merge the SAME way, order-independent for f32 addition
modulo reassociation (the repo's own `MatmulBTAcc64` precedent shows where that has mattered:
router-adjacent sums that can flip a near-tie). Merging is not the hard part; getting the
GPU-side partial output back from CUDA to combine with the CPU-side partial output without a
second host round trip (defeating the whole point) is — this needs its own design once the
above is resolved, not before.

## 7. The speculation-antagonism risk — RE-CONFIRMED 2026-09-10, still holds

Ran `cuda/spec_pager_interaction_test.go` (`TestSpecPagerInteraction`) with
`GOINFER_HEAVY_TESTS=1` on real hardware (nobara-pc, Qwen3.6-35B-A3B int4, 64 slots/layer,
today, not re-derived from the doc): **PASS, 217.87s.** Every speculative-decode arm — block
verify at width 4 and 7, n-gram speculation at width 4 and 7 — **DECLINED**, citing "this model
has recurrent state (Mamba-2 / Gated DeltaNet)... that speculative rollback cannot losslessly
restore." Only the `off` arm ran (15.6–17.5 tok/s, 77.3% hit rate, 1.819 misses/stage — this
run's own miss rate, close to but not identical to G33's 1.915 from a different trace/config,
consistent). **The four-gate refusal `docs/task-freetoken-techniques.md` documented is still
exactly what happens today** — nothing large enough to need expert streaming can speculate on
CUDA. This was a direct re-run, not a grep or a re-derivation of the prior finding.

## 8. Miss-count DISTRIBUTION (not just the mean) — pulled 2026-09-10, no new hardware run

`docs/measurements/g33-routing-trace.json` replayed for the full per-decision histogram (not
just G33's own headline mean), same validated LRU model, 30 slots, 2730 decisions:

| m (misses/decision) | decisions | % |
|---|---|---|
| 0 | 692 | 25.3% |
| 1 | 693 | 25.4% |
| 2 | 511 | 18.7% |
| 3 | 353 | 12.9% |
| 4 | 205 | 7.5% |
| 5 | 134 | 4.9% |
| 6 | 72 | 2.6% |
| 7 | 27 | 1.0% |
| 8 (every requested expert missed) | 43 | 1.6% |

mean 1.915 (matches G33), **median 1**, p90=5, p99=8=max.

**This settles §4's open question in favour of a threshold-gated design, not a uniform ratio.**
Half of all decisions (m≤1, 50.7%) are at or below the point where §3 already found CPU offload
loses outright — routing THESE through any hybrid-split logic is pure overhead with no upside.
The audit's `q⋆ ≈ m·(B_PCIe/B_host)` formula applies the SAME ratio uniformly to every decision
regardless of m, which is exactly wrong here: it would try to peel a fraction off of m=1 (where
there is nothing to gain) as readily as off of m=8 (where CPU's fixed ~1.98ms cost, if it could
be shared across several missed experts via cross-expert parallelism, has real headroom against
that decision's current 1.07+8×0.346=3.84ms). The tail worth targeting is small — roughly the
top 10% (m≥5, 6.5% of decisions) — but concentrated exactly where the mechanism's premise (CPU
compute vs. a saturating PCIe bus) is most true, since a bus already asked to move 8 experts in
one layer is precisely where "bandwidth-bound at line rate" (G32) bites hardest.

## 9. Recommended next step

**Not CUDA code.** §7 and §8 above are done (real hardware re-run; existing-trace re-analysis).
What remains, in order:

1. The isolated CPU expert-kernel microbenchmark (§4.1) — cheap, synthetic, answers whether
   B_host≈1.98ms is real or an artifact of folding in attention/router cost into §2's derived
   estimate. Still not done — the natural next step, and now sharper: it specifically needs to
   answer whether computing e.g. 5–8 missed experts with SOME cross-expert parallelism (dividing
   cores across the missed set, not giving each expert the full width) beats today's DMA-only
   cost at exactly those high-m decisions §8 found — not the mean case, which §3 already answered.
2. A scoped kernel-design pass for the threshold-gated mechanism §8 now points at, with its own
   pre-registered, paired-and-interleaved measurement plan on the real Qwen3.6-35B-A3B cell per
   §0's decision rule — sized against the ~6.5% tail (m≥5), not the 93.5% where §3 already says
   there's nothing to gain.

## Sources

`docs/audit-2026-09-02.md` L-01 (mechanism, decision rule, disposition) · `docs/task-freetoken-techniques.md`
Lead 5 (architecture, antagonism risk) · `docs/QUEUE.md` G31–G34 (DMA cost law, overlap ceiling,
miss classification, block-verify's own infra-gap finding) · `decoder/mlp.go` (`moeMLP`,
`swiGLUExpert` — the existing sequential CPU expert loop) · aikit `linalg.MatmulBT`/`parallelCols`
(the existing per-expert core-width parallelism) · `internal/fitcmd/fit.go` (`-measure`, used
as-is for §2's new number) · `docs/measurements/g33-routing-trace.json` (re-read for its
per-decision distribution, §8) · `cuda/spec_pager_interaction_test.go` (re-run on real hardware
for §7's confirmation).
