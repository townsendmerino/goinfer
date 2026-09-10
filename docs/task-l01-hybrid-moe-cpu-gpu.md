# Task: L-01 — hybrid CPU/GPU expert execution for the 8 GB CUDA MoE path (design pass, 2026-09-10)

> **Status: SCOPING ONLY, no CUDA code — but a real isolated Go microbenchmark now exists
> (`decoder/l01_expert_bench_test.go`) and CORRECTS this doc's own earlier headline finding.**
> §2's first-pass derived B_host (1.98 ms/expert, folded in with attention/router cost from a
> whole-decode-step measurement) was **~7× too high**. The isolated benchmark, run on the real
> target hardware (nobara-pc, AMD Ryzen 7 3700X — the CUDA box's actual host CPU), measures a
> single expert at **272 µs**, and — the number that matters — **computing ALL of a layer's
> missed experts on CPU IN PARALLEL (one goroutine per expert) is faster than that layer's own
> current GPU-compute-plus-DMA cost at every miss count from m=1 to m=8**, by 1.32× at m=1
> rising to **3.40× at m=8** (§3, revised). **This reverses §3's original pessimistic
> conclusion and points at a SIMPLER mechanism than the audit's tunable q⋆ split: send every
> miss to CPU, computed in parallel, rather than a smoothly-tuned fraction.** It is still not a
> funding decision — the isolated number does not model the merge-back-to-GPU cost, real
> async concurrency, or the actual target model (Qwen3.6-35B-A3B, not gemma-4-26b's geometry) —
> but the bar this pass needed to clear before recommending a real prototype is now cleared,
> which it was not an hour earlier in this same pass. §7 (speculation-antagonism) was re-run on
> real hardware and still holds as documented.

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

**Derived B_host (rough, NOT an isolated microbenchmark, and — §3 below — ~7× too high):**
1/2.1 tok/s ≈ 476 ms/token total; gemma-4-26b has 30 MoE layers × topK=8 = 240 expert-FFN
computes per token. Dividing (ignoring attention/router/shared-expert share, which is real but
a minority of FLOPs for a MoE model) gives ≈1.98 ms per expert. **Keeping this in the record
rather than deleting it: it is why this pass initially concluded L-01 likely loses, and the
isolated measurement below is what caught the error** — folding whole-decode overhead into a
per-expert rate over-counted badly, exactly the failure mode an isolated benchmark exists to
catch.

## 3. The isolated microbenchmark (2026-09-10) — corrects §2, reverses the framing

`decoder/l01_expert_bench_test.go`, written this pass: synthetic int4 weights via `quantizeWM`
(the SAME `repackW4A8IfEligible`/`maybeF16RoundInt4Scales` chain production loading uses, not a
bare `QuantizeInt4` and not f32 — has to exercise the real W4A8 kernel path or the number means
nothing), gemma-4-26b's REAL geometry probed via `Model.Dims()`/`MoEResidentParams()` on the
actual checkpoint (hidden=2816, expert inter=704, nExperts=128, topK=8, no shared expert). Run
on nobara-pc's real host CPU (AMD Ryzen 7 3700X, the CUDA box's actual target hardware —
not this Mac, which read ~12× faster on the same code and would have been the wrong number to
build against):

| n (missed experts) | sequential (today's pattern) | parallel (1 goroutine/expert) |
|---|---|---|
| 1 | 380 µs (single-expert isolated: 272 µs) | — |
| 2 | 546 µs | 450 µs |
| 4 | 1089 µs | 694 µs |
| 5 | 1656 µs | 892 µs |
| 8 | 2880 µs | 1128 µs |

**Two corrections to §2's rough estimate, both in L-01's favour:**

1. **A single expert costs 272 µs, not 1.98 ms** — the whole-decode-time division folded in
   attention/router/KV/sampling overhead as if it were expert compute. That overhead is real
   but is NOT what a CPU-offload decision is choosing between.
2. **Cross-expert parallelism works, and the gap widens with n** — read `decoder/mlp.go`'s
   `moeMLP` directly: the production loop IS sequential ("the experts run sequentially, so k
   pairs were never simultaneously live"), and each expert's own matmul already claims most of
   the core pool via `parallelCols` — so the a-priori worry (§3 in the version of this doc
   written an hour earlier) was that parallelizing across experts would just divide an
   already-saturated pool with no net win. **Measured, it isn't true**: at n=8, parallel
   (1128 µs) beats sequential (2880 µs) by 2.55×. Nested parallelism (goroutines each internally
   re-parallelizing via `parallelCols`) causes some oversubscription, but nowhere near enough to
   erase the win — there is more slack in an 8-core/16-thread box at this matmul shape than the
   a-priori worry assumed.

**The per-layer comparison that matters** (GPU's own compute per layer ≈1.07 ms from G31;
DMA 346 µs/expert from G31–G33; CPU-parallel from the table above, assuming every miss is sent
to CPU rather than any DMA'd — k=m, not a partial q⋆ split):

| m | today: GPU compute + DMA(m) | CPU-all-parallel, concurrent with GPU compute | speedup |
|---|---|---|---|
| 1 | 1416 µs | max(1070, 272) = 1070 µs | 1.32× |
| 2 | 1762 µs | max(1070, 450) = 1070 µs | 1.65× |
| 4 | 2454 µs | max(1070, 694) = 1070 µs | 2.29× |
| 5 | 2800 µs | max(1070, 892) = 1070 µs | 2.62× |
| 8 | 3838 µs | max(1070, 1128) = 1128 µs | **3.40×** |

**At every measured miss count, sending ALL of it to CPU (computed in parallel) beats today's
GPU-compute-plus-DMA cost — and a partial q⋆ split is not obviously better than sending
everything, because CPU-parallel time stays at or below the GPU's own compute floor (1070 µs)
all the way out to m=8.** This is a materially different, SIMPLER mechanism than the audit's
tunable ratio: "on a miss, compute it on CPU, in parallel with whatever else missed this layer"
— no q⋆ to size at all, at least not for balancing PER-LAYER latency (§5 below revisits what a
q⋆-shaped decision would still be for).

**What this table does NOT model, and why it is not yet a funding decision:**

- **The merge-back cost.** §6 already named this as undesigned; it matters more now that the
  mechanism looks worth building. Getting a GPU-computed partial sum and a CPU-computed partial
  sum combined without a second, unbudgeted host round trip is real engineering, not free.
- **Real concurrency, not an assumption of it.** The table's "max(GPU, CPU)" line assumes the
  CPU-side goroutines are kicked off early enough to run WHILE the GPU is doing its own
  hit-path compute for the same layer, and that nothing else on the host CPU (the routing
  readback itself, other decode-loop work) contends with them. Neither is verified here.
- **Different model, different card.** This is gemma-4-26b's geometry on nobara's CPU;
  L-01's own decision rule is pre-registered against Qwen3.6-35B-A3B, end to end, on the CUDA
  card, paired and interleaved. The isolated number motivates a prototype; it isn't the funding
  measurement.
- **Synthetic random weights**, not real gemma-4-26b tensors — unlikely to matter for a
  compute-bound matmul's timing, but not proven.

## 4. What the microbenchmark leaves open — now a prototype question, not a numbers question

The three-way list this section held before the microbenchmark ("B_host might be lower," "might
need a faster kernel," "might need a narrower trigger") is answered by §3: B_host IS much lower
than first estimated, and a genuinely faster kernel is not even needed — the EXISTING
`swiGLUExpert`/`parallelCols` path, called from separate goroutines, already clears the bar at
every measured m. What remains is not "does the compute clear the bar" but "does a real,
concurrent, merge-including implementation clear it too" — §3's own caveats, and the next step
named in §9.

## 5. q⋆ — still meaningful, but for a different question than §3 answered

§3 found that sending EVERY missed expert to CPU (k=m, no DMA at all for misses) beats today's
path at every measured m. That does not make q⋆ pointless — it reframes what q⋆ would be
choosing between:

- **If CPU has spare capacity beyond what one layer's misses need** (true up to m=8 on this
  8-core/16-thread box, since CPU-parallel time stays ≤ the GPU's own 1070 µs compute floor
  even at m=8), there is no latency reason to send anything to GPU/DMA at all — q⋆=0 is optimal
  for LATENCY.
- **q⋆ would matter for a reason §3's per-layer latency framing doesn't capture: aggregate CPU
  occupancy across a token's 30 layers, and interference with whatever else needs the host CPU**
  (the routing readback, other decode-loop work, a second concurrent request). A design that
  always sends 100% of misses to CPU claims the host for ~1.1 ms × 30 layers ≈ 33 ms/token in
  the worst case (all layers at m=8) — real host-CPU budget that competes with everything else
  the process does, not free just because it overlaps ONE layer's GPU compute. This is exactly
  the kind of aggregate cost the per-layer table in §3 cannot see, and it's the reason a q⋆-style
  throttle (send only what's needed to stay within GPU's own compute window, no more) could
  still beat "send everything" once host-CPU contention with OTHER work is accounted for —
  untested here.

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

**Written before §3's microbenchmark corrected the per-layer picture — re-read with that
correction applied.** §3 found a win at EVERY measured m (1.32× at m=1 up to 3.40× at m=8), not
just a high-m tail, so this distribution does NOT settle things in favour of a threshold gate
over "always offload." What it DOES settle: **where the ABSOLUTE savings concentrate.** At m=1
(25.4% of decisions) the win is 346 µs/layer; at m=8 (1.6% of decisions) it's 2710 µs/layer —
nearly 8× the per-decision saving, on a rarer event. Weighting §3's per-m savings by §8's own
frequencies (m=3/6/7 linearly interpolated between the measured points, since the benchmark
didn't cover them): **mean per-decision saving ≈ 662 µs**, and m≥5 (10.1% of decisions) alone
accounts for ≈31% of the total aggregate saving across all decisions —
**most of the aggregate win still comes disproportionately from the tail even though the low-m
cases are not worthless** (m≤1 decisions are 50.7% of the total but contribute ≈13% of the
aggregate saving). Relevant to §5's aggregate-host-occupancy question: a throttle that skips the
cheap, common, low-value cases and only engages CPU for m≥ some threshold trades a small amount
of the aggregate win for a large cut in how often the host CPU is claimed at all, which §5 named
as the thing "send everything" does not account for.

## 9. Recommended next step

**Still not CUDA code**, but the microbenchmark that was this section's top item is now done
(§3) and reversed the picture it was checking. What remains, in order:

1. **Model host-CPU aggregate occupancy across a full token** (§5), not just one layer's
   latency — a real cost if "send everything" claims the CPU for ~30 layers' worth of parallel
   expert compute every token, competing with the routing readback and anything else on the
   host. This is arithmetic on already-measured numbers (§3's per-m costs × §8's frequencies
   × 30 layers), not a new hardware run, and is the natural tie-breaker between "send
   everything" and a throttled/threshold variant.
2. **Design the merge path** (§6) concretely enough to estimate its own cost — the one piece of
   §3's per-layer table that is currently assumed free.
3. **A real concurrent prototype** — not a synchronous isolated benchmark — that actually
   overlaps CPU expert compute with GPU hit-path compute for the SAME layer and measures
   wall-clock, not two separate numbers added by hand. This is where CUDA/async work starts
   being unavoidable, and it should happen on Qwen3.6-35B-A3B (the audit's own decision-rule
   model), not gemma-4-26b (this pass's stand-in for geometry convenience).
4. Only then: the audit's pre-registered, paired-and-interleaved measurement on the real card
   per §0's decision rule (fund ≥1.3×, park <1.15×).

## Sources

`docs/audit-2026-09-02.md` L-01 (mechanism, decision rule, disposition) · `docs/task-freetoken-techniques.md`
Lead 5 (architecture, antagonism risk) · `docs/QUEUE.md` G31–G34 (DMA cost law, overlap ceiling,
miss classification, block-verify's own infra-gap finding) · `decoder/mlp.go` (`moeMLP`,
`swiGLUExpert` — the existing sequential CPU expert loop) · aikit `linalg.MatmulBT`/`parallelCols`
(the existing per-expert core-width parallelism) · `internal/fitcmd/fit.go` (`-measure`, used
as-is for §2's new number) · `docs/measurements/g33-routing-trace.json` (re-read for its
per-decision distribution, §8) · `cuda/spec_pager_interaction_test.go` (re-run on real hardware
for §7's confirmation).
