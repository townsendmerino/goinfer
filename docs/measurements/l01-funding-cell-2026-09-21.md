# L-01 funding cell, real hardware: KILLED — CPU-offloaded MoE experts permanently poison the C′ cache (~11x regression, not the projected win)

Ran `docs/tasks/task-l01-hybrid-moe-cpu-gpu.md` §0's pre-registered funding measurement (fund >=1.3x end-to-end, park <1.15x, 1.15-1.3x ambiguous), after building §9's stated remainder (goroutine-per-expert
parallel CPU compute alongside the GPU's own async hit-path launches, `cuda/l01_cpu_offload.go`, bit-identical per `TestL01_e2eDecode_matchesBaseline` — race-clean). Qwen3.6-35B-A3B int4 (`qwen3.6-35b-a3b-int4.giw`),
CUDA, RTX 2070 SUPER, driver 595.91.07, `-moe-cache-experts` with `GOINFER_MOE_CACHE_SLOTS=64` (the shipped-optimal slot count per `deltanet-residency-shipped`'s own 15.7 tok/s baseline), greedy, one real prompt,
paired arms (C′ off = the do-nothing arm vs L01 on), committed driver `cuda/l01_funding_cell_test.go`.

## Result: not ambiguous, not close to park — a ~11x regression

| tokens | arm | tok/s | C′ hit rate | misses | token-id hash |
|---:|---|---:|---:|---:|---|
| 128 | off (round 1) | 19.623 | 81.10% | 9,617 | `b58cb1ee5c120f78` |
| 128 | on (round 1) | 1.622 | **0.00%** | 50,880 | `7db490a7cdd7e830` |
| 128 | on (round 2, independent load) | 1.615 | **0.00%** | 50,880 | `7db490a7cdd7e830` (identical to round 1) |
| 32 | off (clean rerun) | 10.989 | 72.37% | 5,570 | `0f58521bef119322` |
| 32 | on (clean rerun) | 1.038 | **0.00%** | 20,160 | `d51b0c06e18ec70a` |

**Ratio: 0.083x at 128 tokens (12.1x slower), 0.094x at 32 tokens (10.6x slower).** §0's own band starts its "ambiguous" zone at 0.87x (1/1.15); this is an order of magnitude below even that. Under this repo's own
"a do-nothing arm can win outright" and "a regression is recorded like a win, not softened" conventions: **KILLED**, not parked. Determinism check: L01-on's two independent 128-token loads produced byte-identical
output token sequences (same hash) — greedy decode really is bit-identical with the flag on, matching `TestL01_e2eDecode_matchesBaseline`; the regression is real work, not noise or drift.

## Root cause, read directly from source, not inferred from the ratio

**The C′ cache can never warm once L01 is on, so 100% of experts route to CPU every token, not the "excess beyond DMA bandwidth" the audit's q* design describes.**

`loadRoutedExperts` (`cuda/resident.go`): for each routed position, `c.admit(e)` provisionally marks a (possibly LRU-evicted) slot as now holding `e` — bookkeeping only, before any bytes move (this is
deliberate, N-09, so a whole layer's DMAs can be issued as one batched upload). On a miss, with `r.l01Enabled`, the code calls `c.unadmit(slot, e)` instead of DMA'ing, and `unadmit`'s own doc comment says exactly
what it does: **"Rolled back to EMPTY rather than to the evicted expert"** — the slot's bookkeeping goes to EMPTY, not back to whatever expert it held before this call. Two compounding effects, both intended
behavior for `unadmit`'s ACTUAL purpose (N-09's upload-failure rollback), misapplied here as the normal case:

1. **`e` itself is never marked resident** (correct: no DMA happened, so it genuinely is not in VRAM) — so it can never hit on a later token either.
2. **The slot's PREVIOUS occupant is also forgotten**, even though nothing was actually evicted from VRAM (no copy overwrote it) — the cache's bookkeeping just stops tracking it. On the very next token, if
   that same old expert routes again, `admit` sees `slotOf[old] == -1` and reports a miss, even though the physical bytes might coincidentally still be sitting there unreferenced.

Every L01-on token therefore leaves every touched slot EMPTY at the end of the layer. The next token's routing finds nothing resident anywhere it touched, so it misses again, everywhere, and the cache never
recovers — hit rate is not degraded, it is **permanently exactly 0%** by the third token onward (the two independent runs agreeing to the token confirms this settles immediately and stays put). This was
never visible in the doc's own §3 microbenchmark, because that measured ONE layer's CPU-vs-GPU cost for a GIVEN miss count `m` in isolation, snapshotting a WARM cache's typical miss distribution — it could not
see the second-order effect that sending 100% of a layer's misses to CPU also erases the cache's ability to ever produce a hit again, which turns "m` missed experts this layer" into "topK missed experts every
layer, forever" — a completely different, unfavourable regime the isolated benchmark's own numbers do not speak to.

## A second, compounding mechanism (plausible from the architecture, NOT independently isolated here)

Even granting the cache-poisoning above, the CPU-compute-vs-DMA comparison itself changes character at 100% offload: **DMA is an asynchronous hardware engine that runs while the GPU's own SMs are busy with
other kernels for the same token; the CPU-compute path is not.** Per-layer, `moeMLPPost`'s hit-path loop launches a GPU kernel per hit position — with 0% hits, it launches ZERO kernels, so there is nothing
async for `l01MergeCPUExperts`'s CPU work to overlap WITH; and the next layer cannot even start (the residual `x` it reads is what this layer's CPU merge is still computing) — so the token's critical path
becomes a strict, unhidden CPU-then-upload-then-launch chain, 40 times over, with the GPU idle for most of it. A back-of-envelope check makes clear this is NOT the whole story either: §3's isolated 272 us/expert,
even run fully sequentially (no parallelism credit at all) across topK=8 experts x 40 layers, is only ~87 ms of added CPU time — nowhere near enough to turn a ~51-91 ms token (19.6-11 tok/s) into a 617-980 ms one
(1.0-1.6 tok/s). Something beyond the lost overlap and the isolated per-expert cost is also at work at full real-decode volume and allocation pressure (`l01ExtractExpertGU/Down` allocate fresh `scratch`/`scales`
slices on every single call, 320+ times per token here) — **not measured or attributed further in this pass**; a `pprof` CPU profile of a full L01-on token would be the next diagnostic step if this is revisited.

## Decision, per §0

**KILLED for the AS-WIRED prototype.** Turning L01 on is a ~11x regression on the real target model, not the 1.3x-plus win the isolated microbenchmark(s) projected. `GOINFER_CUDA_L01_CPU_OFFLOAD` stays default-off
(already true; no user-facing exposure). The goroutine-per-expert parallelization landed this session (`cuda/l01_cpu_offload.go`) is still correct and bit-identical on its own terms — it is a real fix to §9's
"sequential, not parallel" gap — but it cannot rescue a design that sends 100% of every layer's experts to CPU, forever, by construction.

## What a real re-attempt needs (not attempted here — a redesign, not a re-roll)

The audit's own q* design (`docs/tasks/task-l01-hybrid-moe-cpu-gpu.md` §0: "q* ~ m*(B_PCIe/B_host) fetched into slots and run on the GPU, **the rest** computed on CPU") explicitly keeps a bounded fraction on the
NORMAL admit+DMA path so the cache stays warm; the shipped prototype implements q*=0 (route every miss to CPU) unconditionally, which is a different, self-defeating regime this measurement was needed to expose.
A real fix needs, at minimum: (a) still DMA a bounded number of misses per layer normally (keeping `admit`'s bookkeeping intact for them, so the cache actually accumulates residents), and offload only the
remainder past that bound to CPU; (b) a `pprof` profile of one L01-on token to separate the allocation/GC cost from the lost-overlap cost named above; (c) re-run this SAME committed driver
(`cuda/l01_funding_cell_test.go`) once (a) exists — it is ready to reuse, not a new instrument.

## Not established

Whether a correctly-bounded q* split would clear the 1.3x bar at all (the mechanism this measurement killed is different from the one §0 asks about); the Metal pager track (R11c) is untouched by this; P20
(CUDA MoE prefill, the other R11 item) is untouched by this — it is a separate mechanism (batched expert-major GEMM, not CPU offload) and was not attempted this session.
