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
> funding decision — real async concurrency and the actual target model (Qwen3.6-35B-A3B, not
> gemma-4-26b's geometry) are still unverified — but the bar this pass needed to clear before
> recommending a real prototype is now cleared, which it was not an hour earlier in this same
> pass. §7 (speculation-antagonism) was re-run on real hardware and still holds as documented.
> **§5's aggregate-occupancy question is resolved** (expected case: 35% of the GPU's own
> per-token compute budget) **and so is multi-tenant contention** (measured, real hardware:
> degrades gracefully — the realistic mean case is untouched by a second concurrent stream;
> only the rare worst-case tail erodes, down to breakeven at three simultaneous worst cases,
> never a net loss in what was measured). **§6's merge path is sketched and verified against
> the real CUDA kernel signatures** — no kernel rewrite needed, estimated single-digit-µs cost.
> Every item this pass could resolve without writing CUDA is now resolved. **First real code
> landed 2026-09-10** (`cuda/l01_cpu_offload.go`): the pinned-host extraction (unpermute +
> f16-scale decode + `linalg.WrapInt4`), CORRECTNESS-VERIFIED against an independent ground
> truth on real hardware (cosine 0.99964–0.9999999 across nine expert/input combinations,
> `testdata/qwen35-tiny` — the same family as the audit's own decision-rule model) — but NOT
> yet wired into `loadRoutedExperts`. §9 has the concrete remainder: async overlap, the merge
> kernel, then the real prototype and funding measurement.

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

## 5. q⋆ / aggregate host-CPU occupancy — RESOLVED 2026-09-10 in favour of "send everything"

§3 found that sending EVERY missed expert to CPU (k=m, no DMA at all for misses) beats today's
path at every measured m. That left one open question — whether "always offload" claims too
much AGGREGATE host CPU across a token's 30 layers, in which case a q⋆-style throttle (send
only enough to stay within GPU's own per-layer compute window) might still win once
host-CPU contention with other work is priced in. Answered this pass, arithmetic only, on
already-measured numbers (§3's per-m CPU-parallel costs weighted by §8's own miss-count
frequencies, no new hardware run):

| | value |
|---|---|
| expected CPU-parallel time, per layer, weighted by measured miss frequency | 377 µs |
| **expected CPU-parallel time, per TOKEN (× 30 layers)** | **11.3 ms** |
| GPU's own compute budget, per token (G31, unchanged by any of this) | 32.1 ms |
| **CPU aggregate as a share of GPU's own budget** | **35%** |
| worst case: every one of a token's 30 layers at m=8 (astronomically unlikely — P(m=8) alone is 1.6% per layer, so all 30 landing there in one token is not a real scenario, but it bounds the pathological case) | 33.8 ms — barely exceeds the 32.1 ms budget, not a large overrun |

**In the typical case, "always offload" claims barely a third of the CPU time the GPU's own
compute window already provides — there is no aggregate-occupancy problem to throttle against,
and even the pathological worst case only marginally exceeds the budget.** This resolves the
open question in favour of the SIMPLER mechanism: q⋆=0 (send everything, throttle nothing) is
not just latency-optimal per layer (§3) but also aggregate-occupancy-safe per token, for a
SINGLE decode stream.

**Multi-tenant contention — measured 2026-09-10, `decoder/l01_expert_bench_test.go`'s
`l01Concurrent`, real hardware (nobara-pc, same box, re-confirmed clean after waiting out an
unrelated heavy test another session was running).** Runs N independent "streams" worth of
missed-expert goroutines at the exact same instant and times the wall-clock for all of them to
finish — the tail latency either stream would actually feel under real concurrent decode:

| scenario | solo (1 stream) | 2 concurrent streams | 3 concurrent streams |
|---|---|---|---|
| m=8 (worst case, 1.6% of decisions): hybrid speedup vs today | 3.46× | 1.56× | 1.00× (breakeven) |
| m=2 (near the measured mean, 1.915): hybrid speedup vs today | 1.65× | **1.65× (unchanged)** | not measured |

**At the realistic mean case, contention doesn't erode the win at all** — even degraded ~1.9×
by a second concurrent stream, CPU-parallel time (925 µs) stays under the GPU's own 1070 µs
compute floor, so the per-layer comparison in §3 is completely unaffected. **Only at the rare
worst-case tail (m=8, and specifically multiple decode streams landing on it at the SAME
instant) does contention meaningfully erode the advantage** — down to a bare breakeven at three
simultaneous worst-case streams, never a net loss in what was measured. Two simultaneous m=8
misses across independent streams is already a coincidence (1.6% × 1.6% if independent); three
at once is not a scenario worth designing against. **This resolves the multi-tenant question
this pass could measure without a full serving harness**: "always offload" degrades gracefully,
not catastrophically, under realistic concurrent load. What it does NOT cover: contention from
non-expert-compute host work (the routing readback itself, sampling, orchestration) running
alongside — those weren't in this benchmark's loop and stay a real, smaller, unmeasured factor.

## 6. Partial-sum merge — sketched concretely, verified against the real kernel signatures

§4/§5 resolved (compute clears the latency bar per layer, and the aggregate-occupancy bar per
token), so this is no longer "design once the above resolves" — it's the next real gap. Read
`cuda/resident.go`/`cuda/moe.cu` directly to check the sketch below is buildable, not assumed:

**The hook point already exists and needs no new readback.** `loadRoutedExperts` (`cuda/resident.go`)
already does a device→host download of `r.rIdx` (the router's topK expert indices) once per MoE
layer per token — "the only host round trip on the decode path" (its own comment) — then checks
each against the layer's LRU slot cache, classifying hits vs. misses BEFORE any DMA is issued.
**This classification point is exactly where L-01 would branch**: instead of queuing a miss for
DMA admission, hand its expert id to a CPU goroutine (the pinned host copy C′ already holds).

**Excluding a CPU-handled expert from the GPU's own accumulation needs no kernel change.**
`gemv_w4a8_moe_wacc` (`cuda/moe.cu:180`) — the "expert combine" kernel, `dst[row] += wgt[slot] *
result` — is dispatched ONE SLOT AT A TIME, in a `for j := 0; j < topK; j++` loop
(`cuda/resident.go:2185`ff, confirmed by reading the actual loop, not assumed). Skipping a slot
CPU is handling is a caller-side `continue` in that loop — zero CUDA changes, because the kernel
was already per-slot, not a fused all-topK-at-once dispatch.

**The merge itself is one small, CONSTANT-cost op, not one-per-expert.** The CPU side should sum
its own weighted contributions locally (`Σ wgt[j] * expert_j_output`, plain float32 addition in
host memory — cheap, and exactly the same order-independent accumulation `moeMLP`'s existing
`out[i] += w*expOut[i]` already does, so GPU-computed and CPU-computed partial sums merge the
identical way `moeMLP` already merges an arbitrary expert subset) BEFORE uploading anything —
producing ONE `[hidden]`-sized vector (2816 floats = 11.26 KB for gemma-4-26b) regardless of how
many experts CPU handled that layer. One async H2D copy of that vector + one trivial add-kernel
(`x[i] += cpu_partial[i]`, single dispatch, hidden=2816 elements) enqueued AFTER it on the SAME
CUDA stream the GPU's own `fMoEWacc` calls already use — no new blocking host sync, because
stream ordering (not a host wait) is what guarantees the add happens after the upload.

**Estimated cost of the merge machinery itself: single-digit microseconds per layer where an
offload happened** — 11.26 KB at G32's own measured 10.90 GB/s line rate is ~1 µs of transfer;
the rest is fixed per-call launch overhead (a small async memcpy + a tiny kernel dispatch),
which on this class of hardware is typically a few µs, not the hundreds-of-µs-to-low-ms this
pass's savings are measured in. **Not independently measured this pass** — a real number needs
an actual CUDA harness, which is exactly §9's next item, not this one.

**The one real risk this sketch surfaces: a stall if CPU compute for a layer's misses finishes
AFTER the GPU would otherwise be ready to move to the next layer.** §3's table shows CPU-parallel
time exceeds GPU's own 1070 µs compute floor only at m=8 (1128 µs, a ~58 µs stall) — for every
m≤7, CPU finishes at or before the GPU's own per-layer compute would, so the upload+add is
already enqueued before it's needed and costs nothing beyond the microseconds above. Bounded,
small, and shrinking as m falls — not the kind of risk that erodes §3's savings materially, but
real and worth confirming against actual CUDA stream semantics rather than this arithmetic.

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
aggregate saving). §5 (updated after this section) resolves the throttle-vs-send-everything
question this distribution motivated: aggregate host-CPU occupancy for a single decode stream
turns out NOT to be the constraint — "send everything" wins there too, for a single stream.

## 9. Recommended next step

Every item this design pass could resolve with arithmetic, an isolated Go benchmark, or reading
the real kernel signatures is done: the isolated microbenchmark (§3), aggregate host-CPU
occupancy (§5), the merge-path sketch (§6), and multi-tenant contention (§5).

**First real code, 2026-09-10: the pinned-host extraction is built and CORRECTNESS-VERIFIED,
still not wired into the decode path.** `cuda/l01_cpu_offload.go` unpermutes an expert's Gate/
Up/Down straight out of C′'s pinned host stack (the SAME bytes the DMA-miss path would
otherwise fetch — `permuteFast`'s exact inverse, round-trip-verified for 2,100,000 sample words)
and hands them to a new `decoder.ComputeExpertMLP` export. `cuda/l01_cpu_offload_test.go`
verified the extraction against an INDEPENDENT ground truth — the same model loaded a second
time on the plain CPU backend, comparing SwiGLU output for the same expert and input across
three experts × three random inputs on `testdata/qwen35-tiny` (the SAME family, `qwen3_5_moe_text`,
as the audit's own Qwen3.6-35B-A3B decision-rule target) — real hardware, nobara-pc: cosine
0.99964–0.9999999, max abs diff 2.5e-7–9.9e-6 across all nine cases. Not bit-identical (this
fixture quantizes independently on each load from an f32 source, unlike a `.giw` bundle's
aliased bytes — a real, small, expected source of noise, not a bug) but far too tight to be the
permutation/field/scale bug this test existed to catch (that would miss by orders of magnitude
more). Caught and fixed one real bug in the process: `r.inter` (dense MLP intermediate) vs
`r.moeInter` (the MoE experts' own, different-sized intermediate) — using the wrong field would
have silently misread every expert's shape.

**Still not wired into `loadRoutedExperts`, and no async/merge code yet.** What remains:

1. **A real concurrent prototype** — not a synchronous isolated benchmark — that actually
   overlaps CPU expert compute with GPU hit-path compute for the SAME layer and measures
   wall-clock, not two separate numbers added by hand. This is where CUDA/async work becomes
   unavoidable, and it should happen on Qwen3.6-35B-A3B (the audit's own decision-rule model),
   not gemma-4-26b (this pass's stand-in for geometry convenience).
2. Only then: the audit's pre-registered, paired-and-interleaved measurement on the real card
   per §0's decision rule (fund ≥1.3×, park <1.15×).

## Sources

`docs/audit-2026-09-02.md` L-01 (mechanism, decision rule, disposition) · `docs/task-freetoken-techniques.md`
Lead 5 (architecture, antagonism risk) · `docs/QUEUE.md` G31–G34 (DMA cost law, overlap ceiling,
miss classification, block-verify's own infra-gap finding) · `decoder/mlp.go` (`moeMLP`,
`swiGLUExpert` — the existing sequential CPU expert loop) · aikit `linalg.MatmulBT`/`parallelCols`
(the existing per-expert core-width parallelism) · `internal/fitcmd/fit.go` (`-measure`, used
as-is for §2's new number) · `docs/measurements/g33-routing-trace.json` (re-read for its
per-decision distribution, §8) · `cuda/spec_pager_interaction_test.go` (re-run on real hardware
for §7's confirmation) · `decoder/l01_expert_bench_test.go` (the isolated microbenchmark and the
multi-tenant `l01Concurrent` benchmark, §3/§5) · `cuda/resident.go` (`loadRoutedExperts`, the
existing routing-readback hook point; the per-slot `fMoEWacc` dispatch loop; `permuteFast`,
`cudaWQ`'s `srcW`/`srcS`/`perExpertW`/`perExpertS`, `cudaLayer`'s `expGU`/`expDown`) ·
`cuda/moe.cu` (`gemv_w4a8_moe_wacc`'s per-slot signature) — both read directly for §6's merge
sketch, not assumed · `cuda/l01_cpu_offload.go`/`decoder/l01_export.go` (the extraction
prototype, §9) · `cuda/l01_cpu_offload_test.go` (the correctness verification, §9) ·
`testdata/qwen35-tiny` (the small same-family fixture the correctness test runs against).
