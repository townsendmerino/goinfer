# Task: MoE weight streaming — closing the gap to `turbo-fieldfare`

goinfer already ships bigger-than-RAM weight streaming (`serve --stream-weights` +
`--weight-cache`, ideas #2 and #4 in `docs/ideas-weight-memory.md`, 2026-06-13). This task
is **not** "build streaming." It is the four levers `drumih/turbo-fieldfare` has that
goinfer does not, found by reading its design against ours, plus one hazard that lands
the next time aikit is bumped.

**The headline finding is platform-specific and it is the reason to do this work.** On
darwin, `mmap.Advise(span, false)` is a **deliberate no-op** — macOS will not force a
resident drop on a read-only file-backed mapping (`aikit/mmap/madvise_darwin.go`, verified
empirically on darwin/arm64: `MADV_DONTNEED` and `MADV_FREE` leave RSS unchanged,
`MADV_FREE_REUSABLE` returns `EPERM`, the `msync` variants are no-ops). So on the Mac,
`--weight-cache` is **bookkeeping, not a cap**: `SpanCache` counts bytes and picks victims,
but the actual replacement policy is macOS's Unified Buffer Cache under memory pressure.
The firm cap is Linux-only.

That is precisely the thing fieldfare avoided by not using mmap at all. It `pread`s into
buffers it owns and runs its own 16-slot LFU per layer, so on an 8 GB M2 Air it gets a
*predictable* 5.1–6.3 tok/s. goinfer on the same machine would be at the mercy of the UBC.
Since the Gemma 4 26B-A4B benchmark rig is an M1 Pro 16 GB (`docs/completed/task-gemma4-moe.md`),
this is on the critical path for that number, not a nice-to-have.

---

## What already ships (the substrate — don't rebuild)

| | |
|---|---|
| `expertPager` — router top-k drives eviction over the read-only `.giw` mapping | `decoder/moepaging.go` |
| `layerPager` — dense windowed prefetch, `WILLNEED` L+1 while computing L | `decoder/layerpaging.go` |
| generic span residency + budget | `aikit/mmap.SpanCache`, `mmap.Advise`, `mmap.AutoBudget` |
| bit-exactness by read-only re-fault | `TestMadvise_dontneedRefaultsIntact`, `TestExpertPaging_bitExact`, `TestLayerPaging_bitExact` |
| the demand-signal seam | `moeMLP` top-k → `pager.touch` (`decoder/mlp.go:81`) |
| an access trace + cost model for the real 35B-A3B | `decoder/moepaging_spike_test.go` |

Validated 2026-06-13 on the real Qwen3.6-35B-A3B: 512 MB expert cache against ~16 GB of
experts, `hits=4706 misses=5534 evictions=5190`, decode **byte-identical** to fully
resident over 24 tokens.

## Reference: what fieldfare does

Resident: a **1.35 GB shared core** + FP16 KV. On disk: **14.3 GB** in a custom `.gturbo`
file. Per layer — Metal computes attention and the router from resident weights; the CPU
takes the top-8 expert IDs, plans against a **16-slot LFU cache for that layer**, and fills
misses with **bounded parallel `pread`** into Metal-visible buffers; Metal computes the
**resident shared-expert branch while those reads are in flight**, then combines. Prefill
runs in chunks of ≤128 tokens so one fetched expert serves multiple rows.

Two arithmetic checks say the designs converge and the target is realistic. Gemma 4
26B-A4B's non-expert weights are ~2.4B params ≈ **1.3 GB at int4** — fieldfare's "1.35 GB
core", independently derived. And one expert is `2112 × 2816` = 5.95M params ≈ **3.0 MB at
int4**, so top-8 × 30 layers = **~714 MB/token if every fetch is cold** — at ~3 GB/s that
is ~4 tok/s, right at their 8 GB M2 figure. Nothing exotic is happening; the gap is
engineering, not insight.

---

## Phase 0 — the aikit bump hazard (**DONE** — verified 2026-08-02)

**Status: complete.** goinfer moved to `aikit v1.16.0` here and is on `v1.17.0` as of 2026-08-12
(`go.mod:6`); the durable claim is that both are past the `6c0483f`
scan-resistant-eviction change. aikit shipped the fix as an `EvictPolicy` API
(`mmap/spancache.go`: `EvictMostRecent` default + `EvictLeastRecent`), and
`decoder/moepaging.go` selects `mmap.NewSpanCacheWithPolicy(budget, mmap.EvictLeastRecent)` —
so the ANN scan keeps scan-resistance and the expert pager keeps its frequency-aware LRU. The
regression is reproduced in `moepaging_spike_test.go`'s B0 comparison: on the real 35B-A3B trace
the scan-resistant (MRU) policy is **−51 pp** vs LRU at a 4 GB budget. All three stale artifacts
are fixed (spancache doc, `moepaging.go`, `moepaging_test.go` cite `EvictLeastRecent`). The bump
gate held: the pager's hit rate is unchanged from v1.12.0 (it uses the LRU policy).

*(Original hazard, kept for the record:)* aikit `main` past v1.14.0 contains **`6c0483f`
"mmap: scan-resistant SpanCache eviction"**, which changed `SpanCache.Touch` to evict
`c.lru.Front()` — the **most**-recently-touched other member — instead of `c.lru.Back()`.

That change is correct for the workload it was measured on: aikit's own demand signal is
`FlatI8`'s paged query walking blocks 0,1,2,… every call, the textbook cyclic-scan
pathology where LRU hits **0% even at a 63/64 budget**. Evicting the most-recent pins a
stable prefix and recovers 10.9 / 45.0 / 88.6% at 8/32/63-of-64.

**It is the wrong policy for `expertPager`, whose demand signal is the opposite shape.**
The spike measured skewed *frequency*, not a cycle: the hottest 10% of experts absorb 72%
of accesses, the hottest 25% absorb 94%, and half the universe is never touched. Under
evict-most-recent, a hot expert becomes the victim as soon as anything else is touched
after it; the policy pins the *oldest* prefix rather than the hot set. Nobody will notice
at load time — it will show up as a hit-rate regression on the next aikit bump.

Work:

1. Re-run `decoder/moepaging_spike_test.go`'s trace against both policies and record the
   hit-rate delta. This is offline replay over a recorded trace — no I/O, no model load.
2. If the regression is real (expect it to be), the fix is **policy per demand signal**, not
   one global policy in the shared type: a `SpanCache` policy knob (or a
   `NewSpanCacheWithPolicy`) so the ANN scan keeps scan-resistance and the expert pager
   keeps a frequency-aware policy. That is an aikit change; land it there and bump.
3. Stale artifacts to fix in the same pass: `aikit/mmap/spancache.go`'s type doc still says
   Touch "releases the least-recently-touched members"; `decoder/moepaging.go:15–16`
   describes "the generic span-residency LRU … release the budget tail";
   `decoder/moepaging_test.go:13` cites `TestSpanCache_evictsLRUTailOverBudget`, renamed in
   aikit to `TestSpanCache_evictsMostRecentOverBudget`.

**Gate:** the bump does not land until the expert-pager hit rate on the replayed trace is
≥ its v1.12.0 value.

---

## Lever 1 — an owned-buffer `pread` miss path (the big one)

**Today.** A miss is `Advise(span, WILLNEED)` followed by a synchronous page fault when the
matmul reads the span. That is one fault at a time on the decode thread — queue depth 1 —
and the spike's cost model (`NVMe ≈ 20 µs seek + 3 GB/s`) assumed the *bandwidth*, which a
QD1 fault stream is unlikely to reach on any NVMe device. On darwin it is worse than a
throughput question: with eviction a no-op, there is no cap to enforce at all.

**Two steps, in order.**

**1a — concurrency without changing the data path.** Keep the mapping and the zero-copy
`WeightMat` aliasing exactly as they are; add a small worker pool that faults the missing
spans *concurrently* ahead of the matmul (`Advise WILLNEED` plus an explicit touch-a-byte-
per-page read, since `WILLNEED` is only a hint). Every existing bit-exactness guarantee
holds unchanged — the bytes still come from the same read-only mapping. This is the cheap
half and it should be measured on its own: **if queue depth is the whole story, stop here.**

**1b — owned buffers.** If 1a does not close the gap, or the darwin cap matters more than
zero-copy, move the expert path off mmap: `pread` into pooled, page-aligned buffers the
pager owns, bounded-parallel, with the pager's own residency accounting. This buys a firm
RAM cap on macOS — the thing `madvise_darwin.go` says cannot be had any other way — at the
cost of the `.giw` zero-copy aliasing for expert tensors and a real memcpy per fill.

1b is a genuine architectural change, so gate it on 1a's measurement rather than assuming.
Bit-exactness stays trivially provable either way (the bytes are the same file bytes), and
`TestExpertPaging_bitExact` is the existing gate.

**Deliverable:** a tok/s-and-p99-latency-vs-budget curve on both darwin and linux, for
QD1-mmap / parallel-mmap / parallel-pread. The darwin column is the one that decides 1b.

## Lever 2 — eviction policy matched to the demand signal — **REPLAYED; verdict: keep LRU**

The hypothesis was that fieldfare's **LFU** should beat goinfer's LRU because the skew (10% of
experts absorb 72%) "is an LFU distribution." The offline replay (`moepaging_spike_test.go`,
LRU/MRU/LFU/LFU-aging over the real 35B-A3B trace, 2026-08-02) **falsifies it — do not build a
frequency-aware policy.** The table:

| cacheGB | LRU | MRU | LFU | LFU-aging |
|--:|--:|--:|--:|--:|
| 1.0 | 12.7 | 6.5 | 1.3 | 3.3 |
| 2.0 | 39.2 | 11.8 | 14.0 | 40.5 |
| 3.0 | 51.3 | 15.7 | 35.9 | **60.7** |
| 4.0 | 71.9 | 20.9 | 51.2 | **73.5** |
| 6.0 | 84.3 | 38.7 | 73.9 | 85.3 |
| 8.1 | 89.6 | 57.0 | 86.5 | 89.6 |
| 16.1 | 92.9 | 92.9 | 92.9 | 92.9 |

Two findings. **(1) Plain LFU is strictly WORSE than LRU** (−3 to −25 pp) — the classic LFU
establishment pathology: an evicted-then-re-faulted hot expert restarts at count 1 and is
re-evicted before it re-accumulates, so LFU keeps a stale cold set. **(2) LFU-aging fixes that
and does beat LRU — but only at 3–4 GB budgets** (+9 pp at 3 GB), and at the realistic ≥8 GB
range all three converge to LRU because on a **stationary** skewed signal frequency ⇒ recency:
the hot experts are touched every few tokens, so LRU already keeps them warm. The +9 pp lives in
a budget regime (3–4 GB against 16 GB of experts) nobody runs interactively.

**So LRU (the current `EvictLeastRecent`) is the right policy; Lever 2 is closed with no code.**
This is exactly what "do the replay before writing any policy code" buys — the policy work is
saved. The interaction with Lever 1 stands: on darwin the *replacement decision* belongs to the
UBC anyway (no firm cap), so even a better policy wouldn't bind there; and the replay's verdict
means Lever 1b's motivation is the darwin firm cap alone, not a policy win.

**The principle, so the next person doesn't re-derive it wrong:** *on a stationary skewed signal,
recency is a sufficient statistic for frequency.* "Hottest 10% absorb 72%" looks like an LFU
distribution, but if a hot expert is touched every few tokens then LRU never evicts it, so the
recency ordering already encodes the frequency ordering. LFU only wins when the cache is far
smaller than the hot set — and naive LFU's establishment pathology (a re-faulted hot item
restarting at count 1) means it can also *lose*, which is what the table shows.

**Reconciling with fieldfare (or the record misreads).** This is NOT "fieldfare chose LFU and
LFU is worse." fieldfare runs a **16-slot cache per layer against 128 experts** — a cache-to-
working-set ratio far smaller than goinfer's 8–16 GB budgets, and precisely the tight regime
where this replay shows LFU-aging beating LRU (the 3–4 GB rows). Both results are correct at
their own operating points; **the policy choice is a function of that ratio, not an absolute.**
goinfer streams at generous budgets where LRU is optimal; fieldfare caches at a tight per-layer
budget where a frequency-aware policy pays. If goinfer ever runs at a fieldfare-like ratio (a
firm-capped darwin build under real pressure, Lever 1b), revisit LFU-aging then — the replay
already says where its +9 pp lives.

## Lever 3 — overlap routed reads with the resident branch

`docs/ideas-weight-memory.md` states the cold-miss cost is "bandwidth-bound and
unhideable … the router selects just-in-time, so there's no prefetch lead." True for the
architectures goinfer had. fieldfare found the hiding place: run the **resident** branch
while the routed reads are in flight.

**Gemma 4 26B-A4B hands goinfer the same opportunity for free.** Its FFN sub-block is a
*parallel* dense-MLP + MoE pair (`docs/completed/task-gemma4-moe.md`, Delta 1): the dense branch is
always resident, independent of the routed branch, and its output is simply summed. So the
sequence becomes — route → issue the expert fills (Lever 1's pool) → **compute the dense
branch** → join → combine. On a 30-layer model that is 30 overlap windows per token.

This lever is therefore **sequenced after `task-gemma4-moe.md` phase 2** (the forward), and
it is the one that makes the "unhideable" line in `ideas-weight-memory.md` need an
amendment. Generalizes to any arch with an always-on shared expert (Qwen2-MoE, GLM,
DeepSeek), where the shared expert plays the dense branch's role.

**Measure the right thing:** the win is p99 token latency and the miss-cost coefficient,
not mean tok/s at a budget where everything hits anyway. Run it at a budget that forces a
real miss rate.

## Parked candidate — int32-per-group GEMV (opens IMMA, at a parity-refresh price)

The batched prefill GEMV (`cuda/gemv_w4a8_batched.cu`, milestone 1) is a **weight-stationary
dp4a batched GEMV**, deliberately not IMMA: goinfer's int4 weights are **group-scaled** (one f16
scale per 32-element group along K), so the cross-group sum must be in **float**, and float add is
non-associative — a K-tiled IMMA/MMA reorder would not be bit-identical to the M=1
`gemv_w4a8_fwd`. dp4a is ~1/3 of Turing IMMA int8 throughput, so the kernel is compute-bound above
M≈45 and reaches ~23× on TTFT at M=512 rather than the ~72× an IMMA path could theoretically hit —
a ceiling **chosen, not missed** (recorded in the kernel's own doc comment).

**The unlock, and why it is parked:** change the GEMV to accumulate int32 across a group's four
words and apply the f16 scale **once per group** (int32 associative → K-tileable → IMMA-eligible).
That would match the numeric contract to the quantization structure and open the ~3× IMMA compute
headroom. **Price: it changes decode BITS for every int4 model** (the accumulation reorders), i.e. a
full parity refresh with the goldens re-run across all families, and a `validated_at` bump — the T3
ritual, not a hash refresh. So it is its **own candidate**; it must NOT ride in on a prefill kernel.
Fund it only if the prefill compute ceiling (not the bandwidth win, which dp4a already captures)
becomes the binding constraint after the batched GEMV lands.

## Lever 4 — expert-major prefill batching

`decoder/forwardn.go:38` states it plainly: batched prefill vectorizes attention but "the
MoE FFN itself stays per-row (router picks different experts per token)" — the per-row call
is `decoder/forwardn.go:334`. Under streaming that is the worst case: the same expert can be
fetched once per row.

fieldfare's fix is chunks of ≤128 tokens so one fetched expert serves multiple rows.
The goinfer version: within a prefill chunk, **group rows by selected expert** and run each
fetched expert once over all its rows (a gather → per-expert GEMM → scatter), so fetch
count falls from `rows × k` toward `distinct experts in the chunk`. This is a throughput
win even fully resident (it turns k GEMVs per row into one GEMM per expert), which is what
makes it worth doing independently of the streaming case — and it is the missing half of
the 2.4× batched-MoE-prefill result on Mellum2 (`CHANGELOG` v0.5.0, `08acc11`).

`ideas-weight-memory.md` #4 hedges that streamed mode may need to "cap the streamed mode to
modest prompts." This is the alternative to that cap.

---

## Framing (for the writeup): a capability result, and the premise now empirically grounded

Two things this track establishes, worth stating plainly:

- **The 77.5%-hit-rate-at-25%-residency number is the empirical basis of the design, not a
  performance footnote.** The whole architecture rests on "the router signal is a stationary skew,
  so recency is a sufficient statistic for frequency, so LRU + a small resident fraction captures
  most reads." Until B′ that argument came only from the 35B-A3B trace replay — an extrapolation
  across a *different* expert count and a *different* trained router. The 26B run confirms it on
  Gemma 4's own routing: 32 of 128 experts resident, 77.5% of reads served from cache. `fieldfare`
  reaches the same conclusion from the other direction (16-slot-per-layer LFU) — two different
  eviction mechanisms, one underlying property of trained MoE routers.
- **This is a CAPABILITY result — but NOT one peers lack, and NOT the fast way to get it.** ⚠
  *Corrected 2026-08-04:* the earlier claim that "llama.cpp and Ollama simply fail to load" this
  26B on 8 GB is **FALSE for current Ollama** — v0.32.5 loads and runs it via a **42% GPU / 58%
  CPU-RAM layer split at ~24.5 tok/s** (measured same-box, `docs/benchmarks.md` §B4), *faster* than
  goinfer's **16.98**. goinfer's genuine distinction is running it **fully GPU-resident** (experts
  streamed host→VRAM and executed on the GPU), not that the model can't otherwise run.
  **And expert paging is very likely the wrong shape here:** a layer split moves an activation
  vector (~10–16 KB) across PCIe per token; expert paging moves **~380 MB of weights** (~31 ms of
  DMA — the wall). See `docs/ollama-chase.md` §D5 for the layer-split scoping and its bound
  (goinfer's CPU path caps the split at ~9–10 tok/s until a GGML-class CPU kernel lands). **The
  durable value of this whole paging line is the method record — the LRU expert cache, the slot-id
  device-read trick, the mixed-M join, the isolation-proves-the-primitive lesson — not the tok/s.**
  The **16.98** (capture-free, 38 slots auto-capped, 81.6% hit rate, sync H2D) sits against
  fieldfare's 5.1–6.3 on its own 8 GB M2 Air — different silicon, so a floor in the peer's
  constrained regime, NOT a comparison row.

## Residency-track pivot: host↔VRAM expert streaming (A′), and why zero-copy stalled at the kernel

The streaming levers above move bytes disk↔host. They do not unblock the stated goal — running
the real 26B-A4B on the 8 GB 2070 — because Gemma 4 declines residency-with-experts-in-VRAM: the
~11.4 GB of int4 experts do not fit. The goal-aligned move is host↔VRAM: keep the ~1.3 GB
non-expert core resident in device memory, read the experts from pinned host memory over PCIe.
This is the "revisit with the residency track" note in Out-of-scope, now taken up.

**A′ = zero-copy expert stacks.** aikit `gpu.NewMappedHostBuffer` (tag `gpu/v0.18.0`): pinned host
memory (`cuMemAllocHost`) whose host pointer IS the device pointer under UVA — no
`cuMemHostGetDevicePointer`, no gocudrv change. The goinfer wiring (experts host-mapped, GEMV reads
them over PCIe) was prototyped, proven wrong at width, and then **removed from shipped code** — a
path that yields wrong logits does not ship even env-gated. The finding below is the record; the
primitive stays (C′'s DMA source), and its exported doc now carries the mis-read caveat.

**The premise — "one allocation change, kernels unchanged, bit-identical" — is FALSE at scale.**
Layered evidence (all on the RTX 2070 SUPER, deterministic):

- The mapped-read PRIMITIVE is correct in isolation, far past anything the model needs: the plain
  W4A8 GEMV reads mapped host memory bit-identically to a device buffer at K up to 4096 words
  (1 MB rows), byte offsets up to 64 MB, and N up to 4096 rows (occupancy). aikit `gpu`
  `TestMappedHost_*` (large-K / offset / bigN) — all 0-diff.
- The goinfer resident MoE forward is BIT-IDENTICAL streamed==resident at toy scale
  (`TestGemma4MoE_streamExpertsBitExact`, gemma4-moe-tiny) and on a many-expert/small-K fixture
  (32 experts, top-8, hidden 256 — 0/256), but DIVERGES at realistic width (hidden 2048, moe_inter
  768 — 255/256 logits differ). One wrong layer-0 expert read cascades through routing.
- The divergence is **not** offset, K, occupancy, expert count, or the allocation/VRAM-layout
  change: the **holdalive control** — allocate the experts host-mapped (identical device-footprint
  relief) but have the kernel READ device copies — is 0/256, bit-exact. Only when the kernel READS
  mapped memory does it diverge.
- Locus: `gemv_w4a8_moe`'s indexed read from mapped host memory, in-context at width. The plain
  W4A8 GEMV reads mapped memory fine in isolation, so the MoE kernel's mapped read is the open
  mechanism — a KERNEL problem, not an allocation one. `moe.ptx` is audited/frozen, not a cheap edit.
- **NOT a stream-ordering / idx race.** Hypothesis: `moe_route` writes `idx[]` and `gemv_w4a8_moe`
  reads it to compute the expert base; maybe the ordering was only *usually* satisfied and the
  slower zero-copy reads changed the interleaving enough to expose a latent race — which C′ would
  then INHERIT. Probe: insert an explicit `r.stream.Sync()` between the route and the expert GEMV,
  streamed path only. Result: **still 255/256** — forcing the ordering changed nothing. So the
  single-stream ordering was already holding (all kernels are on the one `r.stream`; CUDA
  serializes same-stream launches), **there is no latent idx race in shipped code, and C′ does not
  inherit one.** The 5.51× was not the mechanism; the mapped read itself is.

**Recurring lesson (record it next to the fixture-representativeness rule): isolation proves the
primitive, never the composition.** A′'s premise — "one allocation change, kernels unchanged,
therefore bit-identical" — was wrong, and the reason generalizes. The mapped read is correct in
isolation (1 MB rows / 64 MB offsets / 4096 warps) and wrong in the composed forward at width. This
is the third instance on this track: attention was "hd-parametric so 512 should work" (tested
anyway); gelu-tanh was "proven by the cross-product of two shipping paths" (kept the isolated
check); this one had NO composition test until the scaled fixture caught it. A passing isolation
test is necessary, not sufficient — every new primitive needs a composition gate at realistic width
before it is trusted in the full forward.

**Consequence for phasing.** A′-as-one-allocation-change does not reach B′ (the real 26B). The
robust path is **C′ — stage experts into device VRAM slots** (LRU cache, budget-bounded): the
kernel then reads DEVICE memory, which the holdalive control just proved is bit-exact at width. C′
is therefore correctness-required, not merely a latency optimization. Either (a) root-cause why
`gemv_w4a8_moe` specifically mis-reads mapped memory in the full forward (fastest route to the 26B
IF fixable), or (b) build C′ on the proven-correct device-read path. The `NewMappedHostBuffer`
primitive stays valid and useful (correct in isolation; the host-side staging buffer for C′'s DMA).

Reproduce: `scripts/pin_gemma4_moe_forward.py` now takes `PIN_OUT`/`PIN_NUM_EXPERTS`/`PIN_TOPK`
(defaults unchanged → committed golden). Build bigk `PIN_OUT=… PIN_HIDDEN=2048 PIN_MOE_INTER=768
PIN_NUM_EXPERTS=4 PIN_TOPK=2`; the harness is `cuda/gemma4_moe_stream_test.go`
(`GOINFER_HEAVY_TESTS=1 GOINFER_MOE_SCALED_FIXTURE=<dir>`).

**KNOWN CONDITION (left open, not closed): `gemv_w4a8_moe` returns wrong data when its weight
argument is host-mapped (zero-copy) memory at width, while `gemv_w4a8_fwd` reads the same memory
correctly.** Moot for C′ (which reads device memory) and not a shipped-code defect (the shipped
resident path always uploads to device), but it is a real state in which a shipped kernel is
wrong: point that kernel at anything other than device memory and it will be rediscovered the hard
way. The mechanism is unexplained (not offset/K/occupancy/ordering — all ruled out above).

## C′ — VRAM expert cache (the path to B′)

The correct and now-chosen path: experts live in **pinned host memory** (the `NewMappedHostBuffer`
primitive, reused here as a DMA *source*, not a zero-copy target); each token, the routed experts
are **DMA'd into device slots** that the GEMV reads — the device read `holdalive` proved bit-exact
and the sync probe proved race-free.

- **No kernel change (confirmed as the first gate).** `gemv_w4a8_moe` computes
  `weightRow = idx[slot]*rowsPerExpert + n` against a stacked buffer and has no notion of how many
  experts the stack holds. So the host DMAs the routed experts into a small slot-stacked device
  buffer and writes **slot ids** into the buffer the kernel reads as `idx`; the kernel runs
  unmodified against a smaller stack. `moe.ptx` (audited/frozen) is untouched. "Does C′ touch
  moe.ptx" is the cost-deciding question, so it is gate #1, not an afterthought.
- **Build on gocudrv v0.2.0 with SYNCHRONOUS H2D.** The bump to v0.3.0 (async H2D for overlap) is a
  later *optimization* of the fill with its own aikit-wide CUDA sweep (anncuda/enccuda/qwencuda/
  visioncuda), not a prerequisite — sync H2D is a correct first cut. Landing v0.3.1's breaking
  `PinnedHost` signature change under a new architecture is two moving variables; correctness first
  on known-good deps, then bump + measure.
- **Step 1 = STAGING, not yet a cache (shipped, `GOINFER_MOE_CACHE_EXPERTS`).** nSlots = topK, load
  the routed experts fresh every token, slot id = j, NO cross-token reuse. This already fits VRAM
  (only topK experts resident) and reaches B′ — but with topK slots every token re-DMAs all routed
  experts: ~8×30 = 240 experts ≈ **714 MB/token** over PCIe, a ~60 ms/token floor before compute.
  Correct and the right first cut; the "cache" name describes step 2's ambition, not step 1's
  behaviour. Bit-identical to fully-resident at tiny + bigk + full-scaled width (the exact widths
  A′ zero-copy was 255/256), unmodified kernel.
- **Step 2 = the actual cache (BUILT + MEASURED — the win).** `GOINFER_MOE_CACHE_SLOTS=N` gives an
  N-slot per-layer LRU cache (`expertCache`, `EvictLeastRecent` in spirit — the Lever-2 stationary-
  skew verdict finally used); a routed expert already resident skips its DMA. Bit-identical to
  fully-resident at tiny + scaled with reuse+eviction both active (`TestGemma4MoE_cacheReuse_*`).
  **On the real 26B at 32 slots (25% of 128 experts): 77.5% hit rate** (16935 hits / 4905 misses over
  a 64-tok gen), **15.83 tok/s vs 4.98 at nSlots=topK — 3.2×** (both with the capture readback;
  apples-to-apples). Per-token expert bytes 714 MB → ~161 MB. This VALIDATES the Lever-2 premise on
  the real model: recency is a sufficient statistic for the router's frequency, so a small resident
  fraction captures most reads. The slot count is **auto-capped to measured free VRAM at build**
  (`allocSlots` defers the slot allocation until after the core + KV are up, queries
  `Context().MemInfo()`, and caps-and-logs — the repo's "adjust honestly at load, never OOM"
  discipline; e.g. `48 slots would need 4.8 GB but only 4.2 GB free — capping to 38`). At the
  resulting 38 slots (30% of 128): **81.6% hit rate, 16.98 tok/s capture-free**. Next lever: gocudrv
  v0.3.0 async-H2D overlap of the miss DMAs with compute (its own task) — collapses the remaining
  per-token bytes toward the ~50 MB estimate.
- **B′ ACHIEVED (the milestone).** The real gemma4 26B-A4B (~11.4 GB int4 experts, does NOT fit the
  8 GB 2070) decodes RESIDENT via C′ staging: `cuda/gemma4_26b_cache_test.go`. Load 4m49s (the
  11.4 GB pinned alloc + copy — the cost is the load, not the decode; swap-thrashed but no OOM on
  62 GB). Routing CLEAN at 128/top-8 (2730 decisions through the gen, all 128 experts exercised, in
  range — `gemv_f32_f32` confirmed, not discovered). Coherent through the real chat template:
  distinct-trigram 0.818 (floor 0.70), "…**Paris**… the Eiffel Tower, the Louvre Museum… **Gastronomy:**".
  Latency 201 ms/tok (4.98 tok/s) WITH the G4_CAPTURE readback — informative, NOT a benchmark
  (staging floor ~714 MB/tok PCIe + ~30 D2H/tok; step 2 is where that collapses).
- **ARCHITECTURAL COST, recorded up front:** the host must know `idx` to decide what to DMA, and
  `idx` is written on-device by the router. So C′ inherently pays a **device→host `idx` readback per
  MoE layer per token** — ~30 round-trips/token for the 26B, each draining the stream. This is
  *exactly* what `task-metal-moe.md` says the on-GPU router exists to avoid ("a host-side top-k
  readback between router and experts would force a per-token sync and destroy that pipeline"). C′
  knowingly pays it as a correctness vehicle and to reach B′; its tok/s must NOT be read as a
  production streaming number. The production design manages the cache on-device (no readback) and
  is a separate effort.

## Production-config decomposition — measured, and it corrects the lever budget (2026-08-03)

The CUDA-perf levers (structural readback ~12 ms, async miss-DMA ~11 ms, CUDA graphs for dispatch)
were sized against a ~59 ms/token "forward decomposition." Measured **directly at production config**
(`GOINFER_MOE_CACHE_SLOTS=38`, real 26B, N=48 steady-state, direct-drive `ForwardArgmax`; readback and
miss-DMA timed in place, compute/dispatch the residual):

| component | measured | prior estimate |
|---|---|---|
| idx readback (Sync+Download) | **0.81 ms/tok** (30 round-trips × 0.027 ms) | ~12 ms |
| miss-DMA (loadExpertSlot) | **4.28 ms/tok** (12.2 loads/tok, 89.1% LRU hit) | ~11 ms |
| compute/dispatch (residual) | **24.20 ms/tok** | ~17 + ~19 |
| **forward total** | **29.29 ms/tok (34.15 tok/s)** | — |

Two of the three levers were mis-budgeted by an order of magnitude: the per-layer idx readback the
"architectural cost" bullet above worried about is **0.81 ms, not ~12 ms** (Task-1 readback elimination
is retired as an independent lever); and at 38 slots the miss-DMA is **4.28 ms, not ~11** — the LRU
cache already did the heavy lifting (89% hit), so async-DMA overlap can hide ≤4.3 ms.

### Reconciliation against §B4's 16.98 tok/s (58.9 ms/tok) — §B4 confirmed
Same process, the serve path (`m.Generate`, `SamplingParams{}`) measured 17.62 tok/s (56.75 ms) —
**reproducing §B4 within noise. §B4 is validated, not stale. No edit.**

### CORRECTION: the "27 ms serve path" above was a measurement artifact
That earlier reconciliation attributed the 34→17.6 tok/s gap to a "27 ms serve path (full-logits D2H +
sampling + detok)." **Direct per-phase measurement (`GOINFER_DECODE_TIMING`, already in `generateInto`)
refutes that.** Per **decode** token (excludes prefill), 26B @ 38 slots:

| path | forward | sample | logitProc | embed |
|---|---|---|---|---|
| **greedy** (`SamplingParams{}`) | 36.4 ms | **0.00** | 0.00 | 0.02 |
| non-greedy (`GOINFER_NO_GREEDY_FASTPATH=1`) | 41.2 ms | 0.23 | 0.00 | 0.02 |

The greedy per-token serve overhead is **~zero** — the forward is everything. So the 27 ms gap was
**not** per-token serve cost; it was **(a) prefill amortization** — the 26B has no batched `PrefillLast`,
so `generateInto` prefills the 27-token prompt as **27 sequential full-logits `Forward` calls**, and the
harness's `genDur/64` amortized those into the per-token figure (~15 ms/tok here; it shrinks as output
length grows — a per-*request* cost, not per-token) — plus **(b) context-depth forward growth** (36.4 ms
decoding at pos 27–91 vs 29 ms at the direct-drive's pos 12–60; attention is O(context)).

### Structural answers
- **Q1 — does temperature 0 short-circuit to argmax?** **Yes, by construction.** `SamplingParams{}` →
  `Sampler.ArgmaxEquivalent()` true → the `fastGreedy` path: `next = fastNext` (no sampler call) and
  `ForwardArgmax` (on-device argmax, 4-byte readback). No full-logits readback, no 262144-float sort,
  and softcap is skipped (monotone → argmax-invariant). Confirmed empirically: greedy `sample = 0.00`.
  The greedy sampling path is already optimal; there is nothing to fix there.
- **Q2 — vocab-scaled vs fixed** (phi3 **32064** vocab vs gemma4 **262144**, 8.2×; non-greedy):

  | | forward | sample | notes |
  |---|---|---|---|
  | gemma4 262k | 41.2 ms | 0.23 ms | non-greedy adds **4.8 ms** vs greedy |
  | phi3 32k | 7.8 ms | 0.05 ms | non-greedy adds **~0 ms** vs greedy |

  The non-greedy add-on is **vocab-scaled AND dominated by softcap**: gemma4's 4.8 ms is mostly the
  host `262144 × math.Tanh(f64)` **final-logit softcap loop** — which is **Gemma-family-specific**
  (`finalSoftcap>0`; llama/mistral/qwen/phi3 skip it, hence phi3's ~0). Full-logits D2H (1 MB vs 128 KB)
  and host argmax (0.23 vs 0.05 ms) are vocab-scaled but sub-ms. So "the 262144 vocab amplifies an
  existing cost 8×" is true of the **softcap**, and its blast radius is the Gemma family, temperature>0
  only.
- **Allocation (Step 2): NOT material.** `GODEBUG=gctrace=1` over a fixed 64-token generation: decode-phase
  GC is negligible (GCs are dominated by model *load*). No 1 MB/token garbage — `step()` returns the
  reused `logitsPinned` view and the greedy/no-penalty sampler does **not** clone logits (`work = logits`);
  the only per-token alloc is the ~12 KB embed slice. The 1 MB/token hypothesis is refuted (it would occur
  only with penalties/bias via `slices.Clone`).

### Host-side (backend-independent) items — transfer to the Metal track
- **Softcap** (host `math.Tanh` over vocab) and **host argmax**: pure host, backend-independent —
  Metal's Step-6 paging penalty is being measured against a serve budget carrying this same host cost.
- **Prefill sequential-forward waste** (below): a host-side *algorithm* issue, backend-independent.
- Full-logits D2H is backend-specific (Metal's unified memory makes it ~free).

### Re-ranked levers (both halves, against the corrected budget)
1. **Prefill full-logits waste — NEW, cheap, backend-independent, biggest cheap win (TTFT).** Prefill runs
   `resident.Forward` (full logits + D2H + softcap) for **every** prompt token but uses only the **last**.
   ~26 of 27 full-logits readbacks+softcap are wasted ≈ **~130 ms wasted TTFT** for this prompt at 262k,
   scaling with prompt length. Fix: a no-logits forward (`ForwardArgmax`/KV-only) for `prompt[:-1]`; a
   batched `PrefillLast` on cudaResident is the larger version.
2. **Task 3 (CUDA graphs):** the 24–30 ms compute/dispatch forward floor (~19 ms dispatch) — the largest
   *decode* lever, parked behind the tenancy/MPS gate.
3. **Softcap** (temperature>0, Gemma only): ~4.5 ms/tok host loop — parallelize/SIMD the tanh. Small, gated.
4. **Task 2 (async miss-DMA):** 4.28 ms at 38 slots.
5. **Task 1 (idx readback):** 0.81 ms — retired (re-enters only within on-device cache mgmt).
6. **Allocation:** not material, no lever.

**Verdict:** the greedy decode budget is **the forward** — there is no large serve tail to fund. The
biggest cheap win is the prefill waste (TTFT, backend-independent); the biggest decode lever remains
Task 3 (parked). Measurement scaffolding reverted; numbers are the deliverable.

## TTFT is prompt-length-bounded on the resident CUDA path — a PRODUCT finding (2026-08-03)

Two follow-ons: the LM-head prefill waste was **fixed** (KV-only prefill for `prompt[:-1]`, commit
`05f0d8c` — `ResidentPrefillKV.ForwardNoLogits`; 26B saved 4.39 ms/prompt-token, byte-identical), and
then TTFT was measured vs prompt length. With that fix ON, on the 26B (one load, KV-only prefill on):

| prompt tokens | TTFT | ms/prompt-token |
|---|---|---|
| 32 | 1.30 s | 40.6 |
| 128 | 3.37 s | 26.4 |
| 512 | 12.63 s | 24.7 |
| 2048 | **60.71 s** | 29.6 |

**Linear at ~25–30 ms/prompt-token** (the 32-token point is higher from fixed/cold-cache overhead).
**A 2048-token prompt is ~61 s to first token.** This is a **product-level bound, not a perf note:**
goinfer's usable prompt length on the resident CUDA path is currently capped by **sequential M=1
prefill** — each prompt token runs the full 30-layer forward one at a time. §B2's published rows never
exposed it (short prompts, 256-token completions → prefill amortizes away), but any real long-context
use (RAG, code, long chats) hits a minute-plus TTFT. (Absolute numbers are inflated by this box's swap
pressure under the 11.4 GB pinned load; the *shape* — linear, minute-scale at 2k — is the finding.)

The KV-only fix (Task 1) removes the LM head (~4.4 ms/token) but **not** the layer forward, which
dominates prefill. The real fix is **batched prefill**: a `PrefillLast` that ingests the whole prompt
in one M=len pass (each weight streamed once). 

**Backend shape — one gap, not architecture-wide.** Only the two GPU-resident-via-`generateInto`
backends did the sequential M=1 prefill: **cudaResident** (now KV-only, still M=1 layers — needs
`PrefillLast`) and **webgpu `residentDecoder`** (still full-logits M=1 — the Task-1 `ForwardNoLogits`
is a one-method drop-in it doesn't yet have). The **CPU path never had the waste** (`prefillLogits`
batches with the LM head on the last row only, or the sequential fallback runs `runLayers` = KV-only
for `prompt[:-1]`), and **Metal already has batched `PrefillLast`** (`metal/prefill.go`). So batched
CUDA prefill is the outstanding lever, and it is the single biggest TTFT win — bigger than any decode
lever for long prompts.

## Measurement and gates

- **Rigs:** M1 Pro 16 GB (darwin, no firm cap — the fieldfare-comparable rig) *and* the
  Ryzen 7 3700X box (linux, firm cap). Report them separately; a single number across both
  hides the whole finding.
- **Bit-exactness is non-negotiable and already gated:** `TestExpertPaging_bitExact`,
  `TestLayerPaging_bitExact`, `TestMadvise_dontneedRefaultsIntact`. Any lever here that
  cannot keep decode byte-identical to fully-resident is rejected, not caveated.
- **Publish the curve, not a point.** tok/s and p99 token latency vs `--weight-cache`, per
  lever, per platform — the shape the spike's original table had. Per `docs/benchmarks.md`
  rules: commit + date + machine + thermal note inline.
- **Peer figures PINNED (2026-08-02).** turbo-fieldfare's 5.1–6.3 / 31–35 tok/s are from
  its README at **release `0.4` (commit `8648274`, tagged 2026-08-02)**, read 2026-08-02.
  Checkpoint: **Gemma 4 26B-A4B instruction-tuned, 4-bit MLX affine + 8-bit router**. The
  two figures are on **different rigs — 5.1–6.3 on an 8 GB M2 Air, 31–35 on a 24 GB M5
  Pro** — and NEITHER is goinfer's M1 Pro 16 GB, so per `benchmarks.md`'s same-machine rule
  these stay *reference* figures, not a comparison row, until fieldfare is run on the M1 Pro
  (or goinfer on one of those rigs). The README gives no per-figure measurement date and
  notes decode prefill sped up at `0.3` (2026-07-29), so re-pin if the numbers move in a
  later tag. The 4-bit output-quality look on `mlx-community/gemma-4-26b-a4b-it-4bit`
  (~14.3 GB) is still TODO — a large download on the 16 GB rig, run it deliberately.

## Out of scope

- Streaming into GPU-visible buffers (fieldfare `pread`s straight into Metal-visible
  memory). Gemma 4 declines residency on every goinfer backend (`decoder/residency.go:130`),
  so there is no GPU path to feed on this model; revisit with the residency track.
- A custom on-disk container. `.giw` + mmap is the substrate; Lever 1b changes how expert
  bytes are *read*, not the format.
- Cross-token expert prediction (prefetching next token's experts from this token's
  selection). Speculative, harmless when wrong, and plausibly free once Lever 1's pool
  exists — but it is a separate measurement, not a bundled assumption.

## Sources

In-repo: `docs/ideas-weight-memory.md` §2 (shipped 2026-06-13; skew and hit-rate/latency
tables; the "unhideable" claim) and §4 (dense layer streaming), `decoder/moepaging.go`,
`decoder/layerpaging.go`, `decoder/moepaging_spike_test.go`, `decoder/mlp.go:81`,
`decoder/forwardn.go:38`/`:228`, `decoder/residency.go:130`, `go.mod:6`,
`docs/completed/task-gemma4-moe.md`, `docs/benchmarks.md`.
aikit: `mmap/spancache.go`, `mmap/madvise_darwin.go` (the darwin no-op eviction),
`mmap/madvise_linux.go`, commit `6c0483f` — the accompanying perf-campaign write-up was an uncommitted working note and no copy survives, so the commit is the whole record
item 9 (the scan-resistance measurement), tags through v1.14.0.
Peer: `github.com/drumih/turbo-fieldfare` README **@ release `0.4` (commit `8648274`,
2026-08-02), read 2026-08-02** — 1.35 GB resident core (~2 GB with 4K KV cache), 14.3 GB
`.gturbo` on disk, Gemma-4-26B-A4B-it at 4-bit MLX affine + 8-bit router, 16-slot LFU per
layer, bounded parallel `pread` into Metal-visible buffers, shared-branch/read overlap,
≤128-token prefill chunks, 5.1–6.3 tok/s on an **8 GB M2 Air** and 31–35 tok/s on a **24 GB
M5 Pro** (different rigs — reference figures, not an M1-Pro same-machine comparison row).
