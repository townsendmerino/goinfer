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
Since the Gemma 4 26B-A4B benchmark rig is an M1 Pro 16 GB (`docs/task-gemma4-moe.md`),
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

**Status: complete.** goinfer is now on `aikit v1.16.0` (`go.mod:6`), past the `6c0483f`
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
   `decoder/moepaging_test.go:11` cites `TestSpanCache_evictsLRUTailOverBudget`, renamed in
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
*parallel* dense-MLP + MoE pair (`docs/task-gemma4-moe.md`, Delta 1): the dense branch is
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

## Lever 4 — expert-major prefill batching

`decoder/forwardn.go:14` states it plainly: batched prefill vectorizes attention but "the
MoE FFN itself stays per-row (router picks different experts per token)" — the per-row call
is `forwardn.go:228`. Under streaming that is the worst case: the same expert can be
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
`decoder/forwardn.go:14`/`:228`, `decoder/residency.go:130`, `go.mod:6`,
`docs/task-gemma4-moe.md`, `docs/benchmarks.md`.
aikit: `mmap/spancache.go`, `mmap/madvise_darwin.go` (the darwin no-op eviction),
`mmap/madvise_linux.go`, commit `6c0483f` + `docs/internal/perf-campaign-2026-07-28.md`
item 9 (the scan-resistance measurement), tags through v1.14.0.
Peer: `github.com/drumih/turbo-fieldfare` README **@ release `0.4` (commit `8648274`,
2026-08-02), read 2026-08-02** — 1.35 GB resident core (~2 GB with 4K KV cache), 14.3 GB
`.gturbo` on disk, Gemma-4-26B-A4B-it at 4-bit MLX affine + 8-bit router, 16-slot LFU per
layer, bounded parallel `pread` into Metal-visible buffers, shared-branch/read overlap,
≤128-token prefill chunks, 5.1–6.3 tok/s on an **8 GB M2 Air** and 31–35 tok/s on a **24 GB
M5 Pro** (different rigs — reference figures, not an M1-Pro same-machine comparison row).
