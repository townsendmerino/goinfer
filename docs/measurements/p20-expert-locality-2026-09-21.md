# P20 expert-major on CUDA: "measure the split before building" — real locality data, and a correction to my own first read

`docs/queue-performance.md`'s P20 entry names this exact step ("Capture UploadProfForTest's byte counters in the next probe and size it from those") and never ran it. Real `gemma-4-26b-a4b-it` int4 (`gemma4-26b-int4.giw`), `-moe-cache-experts`, `-ctx 8192`, RTX 2070 SUPER, driver 595.91.07 — the same
checkpoint and config `docs/queue-performance.md`'s M26 guard log used (30 MoE layers, hidden=2816, inter=2112, nE=128, topK=8). Committed instrument: `cuda/p20_expert_locality_test.go`, using a new test-only hook (`cuda/resident.go`'s `routeRecord` field) that records every routed expert id per layer across one whole prefill chunk, so "how many DISTINCT experts does an M-row chunk touch, per layer" can be read directly rather than inferred.
Logs beside this file (`p20-expert-locality-2026-09-21-M512.log`, `-M8.log`).

## First read, and why it was wrong

The R11 brief's build item (b) says: "group rows by expert, **stage each distinct expert once per chunk**, run its rows with a batched-K GEMM." My first reading of that sentence assumed it needs every distinct expert the chunk touches resident in VRAM **simultaneously** (call this Design A). Measured under that reading, M26's own VRAM makes it look dead on arrival: this box caps the C′ cache to **10 slots/layer** (1.6 GB free, capping from a requested 64 to 10, ~1.1 GB), while a 512-row chunk touches **28-117 distinct experts per layer** (median 66) — 3-12x the slot budget, in every layer. Even shrinking the chunk to **M=8 (one topK-width micro-batch)** does not fix it: the *minimum* distinct count across all 30 layers is still 13, above the 10-slot cap.

**That reading is wrong, and the fix is in the sentence itself: "once per chunk" does not mean "all at once."** The design the brief actually describes — and the one real expert-major/grouped-GEMM MoE kernels use — processes experts **sequentially** (or in small pipelined waves), gathering the subset of the chunk's rows that route to the CURRENT expert, running its batched-K GEMM, then evicting and moving to the next expert. That needs only a small, bounded number of concurrently-resident experts (as few as 1-2, pipelined for overlap) — not `distinct` of them — and still achieves "one DMA per distinct expert per chunk" as its TOTAL volume, regardless of how few slots are concurrently held. Design A's VRAM-infeasibility finding above is real and worth keeping (it rules out the naive "hold everything" implementation), but it does not kill the item; it is a design constraint, not a floor.

## The real numbers (Design B's actual ceiling)

| M (chunk width) | distinct/layer: min / p25 / median / p75 / max | current per-row LRU hit rate, this same run |
|---:|---|---:|
| 512 | 28 / 46 / 66 / 95 / 117 | 61.3% (75,305 hits / 47,575 misses) |
| 8 | 13 / 19 / 24 / 26 / 38 | 48.9% (938 hits / 982 misses) |

At M=512: total expert lookups = 512 x 8 x 30 = 122,880 (matches hits+misses exactly). **Sum of per-layer distinct experts = 2,090** — that is the total DMA calls a perfect Design-B implementation would issue this chunk, against the **47,575 the current per-row LRU actually issues** (misses; the LRU already captures 61.3% via ordinary temporal locality with only 10 slots) and the 122,880 the naive "no cache at all" path would issue. Design B's own hit-rate ceiling: (122,880 - 2,090) / 122,880 = **98.3%**.

**Additional DMA-call reduction available beyond what the current LRU already gets: 47,575 / 2,090 = 22.8x.**

## Wall-clock projection (a projection, not a result — flagged as such, matching this repo's own convention)

`docs/queue-performance.md`'s own M26 profile (`GOINFER_MOE_CACHE_PROF=1`, M=512) found C′ DMA at **59.5% of prefill wall time**, identical between the batched and sequential attention arms to within 0.16% — i.e. DMA time there scales with DMA *call count*, not with which arm issues it. Applying that same calls-proportional assumption to the 22.8x call reduction above:

```
new wall fraction = (1 - 0.595) + 0.595 * (2090 / 47575) = 0.431   ->   1 / 0.431 = 2.32x
46.5 ms/token (the flat sequential baseline) -> ~20.0 ms/token
```

**~20 ms/token lands inside R11(b)'s own registered ships band (<=25 ms/token, >=1.86x)**, not the "kill" my first (Design-A) read would have implied. This is exactly the same shape of projection `docs/queue-performance.md` explicitly declined to quote before ("if the ~70% dense share holds... 46.5 -> ~20... **do not quote those two numbers as results**") — the number is coincidentally close, but this one is now grounded in a real, measured distinct-expert count on the real checkpoint, not shape arithmetic on active-parameter ratios. It is still a projection: it assumes (a) the calls-proportional DMA-time relationship holds at the NEW, much lower call count (untested — per-call fixed overhead could dominate at low counts, unlike at today's ~1,586 misses/layer), and (b) the gather/scatter and batched-K GEMM Design B needs add negligible cost of their own (unmeasured).

## Decision

**Not killed. The measurement earns the build, reversing this record's own first (Design-A) read.** The next step is the actual kernel: a batched router pass, a per-expert row gather, a new `.cu` (own PTX module, `moe.ptx` untouched, matching the `decode_splitkv.cu`/`router_f32.cu` precedent), an indexed batched-K GEMM, and a scatter back into the residual — bit-identical to the sequential FFN by construction (same per-row expert math, reordered across rows, no shared accumulation). **Not built in this pass.** Pre-registered decision rule for that build, using this measurement as its basis: paired-and-interleaved on the real M26 checkpoint at M=512, sequential 46.5-46.7 ms/token (the flat, already-measured baseline) is the denominator; **>=1.86x (<=25 ms/token) ships, 1.33-1.86x (25-35 ms/token) parked, <1.33x (>35 ms/token) killed** — R11(b)'s own band, restated here so the build has a fixed target rather than a moving one.

## Not established

Bytes moved per expert (only call counts were captured here, matching what `routeRecord` records; a byte-level capture via `UploadProfForTest` would refine the DMA-time assumption above but was not run). M35 (Gated-DeltaNet, no batched recurrent state — R11(b)'s own scope exclusion, unaffected by this). Whether Design B's own gather/scatter/scheduling overhead is small enough not to eat into the 22.8x — genuinely unmeasured until the kernel exists.
