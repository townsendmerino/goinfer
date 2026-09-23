# Lead 3 — populate-then-pin vs pin-then-populate for the CUDA expert-stack buffer: real win, noisy magnitude, not wired

`docs/tasks/task-freetoken-techniques.md` Lead 3's "cheap, narrow experiment against an already-known,
already-measured cost": `cuda/resident.go`'s `mapBytes` (the C′ expert-stack DMA source, ~11.4 GB) allocates
pinned host memory first (`NewMappedHostBuffer` → `cuMemAllocHost`) then copies into it. FreeToken's loader
does the reverse — populate ordinary memory, pin afterward. The doc flagged one real unknown before writing
any code: whether the primitive to populate-then-pin exists in aikit's dependency stack at all.

## Finding: the primitive already exists, one level down

aikit's `gpu` package doesn't expose it, but the pinned gocudrv (v0.3.2) does:
`cuda.RegisterHost[T](ctx, mem []T)` calls `cuMemHostRegister` on an **already-allocated** Go slice — pin
memory that's already resident, rather than allocate-and-pin in one call. goinfer's `cuda` package already
imports `gocudrv/cuda` directly in a couple of files, and `aikit/gpu.Device.Context()` exposes the
`*gc.Context` `RegisterHost` needs. **No aikit change was needed to measure this** — the doc's own "needs an
aikit-side primitive that may not exist yet" caveat turned out not to block the experiment, only production
wiring (below).

## Method

`TestLead3_pinOrderMicrobench` (`cuda/lead3_pin_order_bench_test.go`), RTX 2070 SUPER, driver 595.91.07, real
scale (11.4 GB, the doc's own cited figure for C′'s expert stack). Two arms, same total work (allocate N bytes
of GPU-reachable host memory, fill it with real content), timed in two phases each so allocation cost and fill
cost don't hide in one number:

- **A (today's `mapBytes` order):** `NewMappedHostBuffer` (pinned alloc) then `copy` into it.
- **B (Lead 3's proposal):** ordinary `make([]byte, N)` + `copy`, then `gc.RegisterHost` (pin in place).

One process, one warm-up pair discarded, 3 timed pairs, ABBA-alternated.

## Result (log: `lead3-pin-order-2026-09-22.log`)

| pair | A: alloc + copy | B: copy + register | B/A |
|---:|---:|---:|---:|
| 0 | 5.76s + 0.91s = 6.67s | 4.24s + 0.79s = 5.03s | 1.33× |
| 1 | 17.19s + 0.99s = 18.18s | 4.73s + 0.73s = 5.46s | 3.33× |
| 2 | 6.40s + 0.87s = 7.27s | 0.82s + 0.81s = 1.63s | 4.46× |

**B won every single pair, by a wide margin every time — the direction is not in question.** Pooled: 32.1s vs
12.1s, 2.65×. Fill time is comparable between arms (both are an ordinary memory write); the whole effect is in
the allocation side: A's `alloc` step (5.8–17.2s) is both far slower than B's `register` step (0.7–0.8s,
consistently) AND itself highly variable, while `register` is not.

**This variance is not noise to shrug off — it's the mechanism, and it corroborates the FreeToken claim rather
than undermining it.** `cuMemAllocHost` on a mostly-idle box still has to find 11.4 GB of *lockable* physical
RAM; if the box's page cache is holding a lot of reclaimable memory at that moment (this box had 16 GB of
`buff/cache` before the run), the kernel has to reclaim it first — a real, variable cost that scales with
however much cache exists right then. `cuMemHostRegister` on already-resident pages (B) needs no such
reclaim: the memory is already committed by the ordinary `copy`. Consistent with this: `buff/cache` measured
862 MB immediately after the test, against 16 GB before it — the three cache-evicting allocations in arm A
visibly drained it over the run.

**Not run, and why the magnitude here should not be taken as the production number.** Production hits this
path exactly ONCE per process lifetime (model load), from whatever cache state a freshly-started server
happens to find — not three repeated allocations in one process with no cache-clearing between them (this
box has no root, so `/proc/sys/vm/drop_caches` wasn't available to give each trial a clean, comparable start).
A single real load, on a real serving process's actual cache state, is the number that matters and was not
measured here.

## Verdict

**Real, worth building — but the exact multiplier is unproven, and this microbenchmark used the raw gocudrv
primitive directly, not a wired-and-tested aikit API.** Two things before this reaches `mapBytes`:

1. **aikit needs a proper wrapper** (mirroring `NewMappedHostBuffer`'s shape: allocate/accept a
   `[]byte`, expose the zero-copy device `Buffer()` the same way) — this measurement went around aikit
   through `Device.Context()` on purpose, to answer the primitive-exists question cheaply, not to ship
   production code through a side door.
2. **The real decision measurement is a single real model load, old order vs new, on a freshly started
   process each time** (not repeated allocations in one process) — the number that answers "does this move
   the real 4m49s load time `task-moe-streaming.md` measured," which this microbenchmark deliberately did
   not attempt.

Pre-registered for that follow-up: **ship if the real 26B load time drops ≥15% (the same bar `task-moe-
streaming.md` implicitly set by naming the pinned-alloc-plus-copy as "the" 4m49s cost); park 5–15%; kill
<5%.** Not started — a separate build step from this measurement, per this repo's own split-before-build
convention.

## Follow-up, same day: the real decision measurement — PARK

Built both pieces the follow-up named: aikit `Device.RegisterMappedHostBuffer` (populate-then-pin, same
`Bytes()`/`Buffer()`/`Close()` shape as `NewMappedHostBuffer`, pinned in place with no allocation and no
copy — better than this measurement's own microbenchmark arm B, which still paid a `make`+`copy`) and
`Queue.UploadAsyncAtFrom` (additive — `UploadAsyncAt`'s existing shipped signature, gpu/v0.33.2, is
untouched; the C′ decode overlap's DMA path needed a source-agnostic entry point, since `.Host()` returns
nil for a register-origin `MappedHostBuffer` and the overlap path is default-on). `cuda/resident.go`'s
`mapBytes` gets a third arm, `GOINFER_MOE_PIN_REGISTER` (opt-in, default off): register `src` — the
already-fully-populated merged expert-stack slice `packWeight` built — in place, no copy at all. Correctness:
`TestCUDA_registerMappedHostWeight_zeroCopy` (aikit, bit-identical GEMV read) and every existing C′
correctness gate (`TestGemma4MoE_cacheExpertsBitExact_{tiny,scaled}`, `TestGemma4MoE_cacheReuse_{tiny,scaled}`,
`TestGptOssExpertCacheAB`) re-run WITH the flag on — all pass; the flag OFF path is untouched and still
passes.

**Real 26B load, `GOINFER_MOE_CACHE_EXPERTS=1`, one process per sample (not repeated in-process trials, per
this record's own earlier caveat), ABBA-interleaved, first trial (a cold driver/module JIT outlier, 59.2s)
discarded** (log `lead3-realload-2026-09-22.log`):

| arm | n | mean | stdev |
|---|---:|---:|---:|
| old (allocate-then-copy, default) | 4 | 46,243 ms | 591 |
| new (populate-then-register) | 5 | 41,865 ms | 3,958 |

**Ratio: 1.105× — 10.5% faster. By the pre-registered rule (ship ≥15%, park 5–15%, kill <5%): PARK.** New
never lost a single trial (5/5 ≤ old's range), so the direction is not in doubt — but the magnitude is
noisier than the isolated microbenchmark's 2.65× suggested, for a plain reason: a real load spends most of
its ~46s on the checkpoint's disk read and tensor decode, not the ~11.4 GB pin step alone, so the pin
order's effect is diluted inside a much larger total. One further pattern, named rather than explained away:
new's first two samples (46.4, 46.0s) matched old closely, then its last three (38.8, 39.1, 39.0s) dropped
and tightened sharply (stdev 0.13s) — consistent with *something* about system state settling over the run
(plausibly the `.giw` file's own page-cache residency), but not isolated here, and old showed no matching
drift. A bigger run, ideally with `/proc/meminfo` sampled per trial (needs root this box doesn't have, for
`drop_caches` between samples), would be needed to separate "the real effect is ~15%+ once the system
reaches its steady state" from "the real effect is ~10% and the tail three samples are a coincidence."

**Verdict: PARKED, not shipped, not killed.** `GOINFER_MOE_PIN_REGISTER` stays opt-in, default off. The
code is correct and gated (kept, not thrown away — it is real API surface with a real, if inconclusive,
measured benefit), but does not clear the ship bar this record itself set before measuring. Revisit with a
larger sample, or on a box where cache state can be reset between trials, before reconsidering the default.
