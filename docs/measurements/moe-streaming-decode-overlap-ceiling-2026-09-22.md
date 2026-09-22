# CUDA gemma4 decode: compute/DMA overlap — ceiling measured before any build

Follow-up to `moe-streaming-decode-dma-2026-09-22.md`'s "genuinely still open" item: CUDA's gemma4 decode has no
equivalent of the CPU's Lever 3 (`docs/completed/task-moe-streaming.md`). Today `gemma4MoeMLPPre` issues the dense
branch and THEN the router; `layerTail` drains the stream (`r.stream.Sync()`) for the `g4x2` clear, and
`loadRoutedExperts` drains it again before the idx readback — so by the time a miss DMA is issued the GPU has
nothing left to run. 30.5% of the token is DMA against an idle device. RTX 2070 SUPER, driver 595.91.07, real 26B
`.giw`, C′ capped to 29 slots by free VRAM, 48 synthetic-embedding tokens (`TestMoEStreamingDecodeProfile`).

## What can actually overlap (read from the code, before measuring)

The routing dependency chain is real: layer l's readback needs layer l's router, which needs layer l's attention,
which needs layer l-1's experts. Nothing from layer l+1 can move under layer l's DMA. Inside layer l, exactly two
things are independent of the readback:

1. **The dense branch** (`rms → g,u GEMV → swiglu → d GEMV → normF32`, all on `h`): reorder it AFTER the router and it
   executes while the host reads back idx and issues the DMA. The router and the dense branch share no buffers
   (router: `g4rn/rLogits/rIdx/rWgt`; dense: `mq/mSc/gO/uO/dq/dSc/dScr/g4x1`), so the swap is bit-identical by
   construction.
2. **The hit-expert prefix of segC.** segC accumulates `g4x2 += wgt[j]·down_j` in rank order j=0..7; that order fixes
   the f32 sum and must not change. But a rank whose expert is already resident needs no DMA, so the launches can be
   issued in rank order with a stream-side wait inserted only before each MISS rank (one event per miss, recorded after
   its copy). Every hit rank ahead of a miss then runs under that miss's DMA — same launch order, same arithmetic, bit-
   identical. With ~1 miss per layer uniformly placed among 8 ranks, the expected overlappable prefix is ~3.5 ranks.

Everything else is either serialized by data (attention, router) or is the DMA itself.

## Why the current API cannot do it, and what a build needs

Every host copy on the path is a full-context synchronize: `gpu.Upload` syncs before AND after (audit C-01),
`gpu.UploadBatch` syncs before the first copy and after the last. A leading sync drains the dense branch before a
single byte moves, so no reorder of goinfer's launches can produce overlap through these calls. aikit's `Queue.s`
(the stream) and `Buffer.b` (the device pointer) are unexported; gocudrv v0.3.2 has everything needed
(`Event.Record/Synchronize`, `Stream.WaitEvent`, `cuMemcpyHtoDAsync`, `Buffer.ZeroAsync`), but goinfer cannot reach
it. A build is therefore an **aikit `gpu` API addition first** (a stream-ordered offset H2D from pinned memory, event
record/wait, a stream-ordered zero) plus a gpu/v0.33.x bump, then the goinfer restructure. That is the cost side of the
decision, and it is why the ceiling is measured before anything is written.

## Pre-registered decision rule (written before the profile ran)

Instrument: `GOINFER_MOE_CACHE_PROF` extended with sync-bounded per-class timing (`profMark`) for a g4moe token:
`attn` (segA + rope + attention + o-proj), `dense`, `router` (+ xe norm), `clear` (the g4x2 Upload and its two syncs),
`rt` (loadRoutedExperts: stall+host+dma), `segC` (experts + join), `head`. Each class is sync-to-sync, so it carries
its own launch latency and the profiled token is slower than the real one — every class number is an UPPER bound.

- **C1** = min(dense, dma) per token — design 1 alone.
- **C2** = min(dense + segC, dma) per token — designs 1+2 (an upper bound; the real prefix is ~44% of segC, and
  per-layer mins are smaller than the min of per-token sums).
- **S** = clear + stall — host syncs a stream-ordered design deletes outright.
- **Kill without building if C2 + S < 5% of the profiled token.** Session drift on this box is ~3.5%; an upper bound
  under 5% cannot produce a measurable win.
- If ≥ 5%: build design 1 (+2 only if C2 − C1 is itself ≥ 3%), A/B against the do-nothing arm on the same loaded
  model, alternating 48-token gens, ≥ 5 pairs. **Ship ≥ 3% paired-median tok/s AND bit-identical logits
  (`Float32bits`) against the shipped order on every token; park 1.5–3%; kill < 1.5%.** Ambiguous (within the last 5%
  of a threshold) → parked.

## Result 1 — the ceiling (log: `moe-streaming-decode-overlap-ceiling-2026-09-22.log`)

Profiled token 35.2 ms (class sum 35.1 — the instrument accounts for the whole token; unprofiled 33.6 ms):

| class | ms/tok | share |
|---|---:|---:|
| attn (segA + rope + attention + o-proj) | 4.61 | 13.2% |
| dense branch | 2.59 | 7.4% |
| router + xe norm | 2.12 | 6.1% |
| g4x2 clear (Upload: two full syncs) | 1.48 | 4.2% |
| round trip (stall 1.0% / host 5.3% / **dma 93.7%**) | 11.22 | 32.0% |
| segC (8 expert ranks + join) | 11.14 | 31.8% |
| head | 1.89 | 5.4% |

**C1 = 7.4%, C2 = 29.5%, S = 4.5%, C2 + S = 34.1% — far past the 5% kill line; C2 − C1 = 22% ≥ 3%, so both designs
are funded.** Two things the numbers say beyond the rule: segC's 0.37 ms/layer for 24 MB of expert reads is launch-
latency-bound (24 launches; the bytes alone are ~55 µs at 448 GB/s), which is exactly the kind of work that hides
under a DMA; and the DMA is 0.37 ms/layer against GPU work of ~0.7 ms/layer, so the overlap is DMA-limited, not
compute-limited — the ceiling is the DMA, and the design reaches most of it.

## Result 2 — the build, and the A/B (log: `moe-streaming-decode-overlap-ab-2026-09-22.log`)

**Shipped, default on.** Two halves:

*aikit `gpu` (new surface, gpu/v0.33.2):* `Event` (`Device.NewEvent`, `Queue.Record`, `Queue.Wait`, `Event.Sync`),
`Queue.UploadAsyncAt` (stream-ordered H2D of a slice of a pinned buffer into an offset of a device buffer — gocudrv's
exported async copy insists both sides be whole buffers of equal length, so this goes through the raw driver on the
calling thread; every wrapped gocudrv call returns after its command thread has enqueued, so program order is stream
order), `Queue.ZeroAsync`, `MappedHostBuffer.Host`. Pinned by `TestCUDA_uploadAsyncAt_offsetsAndEvents`: an offset copy
lands exactly its bytes, a zero clears exactly its range, a copy on one queue is visible to another after Record/Wait
and to the host after `Event.Sync`.

*goinfer `cuda` (`GOINFER_MOE_DMA_OVERLAP`, off under graphs and L-01):* every MoE pre-half records `evRoute` right
after its route kernel; gemma4's dense branch moves AFTER the router (the two share no buffers); `loadRoutedExperts`
waits on `evRoute` instead of draining the stream, DMAs each miss on a second queue `dmaQ` with `UploadAsyncAt` and
records one event per miss, uploads the slot-id table with `UploadAsync` on `r.stream`; both `moeMLPPost` and
`gemma4MoeMLPPost` call `waitMiss(j)` before rank j, a device-side `Queue.Wait` only for ranks that missed; the g4x2
clear is a `ZeroAsync` on `r.stream`. Generic: the mechanism lives in the shared `layerTail`/`loadRoutedExperts`/rank-
loop path every C′ architecture takes (the MacBook's note that the gap was never gemma4-only is right); only the
dense-branch reorder is gemma4's.

**A/B (`TestMoEDMAOverlapAB`)** — one loaded 26B, same 48-token input sequence, arms flipped between generations,
ABBA, the shipped draining path as the do-nothing arm, 8 pairs after a discarded warm-up of each arm:

- **Bit-identical: 48 tokens × 262,144 logits, `Float32bits` equal, both arms; argmax sequences equal in all 8 pairs.**
- **Paired median speedup 1.271× (min 1.269, max 1.288); pooled 30.36 → 38.66 tok/s.** Every pair cleared the
  ship bar on its own; the spread across pairs (0.019) is a tenth of the effect.

Verdict by the pre-registered rule: **SHIP**. 27% realised against a 34% upper bound. The remaining gap is the part of
the ceiling that was an over-estimate by construction (C2 counts all of segC; the hit prefix ahead of a miss is ~44% of
it on average) plus the DMA that still has nothing left to hide under.

## What this does not cover, stated

- Prefill is untouched: `moe_expert_major*.go` has its own admission and keeps its syncs; decode was the DMA-idle case.
- Under `GOINFER_CUDA_GRAPHS` the draining path runs (a captured segment cannot record or wait mid-graph). Graphs are
  off by default and were measured not worth flipping on a dense 1.5B; whether that still holds for a 40-launch MoE
  layer is a separate question this profile sharpens (segC is launch-bound) but does not answer.
- `docs/benchmarks.md` carries no row for this: the A/B drives synthetic embeddings, not a prompt, so 38.7 tok/s is a
  paired-effect number, not a provenance-gated benchmark. A `bench_peer.py` row for the 26B is the next measurement.
- Correctness gates re-run on the overlap path: `TestGemma4MoE_cacheExpertsBitExact_{tiny,scaled}`,
  `TestGemma4MoE_cacheReuse_{tiny,scaled}`, `TestGptOssExpertCacheAB` (the generic non-gemma4 C′ path),
  `TestPrefillMoE_bitIdentical`, `TestSlotAllocation_matchesGranularityForm` — all PASS.
