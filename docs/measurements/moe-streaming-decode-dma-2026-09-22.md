# MoE streaming: both open CUDA items in task-moe-streaming.md were already closed elsewhere — here is the evidence

Followed up on a request to bump `gocudrv` to v0.3.0 for async H2D overlap of C′'s miss DMAs, and to re-measure P20's
expert-DMA redirect together with it (`docs/tasks/task-moe-streaming.md`'s two remaining "CUDA-only" items as of
2026-09-21). Checked the doc's own instruction to verify current state before assuming — both items turned out to be
already closed, by work this doc's own cross-references didn't carry forward. RTX 2070 SUPER, driver 595.91.07.

## Item 1: gocudrv v0.3.0 async H2D

**Not v0.2.0.** `cuda/go.mod` pins `github.com/eitamring/gocudrv v0.3.2` — already past v0.3.0 and the v0.3.1
`PinnedHost` breaking change the request itself flagged as a hazard to land carefully.

**Already investigated, already declined, already superseded — `docs/completed/aikit-subrange-async-upload.md`
(2026-08-28), which is BEFORE `task-moe-streaming.md`'s own "next lever: gocudrv v0.3.0" line was written.** That
review found: async H2D's real prize is 5.6% (sync overhead), not the naive 1.18x, because async does not make PCIe
transfer itself faster; gocudrv v0.3.2 has offset copies (`CopyFromAt`) and async copies (`CopyFromHostAsync`) as
SEPARATE primitives, never combined on the H2D path, so the ask would have needed a gocudrv change, not a wrapper;
and the sync exists for a real, previously-hit race (non-blocking streams unordered against the null stream). The
recommended alternative — `gpu.UploadBatch` (N copies, ONE synchronize) — **shipped instead**, needed **no gocudrv
bump at all** (`CopyFromAt` already existed at the time), and is exactly what `cuda/resident.go`'s
`loadExpertSlot`/`loadRoutedExperts` use today via `appendExpertSlot`+`UploadBatch` — the same pattern this session's
own L01 and P20 work read and reused all day without realizing it was the answer to this exact question. Measured
then: 20,916 -> 2,038 syncing copies (10.3x fewer) on the real 35B, H2D time -10.2%, **+9.3% tok/s** (14.55 -> 15.90).

**Re-measured fresh on the 26B, current code, to confirm this is still the right read** (`cuda/moe_streaming_decode_profile_test.go`,
new, committed; `GOINFER_MOE_CACHE_PROF=1`, real decode, synthetic embeddings so no tokenizer format issue gates the
measurement, 48 tokens, this box's current free VRAM caps the cache to 29 slots):

| | value |
|---|---:|
| tok/s | 29.72 |
| C′ hit rate | 87.6% (10,090 hits / 1,430 misses) |
| stall (readback wait) | 1.5% of the round trip |
| host (sync) | 5.3% of the round trip |
| **dma** | **93.2% of the round trip, 30.5% of the whole token** |

**The sync-overhead term async H2D (or UploadBatch) can address is small (5.3%) and already captured. The dominant
remaining cost is genuine PCIe transfer time (93.2% of the round trip), which async cannot reduce — the 2026-08-28
decision's own point, now confirmed on the real 26B instead of reasoned from the 35B.** Building gocudrv v0.3.0
async H2D would not move this number. **Not a lever. Not built.**

## Item 2: P20 redirected toward the expert-DMA cost

**Already closed for prefill, this session, before this request arrived.** `p20-expert-locality-2026-09-21.md` found
the mechanism (per-row admission re-fetches the same expert a mean of ~69x per 512-row chunk on M26's tight VRAM) and
`p20-expert-major-m26-2026-09-21.md` built and shipped `cuda/moe_expert_major_gemma4.go`: admit each distinct expert
once per chunk instead of once per row. Measured on the real 26B: **2.66x / 2.50x / 2.39x / 2.26x at
M=512/2048/4096/8012**, `GOINFER_CUDA_MOE_EXPERT_MAJOR` now default on. This IS the expert-DMA fix the redirect asked
for — on prefill. It was built and shipped as its own item (R11/P20 in `docs/tasks/red-october.md`) without a
cross-reference landing in `task-moe-streaming.md`, which is why this request could ask for it again as open.

**Decode is untouched by it and remains at the 30.5% dma share above** — expert-major restructuring needs multiple
rows to bucket by expert; decode is M=1, so there is nothing to bucket.

## What is genuinely still open (found while checking, not part of either original ask)

**CUDA's gemma4 decode path has no equivalent of the CPU's already-shipped Lever 3** (`docs/tasks/task-moe-streaming.md`,
"overlap routed reads with the resident branch" — shipped default-on on the Mac, ~1.09-1.19x). On CUDA, `gemma4MoeMLPPre`
issues the dense branch's GEMVs as async kernel launches, then `layerTail` calls `r.stream.Sync()` (to make the
g4x2-clear ordering correct) immediately before `loadRoutedExperts`'s routing readback — **which drains the stream, so
the dense branch has already finished executing by the time the routing decision (and therefore the DMA) even starts.**
There is currently nothing running on the GPU while a miss DMA is in flight on CUDA. A genuine compute/DMA overlap
restructuring — the CUDA analog of the CPU's Lever 3 — could address some of the 30.5%-of-token dma share directly.

**This is new, unscoped work, not a completion of either original item, and is NOT attempted in this pass.** It would
need its own correctness design (the current structure serializes on the routing readback for a reason — the DMA
target depends on it — so overlap needs a genuinely different shape, e.g. issuing the DMA and running the FOLLOWING
layer's independent early work, or restructuring which work depends on the readback) and its own pre-registration
before any code is written, matching this session's own discipline throughout. Flagged for the owner to decide
whether it is worth funding, not started unilaterally.

## Also verified, per the request

`git log` / `docs/tasks/task-moe-streaming.md`'s own git history: last touched 2026-09-21 by the doc-review move to
`docs/tasks/`, not by either item resolving — confirming the "next lever" and "P20 redirected" lines were genuinely
stale prose, not recently-written accurate status. Both corrected in place, dated, with the doc-reviewed footer bumped.
