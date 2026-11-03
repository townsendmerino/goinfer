# KV11's KV-share trade curve on CUDA: Gemma 4 26B-A4B with paged experts (registration, 2026-11-02 17:39 PST)

**Registered before the run** (nobara-sb01; the coordinator's night plan of 2026-11-02 17:16, item 4). KV11's "first, measure"
(`docs/tasks/task-kv-architecture-2026-10.md` §6, KV11) asks for "the trade curve on one paged-MoE model per GPU box: decode speed,
and the longest prompt that fits, against the KV share". On CUDA the expert cache is sized once, after KV, from the free VRAM it
measures (C′), so every byte given to KV is a byte the experts do not get. This measures what that costs on nobara.

## Design

- **Model and box.** `~/models/gemma4-26b-int4.giw` (local NVMe), CUDA, C′ on (`GOINFER_MOE_CACHE_EXPERTS=1`), the expert slots
  auto-sized to free VRAM (`GOINFER_MOE_CACHE_SLOTS` unset); RTX 2070 SUPER (8 GB), driver 595.91.07.
- **Cells.** `ResidentContext` 4,096, 8,192, 16,384 and 32,768, in the mirrored order 4,096, 8,192, 16,384, 32,768, 32,768, 16,384,
  8,192, 4,096: two readings per context, a drift in the box showing against both ends. One load per cell, closed before the next.
- **Recorded per cell:** the context requested and the one the build resolved (`ctxCap`, the longest prompt that fits) or its
  refusal; the expert slots per layer; the pooled hit rate; the decode rate over 200 tokens from position 0 (synthetic embeddings,
  the trained router choosing the experts, as `TestMoEPerLayerHitRate`); and, at the cell's start, the 1-minute load average, the
  available memory and the free VRAM.
- **Instrument.** `TestKV11TradeCurve` (`cuda/moe_perlayer_hitrate_test.go`), under the exclusive timing lock (a timed run).

## Pre-registered reading

The pair spread at a context is |pass 1 - pass 2| / their mean, for the decode rate.

1. **The fixed split costs something** (KV11's build case stands): at the largest context of 16,384 or more that loads, decode is more
   than 10% below 4,096's, and the gap is more than twice the larger pair spread of the two contexts.
2. **The fixed split costs little** (KV11's build case weakens on this box): decode at every context that loads is within 3% of
   4,096's.
3. **Ambiguous:** anything between. Reported with its numbers, no conclusion beyond them.
4. **Reported, not graded:** the slots, hit rate and resolved context per cell, and where the context the build resolves stops
   following the request.
5. **Void:** a cell whose 1-minute load average is above 8.0 at its start, or whose available memory is below the load's need plus 2 GB
   (the run-budget rule); re-run in place, listed. Another timed run on the box voids the run.

## Estimated wall time

Eight loads of the 26B .giw (about 30 to 60 s each from NVMe, C′'s host copies pinned) and 200 decode tokens each (about 10 s): about
8 to 12 minutes, by day, under the exclusive lock.

## Results (appended when read)
