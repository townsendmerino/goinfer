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

Run 2026-11-02, 17:39:44 to 17:42:58 PST (about 3 minutes, against the 8 to 12 estimated), at `6f5da35ea`, under the exclusive
timing lock, `kwin_wayland` the only other CUDA context. Every cell started at 7,227 MiB free VRAM and 58.9 to 59.6 GB available;
the 1-minute load average rose from 0.43 to 2.99 over the run, below the void bar of 8.0, so no cell is void. Log: `run-2026-11-02.log`.

| cell | requested context | resolved context | expert slots a layer | pooled hit rate | decode |
|---|---|---|---|---|---|
| 1 | 4,096 | 4,096 | 29 | 96.91% | 62.02 tok/s |
| 2 | 8,192 | 8,192 | 10 | 51.23% | 24.05 tok/s |
| 3 | 16,384 | refused at load: 7.51 GB of KV against 5.47 GB free beside the weights | | | |
| 4 | 32,768 | refused at load: 14.89 GB of KV | | | |
| 5 | 32,768 | refused, as cell 4 | | | |
| 6 | 16,384 | refused, as cell 3 | | | |
| 7 | 8,192 | 8,192 | 10 | 51.23% | 24.05 tok/s |
| 8 | 4,096 | 4,096 | 29 | 96.91% | 62.47 tok/s |

**Reading, by the registered rules: AMBIGUOUS (rule 3), because rule 1's premise is not met.**
- Rule 1 compares the largest context of 16,384 or more that loads against 4,096. **None loads**: at 16,384 the build refuses a KV
  of 7.51 GB (447.5 KB a position across 30 layers) beside 5.47 GB free, so rule 1 cannot be read.
- Rule 2 fails. The one larger context that loads, 8,192, decodes at 24.05 tok/s against 4,096's 62.25 (the pair means): **61.4% below**,
  with pair spreads of 0.0% and 0.7%.
- So by the rules as written this is rule 3, ambiguous, and it is not re-graded. Its numbers are not marginal, and they are stated plainly:
  doubling the KV share from 4,096 to 8,192 cost the expert cache 19 of its 29 slots a layer. The hit rate fell from 96.9% to 51.2%, and
  decode fell to 0.39x. On this box the longest prompt that fits with paged experts is between 8,192 and 16,384.
- Reported, not graded (rule 4): the resolved context followed every request that loaded; the refusal names the KV it needed.
- **What it bears on.** KV11's "first, measure" asked whether the fixed split costs something. These readings say the cost of KV share is
  steep and immediate on the 8 GB card: there is no context between 4,096 and the refusal at which both a long prompt and a fast short chat
  hold. A registered follow-up that can resolve should read the curve between 4,096 and 8,192 (at 5,120, 6,144 and 7,168) and at the
  longest context that loads, which the rules above could not.
- **Card time:** about 3 minutes.
