# Lead 4 step 1 — per-layer C′ hit rate at a fixed total budget: real, but small and already diluted

`docs/tasks/task-freetoken-techniques.md` Lead 4's own "first step": "instrument per-layer hit rate at a
fixed total budget... and see whether the hit-rate distribution across layers is actually uneven before
building anything." A global pool across layers only pays if layers differ in hotness; this measures that,
on the real 26B, before writing any pooling code.

## Method

`TestMoEPerLayerHitRate` (`cuda/moe_perlayer_hitrate_test.go`), new: `PerLayerCacheStatsForTest` (new,
`cuda/testhooks.go`) returns each layer's hit/miss counters individually — the per-layer `expertCache`
already tracked them; nothing before this exposed them per layer, only summed. RTX 2070 SUPER, driver
595.91.07, real 26B `.giw`, C′ on, `GOINFER_MOE_CACHE_SLOTS` unset (auto-capped to free VRAM — 29 slots/layer
today, the same "fixed total budget" every layer already runs under in production, not a number chosen to
make a point). 200 decode tokens, synthetic embeddings (real routing, the trained router still decides which
experts; no real prompt needed for a hit-rate measurement, same reasoning as
`TestMoEStreamingDecodeProfile`). One run — not paired/interleaved, flagged below.

## Result

All 30 layers are MoE (`enable_moe_block` on every layer). Overall pooled hit rate 96.91% (1481 misses /
48,000 routed positions = 30 layers × 200 tokens × topK 8). Per-layer:

| | value |
|---|---:|
| mean | 96.91% |
| stdev | 0.93 pp |
| coldest two (layers 10, 11) | 94.38%, 94.56% (90, 87 misses) |
| next-coldest (layer 17) | 95.75% (68 misses) |
| remaining 27 layers | 96.4%–99.0%, clustered within ~1 pp of each other |
| hottest (layer 4) | 99.00% (16 misses) |

**The unevenness is real but concentrated, not a smooth gradient.** Two of thirty layers (10, 11) sit
~1.9 pp below the mean of the rest; everything else is homogeneous. A raw min/max spread (4.63 pp) reads as
"uneven" by a naive threshold, but that number is driven by two outliers, not a distribution worth re-slicing
the whole budget over.

**Bounded upside, computed, not assumed.** If layers 10 and 11 were pulled down to the ~46.6-miss mean of the
other 28 (via extra slots taken from elsewhere in the same total budget), total misses would drop from 1481 to
~1394 — **~5.7% fewer misses, moving the pooled hit rate from 96.91% to ~97.31% (+0.4 pp).** That is the
ceiling of what Lead 4 can buy at this slot depth, before accounting for the cost of building and validating a
cross-layer pool (a real complexity increase over today's independent per-layer LRUs).

**That upside is now smaller than it would have been a day ago.** `GOINFER_MOE_DMA_OVERLAP` (shipped
2026-09-22, `docs/measurements/moe-streaming-decode-overlap-ceiling-2026-09-22.md`) already hides most of a
miss's cost under concurrent GPU work. A 0.4 pp hit-rate improvement on a cost that is now mostly overlapped
buys less wall-clock than the same 0.4 pp would have pre-overlap.

## Not done, stated

One run, not paired/interleaved against a repeat — this repo's own measurement discipline prefers paired
reads for anything the size of the effect found here (0.4 pp is well within the kind of run-to-run noise this
repo has measured elsewhere, e.g. the ~3.5% CUDA-box session drift). Whether layers 10/11 are structurally
colder (an architectural property worth naming) or an artifact of the synthetic embedding stream and this
one random seed is not distinguished here — a second run with a different seed, or a real prompt, would tell
them apart. Not run, because the answer below doesn't depend on it.

## Verdict

**Park.** Lead 4's own premise (uneven layers) is confirmed, but the measured upside (≤0.4 pp hit rate, and
smaller still after the overlap already shipped) does not clear the bar for building a cross-layer pool —
a real change to the per-layer LRU design, more moving parts, for a bounded, now-diluted win. Revisit only if
a future measurement finds the unevenness is larger on a different model/config, or if per-layer LRU
maintenance itself becomes a measured cost worth removing for its own sake (not indicated by anything here).
