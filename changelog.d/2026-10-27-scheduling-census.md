- **A census of today's scheduling decisions** (SB6). `TestSchedCensus_admission` replays 11 request traces on a virtual
  clock through `serve`'s admission (turn order, the `-max-queue` bound, cancellation, halts, `-max-concurrent`), and
  `TestSchedCensus_placement` replays 10 conversation traces through the KV session LRU (reuse, eviction, KV sharing,
  tiered demotion), against goldens. No behavior changes.
