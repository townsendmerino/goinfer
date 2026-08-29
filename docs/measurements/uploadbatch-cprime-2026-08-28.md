# `gpu.UploadBatch` in the C′ expert cache — +9.3% tok/s, and the sync-term estimate was right

**2026-08-28, nobara-pc (RTX 2070 SUPER, driver 595.91.07, 62 GB, 16 core), gemma-4-26b-a4b-it
int4, C′ expert cache at 30 slots/layer, 64 decode tokens, capture OFF.** Task:
`docs/prompts/goinfer-adopt-uploadbatch.md`. Arms interleaved within one session, two passes each,
`-count=1`.

## Result

| arm | tok/s (pass 1, 2) | mean | ms/tok | H2D total |
|---|---|---|---|---|
| unbatched (per-copy sync) | 14.60, 14.49 | 14.55 | 69 | 1.868 s, 1.872 s |
| **batched (one sync/layer)** | 15.74, 16.06 | **15.90** | 63 | 1.673 s, 1.685 s |

- **Copies → synchronizes: 20,916 → 2,038 (10.3× fewer).** Reproduced identically on every batched
  run. This was the pre-registered gate and it is a counting question, not a timing one.
- **+9.3% tok/s.** Arm spreads are ±1.0% (batched) and ±0.4% (unbatched), so the effect is an order
  of magnitude outside run-to-run noise.
- **H2D time −10.2% = 2.98 ms/token.** The task predicted "~3.0 ms of the 3.6 ms recovered". That
  is a direct hit on the absolute figure.
- End-to-end saves ~6 ms/token against ~3 ms measured inside the H2D path, so roughly half the win
  is outside the copy accounting — consistent with a full-device `Synchronize` draining work
  beyond its own transfer. Not chased further; noted so the next person is not surprised.
- **Bit-exact on hardware:** `TestGemma4MoE_cacheExpertsBitExact_{tiny,scaled}` and the two
  `cacheReuse` tests all `--- PASS` under `-tags 'cuda goinfer_testhooks'`. Batching changes when
  copies are issued, never what is copied.

## Two ways this measurement lied before it told the truth

Recorded because both produced confident, wrong numbers that looked like results.

**1. The box was not quiesced.** The first batched run executed immediately after a 50-minute
CPU-saturating prefill job on the same machine and returned **105 ms/tok** — a 35% *regression*.
Every later batched run returns 62-64 ms/tok. The unbatched arm was stable throughout (68, 69, 69),
so only the contaminated arm moved, which is precisely how a confound passes for a finding. The
task doc warned about exactly this and cited a 5.46-vs-10.74 tok/s incident for byte-identical
work; the warning was quoted and then not followed.

**2. `go test` result caching replayed four "runs" in one second.** A batched/unbatched pair loop
produced four `--- PASS` lines with byte-identical durations and timestamps one second apart. The
tell — `(cached)` — was filtered out by the `grep` capturing the output. **Use `-count=1` for any
timing run, and capture output whole, filtering when reading rather than when writing.** The same
capture-time filter also silently dropped the tok/s line (a `.` in the pattern matching one byte of
a multi-byte em dash), which is why the first two attempts had no throughput number at all.

## Scope

Single model and single slot count. The win is a fixed per-sync cost amortized over a layer's
misses, so it should scale with miss rate and shrink toward zero for a fully-resident model that
misses rarely — untested here. The declined async variant is not revisited: it buys the same term
and moves ordering into the caller's hands.
