# P42 (R15's open half): the min-p candidate scan on all cores ships; the top-k arm is held for a call

<!-- citations-at: bd14f8c2b -->

Question and rule: `docs/queue-performance.md` P42. Code: `internal/runtime/sampling/sampler_chunked.go:minPCandidates`, called
from `topFilterLogits`'s min-p case; its serial reference is `minPScanSerial` in the benchmark file
(`internal/runtime/sampling/sampler_filter_bench_test.go`). Bit-identity: `TestMinPCandidates_matchesSerialScan` (slices.Equal on
the unsorted candidate list, so the ascending-id order is kept) and `TestTopFilterLogits_MatchesReference`.

## Provenance

| | |
|---|---|
| machine | MacBook Pro, Apple M1 Pro (8 performance and efficiency cores as `GOMAXPROCS` 8), 16 GB, arm64, macOS 26.x (Darwin 25.6.0) |
| workload | in-process Go benchmarks, no checkpoint: random logits at vocab 262144, 151936 and 32064; no greedy or seed question |
| versions | Go 1.27.2; goinfer `p42-minp-parallel-2026-11-09` merged with main (`bd14f8c2b`); no aikit code on the path |
| dates | A/B 2026-11-09 16:08-16:12 PST; min-p re-time on the wired form 2026-11-09 16:24-16:31 and 2026-11-13 12:51 PST |
| thermal | not recorded (no sensor read); the arms alternate inside one `go test`, so slow drift cannot pose as an effect |
| disk | not read: nothing is loaded from disk |
| lock | `scripts/timing_lock.py run --label r15`, exclusive, one cell at a time |

## 1. The A/B (2026-11-09), the arms as first built

Test-only arms at `origin/main` 971c0bbc, 10 iterations of `go test ./internal/runtime/sampling -run '^$' -bench
'TopKByLogit_gemmaVocab|MinPScan_gemmaVocab' -benchtime 2000x -count 1`; one void (load1 4.04), re-run.

| component | serial median | parallel median | ratio median (range) | pairs faster |
|---|---|---|---|---|
| min-p scan, vocab 262144 | 165.4 us | 74.7 us | 2.221x (1.865-2.247) | 10/10 |
| top-k, k = 1 | 260.4 us | 98.0 us | 2.651x (2.553-2.683) | 10/10 |
| top-k, k = 8 | 250.0 us | 147.8 us | 1.693x (1.620-1.716) | 10/10 |
| top-k, k = 40 | 258.5 us | 421.1 us | 0.615x (0.594-0.618) | 0/10 |

Min-p cleared its rule (median at least 1.2x with 9 of 10 faster). The first form allocated a slice per chunk on every call,
where the serial scan reuses its scratch, so the shipped form (below) was re-timed. Top-k at k = 40 loses because each of the
64 chunks keeps its own top 40 and the 2,560-entry union is then sorted, which costs more than the serial heap saves.

## 2. The wired min-p scan, re-timed (the rule written before the run)

`minPCandidates` scans by `forEachChunk`'s fixed chunks, each chunk writing its ids into its own span of the caller's
vocab-sized scratch, then compacts the runs in chunk order: no per-call candidate allocation, the same ids in the same order.
Ten pairs, the loop as in section 1 with `-bench MinPScan`. Rule: SHIP at vocab 262144 if the median serial ÷ wired is at least
1.2 with at least 9 of 10 pairs faster; and because the wired form runs at every vocab of 4096 or more, and R15's max-scan lost
3.78x at 32k to the same goroutine overhead, 152k's and 32k's medians must be at least 1.00, else the form does not ship
unconditionally. A pair is void if load1 is above 4.0 or reclaimable memory under 2 GB at its start; it is listed and re-run.

| vocab | serial median | wired median | ratio median (range) | pairs faster | rule |
|---|---|---|---|---|---|
| 262144 | 178.4 us | 66.6 us | 2.686x (2.355-2.717) | 10/10 | SHIP |
| 151936 | 96.2 us | 51.2 us | 1.884x (1.785-1.906) | 10/10 | at least 1.00: holds |
| 32064 | 20.3 us | 19.3 us | 1.050x (0.861-1.082) | 9/10 | at least 1.00: holds |

Per-pair ratios, pairs 1 to 10, vocab 262144: 2.679 2.682 2.690 2.717 2.355 2.710 2.700 2.689 2.677 2.653. Vocab 32064:
1.058 1.070 1.035 1.050 0.861 1.050 1.067 1.082 1.025 1.048 (pair 5's 0.861 is one slow-serial-fast pair; the other nine are
1.025 to 1.082). Raw output of the ten graded cells: `raw-retime.txt`.

**Voids, listed.** Pairs 1 to 6 first ran on 2026-11-09 16:2x with load1 above 4.0 and were re-run (pair 5 at load1 3.98, pairs 1,
2 and 3 at 3.16 to 3.56); pairs 4 and 6 stayed void through three reruns each (load1 4.04 to 8.62) and were run on 2026-11-13
12:51 PST at load1 3.68 and 3.62 (reclaimable 6.9 GB). Pairs 7 to 10 started at load1 3.36 to 3.82.

## Reading

Min-p ships: the wired scan is 2.7x the serial scan at the Gemma vocab, 1.9x at Qwen's, and 1.05x at 32k, where the one slower
pair is still inside the noise of a 20 us operation. Output is unchanged, so no fidelity gate applies. The gain is about 110 us
a sampled token at Gemma vocab with min-p on.

Top-k is not shipped and not killed. Its parallel arm wins at k = 1 and k = 8 and loses at k = 40; shipping it needs a k cutoff
(parallel at k up to 8, or wherever it crosses 1.2x between 8 and 40, which is unmeasured) and the serial path above it. That is
a second path and a new bar, so it waits for Francis. The arm stays test-only
(`internal/runtime/sampling/sampler_filter_parallel_test.go`).
