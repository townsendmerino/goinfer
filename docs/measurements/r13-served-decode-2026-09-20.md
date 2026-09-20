# R13 Build, served — the kernel win is real: 1.32× at depth 8192, parity at 2048

`docs/tasks/red-october.md` R13's Build/Measure: wire aikit's G=6 grouped acc64 kernels
(`docs/measurements/r13-neon-kernel-ab-2026-09-20.md`, 1.53-2.39× faster in isolation) into
goinfer's decode attention path and see whether the served tok/s number moves.

**Bottom line: it does, once a real wiring bug was found and fixed. Three designs were tried before
landing on one that works: Arm A (full-group ownership — a measured wash), Arm B split (measured
1.5-2.2× SLOWER), and Arm B serial (also slower). Profiling the "slower" result with the correct
tool (`go tool trace`'s per-goroutine breakdown, not plain CPU pprof — see "A profiling near-miss"
below) found the real cause: `attendGroupedLayer` had accidentally serialized softmax onto one
goroutine, where the ungrouped path runs it 6-way parallel across the worker pool. Splitting
softmax the same way QK/AV already are closed the gap: parity at depth 2048 (32.33 vs 32.73 tok/s,
~1% overhead, noise-level), and a real 1.32× served speedup at depth 8192 (20.09 vs 15.26 tok/s) —
matching the depth-dependence R13's own step 0 predicted (the isolated kernel's own advantage was
2.39× at depth 8192 against 1.53× at depth 2048). `GOINFER_ATTN_GROUPED` defaults ON.**

## Method

Two measurement tools, used in sequence because the first was too slow to iterate a threading
design against:

1. **`BenchmarkDecodeAtDepth`** (`decoder/decode_depth_bench_test.go`), extended with
   `grouped_off`/`grouped_on` sub-benchmarks (`GOINFER_ATTN_GROUPED`). Real checkpoint
   (qwen2.5-coder-1.5b-instruct-q4_k_m.gguf, `~/models`), real prefill to depth, `-benchtime=30x
   -count=3`. Costs 3-44 minutes per run (prefill to depth is O(depth) and re-run per rep) — the
   tool that surfaced the problem, too slow to debug it with.
2. **`TestZZDiagGroupedFires`** (`decoder/zz_diag_test.go`, a temporary diagnostic kept rather
   than reverted — see its own doc comment and the `GOINFER_ATTN_TIMING_DEBUG` entry in
   `docs/env-vars.md`). Prefills ONCE, times both arms within one process, reports total tok/s, a
   temporary `attnElapsedNanos` atomic (wall time inside `attendBatchedHeads` specifically,
   `GOINFER_ATTN_TIMING_DEBUG=1`), and the `attnGroupedRuns` wiring-proof counter. Costs 15-25s at
   depth 2048 — what made a dozen A/B iterations feasible in one session. Also gained
   `GOINFER_DIAG_CPUPROFILE` (`runtime/pprof`) and `GOINFER_DIAG_TRACE` (`runtime/trace`) hooks
   along the way, both still present, gated off by default.

All runs: qwen2.5-coder-1.5b-instruct-q4_k_m.gguf (NumHeads=12, NumKVHeads=2, HeadDim=128,
group=6, NumLayers=28), Apple M1 Pro, `GOFLAGS=-mod=readonly` against a local, uncommitted
`go.work` override pointing at aikit commit `af926e3` (not yet released — see the aikit kernel A/B
record's own note on this).

## Three wiring designs tried before the fix

**Arm A (`attendGroupedHeads`/`runHeadRange`), full-group ownership.** One worker owns a WHOLE
6-head kv group (no internal split). A real dispatch bug (below) meant it never fired at first;
once fixed, it forces worker count down to `nKV=2` on this model (from 6) whenever it fires.
Served result: `BenchmarkDecodeAtDepth` showed grouped on/off as noise-level identical at both
depth 2048 (31.33 vs 30.92 tok/s median) and depth 8192 (15.77 vs 16.33 tok/s median) — a wash,
consistent with trading 6-way head parallelism for a more efficient kernel call roughly cancelling
out.

**A real bug, found before trusting the first "no effect" number.** `headWorkerPool`'s existing
`headsPer := (nH+workers-1)/workers` gives 2 heads/worker on this model, and Arm A's gate required
a worker's range to contain a WHOLE 6-head group — which a 2-head range never does, so
`attnGroupedRuns` stayed at 0. Fixed by rounding `headsPer` up to a multiple of `group` when
eligible. Caught by the wiring-proof counter, exactly the mechanism R13 Gate (4) exists for.

**Arm B, split-within-group.** QK split by KEY range, AV split by DIM range — both independent-
output-axis splits, no floating-point combining needed. Iterated with the fast diagnostic:

| variant | attn-only / 30 steps (depth 2048) | vs baseline (325ms) |
|---|---|---|
| baseline (ungrouped, today) | 325ms | — |
| spawn-per-task, fork-join per (kv head × phase) | 717ms | 2.2× |
| + persistent worker pool (no `go func(){}()` per round) | 717ms | 2.2× (no change) |
| + coarsened to 1 fork-join per phase per LAYER | 621ms | 1.9× |
| + NEON-block-aligned split boundaries | 486ms | 1.5× |
| + fewer split-workers (3, then 2) | 511ms, 549ms | worse — reverted |

**Arm B, serial-per-kv-head.** One un-split kernel call pair per kv head, fanned out across kv
heads via one fork-join per layer — the fewest synchronization points of any design. Measured
**667ms — WORSE** than the aligned-split version, despite less coordination overhead by every
measure that should matter. This contradiction (simpler design, worse result) is what motivated
profiling instead of further threading guesses.

## A profiling near-miss, and the tool that actually worked

A CPU profile (`go tool pprof`) of grouped-on showed `runtime.pthread_cond_wait` at 62%,
`pthread_cond_signal` at 12.6%, `usleep` at 9.8% — ~85% of samples in apparent scheduler wait,
pointing at aikit's `linalg.parallelSpawnCols` (MLP matmul fan-out) as a pre-existing, decode-wide
bottleneck unrelated to attention. **This matched a prior investigation in this exact repo
(memory: "CPU-decode 8ms is an idle-M artifact") closely enough that it should have been the first
thing checked**: that record already established plain CPU pprof MISCOUNTS idle parked workers as
real cost, and the correct instrument for a park/wake question is `go tool trace`, not `pprof`.
Caught before writing a campaign brief around it, not after — but only because of an explicit
prompt to double-check rather than proceeding.

`go tool trace -pprof=sched` and `-pprof=sync` on the same traces gave numbers that didn't
reconcile cleanly with the measured wall-clock gap either (a `chansend1` delay reading larger than
the actual difference), and a concrete falsification (enlarging the worker-pool channel buffer
100×) changed nothing — ruling that theory out too, empirically rather than by further reasoning
from profile aggregates.

**What actually worked: the trace tool's per-goroutine-group breakdown** (`/goroutines`, then
drilling into specific groups via `go tool trace -http`, queried directly with `curl` since no
browser was available). This gives exact, non-overlapping time breakdown per goroutine (execution
/ blocked-on-channel / blocked-on-sync / sched-wait), not a sampled aggregate. The main goroutine
(`testing.tRunner`) showed:

| run | total | execution time | block time (sync) |
|---|---|---|---|
| baseline (150 steps) | 4.697s | 1.445s (30.8%) | 3.192s (68.0%) |
| grouped, before the fix (150 steps) | 5.783s | 2.202s (38.1%) | 3.515s (60.8%) |

The main goroutine's OWN execution time grew by ~755ms between the two runs — almost exactly the
wall-clock gap. That pointed directly at real, additional SERIAL work on the calling goroutine,
not scheduler contention. Reading `attendGroupedLayer`'s source with that lens found it
immediately: softmax was written as a single serial loop over every kv head's every row, run on
whichever goroutine calls `attendGroupedLayer` — while the ungrouped path
(`attendOneHead`) runs each head's softmax INSIDE that head's own worker, i.e. already 6-way
parallel today. R13 step 0(ii) (`r13-attn-category-split-2026-09-19.md`) had already measured
softmax at ~21-26% of this model's attention time; serializing something that used to run 6-way
parallel is a real, direct, fully mechanistic cost — no exotic contention theory required.

## The fix and the result

Softmax was rewritten to split by ROW (one kv-head's one query-head's softmax is independent of
every other row — the same "split the independent axis" argument the QK/AV splits already use,
via `runSplitAligned(splitWorkers, nKV*group, 1, ...)`, one level finer) instead of running serially.

| depth | grouped_off | grouped_on | delta |
|---|---|---|---|
| 2048 (150 steps, repeated twice) | 32.73-32.94 tok/s | 32.33-32.35 tok/s | ~1% overhead, noise-level |
| 8192 (100 steps) | 15.26 tok/s | 20.09 tok/s | **1.32× faster** |

Attention-only wall time at depth 8192: 43.3ms/token (ungrouped) → 29.3ms/token (grouped), a
1.48× reduction — consistent with the isolated kernel A/B's own depth-dependence (2.39× at depth
8192 vs 1.53× at depth 2048, `r13-neon-kernel-ab-2026-09-20.md`) once softmax's now-equal
parallelism stops masking it.

## What is and isn't established

**Established:** the wiring is correct (gate-1 bit-identical throughout every version, `go test
-race` clean, direct on/off comparison in `TestAttendGroupedHeads_matchesPerHead`/
`TestAttendGroupedLayer_manyWorkers`); a real served win exists at depth 8192 on this model/machine;
parity (not a regression) at depth 2048; the softmax-serialization bug was the actual cause of
every "no effect" and "slower" reading before this record, not scheduler contention, kernel
inefficiency, or channel design. **Not established:** the served effect at depths between 2048 and
8192, or above 8192; other models (0.5B, 7B, phi3-mini — the brief's own registered set); the
Linux box; whether Arm B (still unused — this ships as the softmax-fixed Arm A-successor,
`attendGroupedLayer` doing full un-split kernel calls with parallel softmax) would do even better
with its own key/dim-range splitting now that the softmax confound is gone — untested, since the
simple fix already won and re-testing Arm B was not repeated after this fix landed.

## Disposition

- `GOINFER_ATTN_GROUPED` defaults **ON**.
- `attendGroupedLayer` (the final, shipped version): one un-split `MatmulQKAcc64Group`/
  `MatmulAVAcc64Group` call pair per kv head (reusing `attendGroupedHeads`'s per-kv-head logic),
  fanned out across kv heads via the persistent `globalAttnWorkerPool`, with softmax ALSO
  parallelized across the same pool (the fix this record is about). Arm B's split-based code
  (`runSplitAligned` used for key/dim-range splitting within one kv head) is not currently
  exercised by any live call site but remains in the tree, correct and tested, as a documented
  dead end pending re-evaluation now that the softmax confound that made it look worse than it
  was is fixed.
- R13 status: **shipped**, not parked.

## Lesson

A "slower once wired in" result that contradicts the isolated kernel A/B is not evidence the
kernel doesn't generalize — it can equally mean the wiring itself introduced an unrelated
regression. Three separate wiring designs measured "slower" before the actual cause (a parallelism
regression in code adjacent to, not inside, the new kernel calls) was found — and it was found by
comparing exact per-goroutine time breakdowns between the two runs, not by reasoning from
aggregate CPU-sample profiles, which pointed at the wrong subsystem entirely on the first attempt.
