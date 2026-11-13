# P41 — R13's grouped decode attention, on against off, across depth (registration, for Francis before any run)

**Registered 2026-11-09, before any sample.** Owner: `docs/queue-performance.md`, P41. It can settle a default
(`decoder/cpu_tuning_arm64.go:attnGroupedKernels`, on), so Francis approves it before it runs.

**Question.** On arm64, is grouped attention (`GOINFER_ATTN_GROUPED=1`) still at least as fast as per-head (`=0`) at
every depth where it runs, and still a win deep? It shipped at 1.32x served at depth 8192; no sweep across depth followed.

**Scope.** It runs only where the query-to-KV head ratio is 6 (`attnGroupedNEONSize`) and from 128 attended keys
(`attnGroupedMinKeys`). Of the bench models that is Qwen2.5-1.5B alone (0.5B and 7B are 7, phi3-mini 1). On amd64 both arms
run the per-head path (`decoder/cpu_tuning_other.go:attnGroupedKernels` is off: aikit's grouped kernels are NEON only),
so there is no Linux half until an AVX2 port exists.

**Instrument.** The Mac (M1 Pro), by day, quiet, under the exclusive timing lock. `BenchmarkDecodeAtDepth`'s two arms,
`grouped_off` then `grouped_on`, in one process. Qwen2.5-1.5B-Instruct Q4_K_M from `~/models`, `GOINFER_BENCH_QUANT=int4`,
`-benchtime 30x`. Depths: 128, 2048, 4096, 8192. Ten runs per depth, each run a pair. Control: Qwen2.5-7B (ratio 7, so both arms
are the same work) at depth 2048, ten runs. First, `TestAttendGroupedHeads_matchesPerHead` must pass (bit-identity, and the
grouped dispatch counted).

**Metric.** Per pair, on ÷ off tok/s. The statistic is the median of ten per depth.

**Void.** The heavy-load rule: a run started with the 1-minute load average above 4.0, or free memory below the load's fit
plus 2 GB, is re-run, and the void is listed. The control reads the fixed arm order: a control median outside 1.00 ± 0.02
voids the sweep.

**Stopping rule.** Fixed N, no early stop. About 25 minutes: prefill to 8192 dominates.

**Outcomes.**
- **Keep the default:** median ≥ 0.98 at every depth, and ≥ 1.05 at 8192 with at least 9 of 10 pairs faster.
- **Raise the floor:** median below 0.98 with at least 8 of 10 pairs slower at a shallow depth, while 8192 keeps. Then
  `attnGroupedMinKeys` moves to the smallest depth measured that keeps, and Francis makes the call.
- **Turn it off on arm64:** median below 0.98 with at least 8 of 10 pairs slower at 8192.
- **Ambiguous → parked:** anything else. To Francis with the numbers.

**Run (approved by Francis, 2026-11-09).** On the Mac, by day, quiet, from a clean checkout of `main` at a commit that has this
file (the runner records the commit):
```
ls ~/models | grep -iE 'qwen2\.?5.*(1\.5b|7b).*instruct.*q4_k_m\.gguf'   # the two models; any other quant or size is not this sweep
bash docs/measurements/r13-grouped-sweep-2026-11/run-r13.sh ~/models/<the 1.5B .gguf> ~/models/<the 7B .gguf>
```
`run-r13.sh` refuses a model outside `~/models`, a non-arm64 host and a dirty `decoder/`. It builds `decoder.test` and
`cmd/hostmem` once into `~/goinfer-bench/r13/<UTC time>-<commit>/`, and runs `r13_ab.py` under the exclusive timing lock
(`timing_lock.py run --label p41-r13-sweep`). `r13_ab.py` runs `TestAttendGroupedHeads_matchesPerHead` first (VOID unless
it passes), then the 40 pairs and the 10 control runs. Before each run it waits until the 1-minute load is under 2.0 (a
mechanism, set before any run: a run's own load must decay before the next starts), up to 300 s, and applies the void rule
above with `hostmem -reclaimable`. It writes `raw.jsonl` and `grade.txt` (the outcome as a `VERDICT:` line, graded by the
rules above) there and beside this file; commit those two. About 30 minutes.

## Addendum, 2026-11-09, before the first full run (Francis: control (b))

A mechanism fix; no bar, depth, N or rule moves.

- **Control: Qwen2.5-Coder-0.5B-Instruct Q4_K_M** (`~/models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf`), at depth 2048,
  ten runs, in place of Qwen2.5-7B. The 7B cannot load on the Mac's CPU path: `fit -profile m1pro-near-idle -backend cpu
  -quant int4` refuses it (8.0 GB int4 resident, 4.4 GB of the mapped `.gguf` and 3.5 GB of KV against an 8.5 GB budget), so every
  control run would have been void. The 0.5B is ratio 7 (14 query heads, 2 KV), so both arms run the per-head path, as the
  7B's would have; it fits (1.9 GB of 8.5).
- **The 1.5B is Qwen2.5-Coder-1.5B-Instruct Q4_K_M** (`~/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf`), the match of
  the run command's own `ls | grep` and `bench_peer.py`'s "1.5B". It has the same architecture as Qwen2.5-1.5B-Instruct
  (ratio 6).
- `r13_ab.py`: the control's fit term is 2 GB (was 6, the 7B's), and the grade line names the control by file.
- **Run:** a night job from 22:00 (`night.py`), a full restart: `r13_ab.py` has no resume, and one session is cleaner
  than day pairs mixed with night ones. The settle gate (load1 < 2.0) is why: by day each run's own load took 80-300 s to
  decay under it, so the sweep projected to about 3 h.
- **Pilot, not graded:** a day start at d9875a08e on 2026-11-09, 16:33-17:03, stopped on Francis's word. Identity
  passed; depth 128, ten pairs: median on/off 1.0072, 10 of 10 faster; depth 2048, seven pairs: median 0.9294, 0 of 7 faster;
  no voids. Its `raw.jsonl` is kept as `raw-pilot-2026-11-09.jsonl` beside this file; the graded sweep is the night run alone.

## Addendum 2026-11-10 (coordinator; mechanism fix, no bar moved)
The 2026-11-09 night run timed out at its 6 h limit (04:26) after 40 runs (128 x9, 2048 x10, 4096 x10, 8192 x9; raw-sweep-2026-11-09.jsonl) and never ran its control. Francis chose a control-only re-run: run-r13.sh with R13_PRIOR=<that raw> passes --control-only --prior to r13_ab.py, which runs the 0.5B control (10 pairs at depth 2048) in a separate session and grades the sweep runs with it, by the unchanged rules. 8192 has 9 clean pairs, not 10; the KEEP rule needs 9 of 10 faster, so the grade counts 9 of 9 as met. r13_ab.py and run-r13.sh originals are kept as *.orig-2026-11-09.

## Reading, 2026-11-12: the default stays (Francis)

The sweep is the 2026-11-09 night's 38 pairs (`raw-sweep-2026-11-09.jsonl`: 128 and 8192 have nine pairs each, because the
job hit its 6 h limit) and the control is the 2026-11-10 night's ten runs (`raw.jsonl`), a separate session, graded together
in `grade.txt`. M1 Pro, CPU path, Qwen2.5-Coder-1.5B-Instruct Q4_K_M from `~/models`, int4, at d9875a08e.

| depth | median on ÷ off | pairs faster |
|---|---|---|
| 128 | 1.0153 | 7 of 9 |
| 2048 | 0.9402 | 1 of 10 |
| 4096 | 0.9517 | 1 of 10 |
| 8192 | 1.2638 | 9 of 9 |

- **Control:** 1.0013 at depth 2048 on the 0.5B, inside 1.00 ± 0.02. Two void runs were re-run.
- **By the registered outcomes this is "raise the floor"**, and the scorer printed `attnGroupedMinKeys` to 128, which is its
  value today. The rule assumed a curve that loses shallow and wins deep. This one keeps at 128, loses 5 to 6% at 2,048 and
  4,096, and wins 26% at 8,192, so "the smallest depth measured that keeps" names a depth below the loss. Nothing was measured
  between 4,096 and 8,192, so the crossover is unknown.
- **Francis, 2026-11-12: leave the default as it is** (`attnGroupedKernels` on, `attnGroupedMinKeys` 128), and record the
  loss at mid depths as a known cost. The reach is one head ratio on the arm64 CPU path, where Metal is the usual backend. No
  further run is planned; a floor change would first need the crossover read between 4,096 and 8,192.
- **Limits:** one model, one machine; nine pairs at two depths; the control from another night.
