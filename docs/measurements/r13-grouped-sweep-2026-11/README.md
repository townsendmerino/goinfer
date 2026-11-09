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

