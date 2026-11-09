# L5 — amd64 split-half against canonical int4 decode, on today's kernels (registration)

**Registered 2026-11-09, before any sample of this run exists.** Owner: `docs/tasks/task-int4-layout-2026-09.md`, row L5.
Francis's bar (2026-11-09): one nobara A/B by day; kill under +2% at equal RSS.

**Question.** Does the AVX2 split-half W4A8 layout still beat canonical int4 on single-token decode on nobara (Zen 2, AVX2, no
VNNI) with today's kernels? The 2026-08-30 A/B read +2.10% (`docs/measurements/w4a8-splithalf-decode-ab-PREREGISTERED.md`);
canonical's kernels have moved since. Only a split-half-only build (canonical dropped under `Backend: "cpu"`) makes RSS equal,
so this run decides whether that build is worth writing. RSS here is recorded, not graded: the ON arm still holds both copies.

**Instrument.** `BenchmarkDecode`, Qwen2.5-Coder-1.5B-Instruct Q4_K_M from `~/models` on nobara, `GOINFER_BENCH_QUANT=int4`,
`-benchtime 30x -count 4`, batch 1, greedy. One test binary for both arms; the arm is `GOINFER_W4A8_SPLITHALF=1` (ON) or unset
(OFF). First, `TestW4A8SplitHalfFires_onBenchModel` must report a non-zero repacked count, or the run is VOID. Order ON OFF OFF ON,
five times: ten adjacent pairs. Each visit is its own process; its first sample is discarded (the 2026-08-30 rule, unchanged).

**Metric.** Per pair, the ON visit's median retained tok/s over the OFF visit's, minus 1. The statistic is the median of the ten.

**Void.** A visit started with the 1-minute load average above 8.0, or `MemAvailable` below 5 GB (the ON arm's fit, about
3 GB, plus 2 GB), is VOID: listed in the record and re-run at the end in the same arm, up to four re-runs; never averaged.

**Stopping rule.** Fixed N: ten pairs, no early stop.

**Outcomes.**
- **KILL L5:** median gain below +2.0%. The row is killed and the layout's opt-in stays as it is.
- **BUILD:** median gain at or above +2.0% and at least 9 of 10 pairs positive. Write split-half-only under `Backend: "cpu"`
  (a new `.giw` kind for `cpu-amd64`); its merge gate is a tier-0 byte count no higher than canonical's plus a by-day re-run of
  this A/B against the build.
- **Ambiguous → parked:** median at or above +2.0% with fewer than 9 positive pairs. To Francis with the numbers.

**Run.** `bash docs/measurements/l5-splithalf-2026-11/run-l5.sh` on nobara, by day, under the exclusive timing lock (the
script takes it). About 15 minutes: the build, the fires check, and twenty visits of about 25 s each.
