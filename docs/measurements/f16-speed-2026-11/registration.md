# F16-speed: CUDA f16 against f32 KV, served decode on the paged KV (registration)

Registered 2026-11-09, before the run. Ledger row F16-speed, `docs/tasks/task-kv-precision-everywhere-2026-10.md`. This is the paired speed
check that `docs/completed/task-cuda-f16-kv-2026-10.md` registered (§ "The paired speed check") and that was stopped before its deep cells
when the owner made f16 CUDA's `auto` (§ "Speed read and owner override"). It runs unchanged, on the KV that shipped since: the paged KV.
Tier 2: it settles whether the override stands as a pass.

**Question.** On the shipped tree, does `--kv f16` (the `auto` pick for CUDA's dense llama, mistral, qwen2, qwen3 and phi3) decode as fast as `--kv f32` at depth?

**Instrument.** `run-f16-speed.sh` beside this file: one serve binary, arms `-kv f16` (goinfer) and `-kv f32` (goinfer_old), `bench_peer.py`'s
two-goinfer form, greedy, five runs a cell, the sweep run twice with the arm order reversed, the timing lock held, nobara. Every serve log must
show `decode path: cuda-resident` and the arm's KV as delivered, or the pass is void.

**Cells.** 0.5B, 1.5B (qwen2.5-coder, q4_k_m) and phi3-mini at 512, 2048 and 3900 keys; qwen2.5-7b at 8000 (context 8448). The 128-key cells
are not re-run: the new kernel read them at 1.006 to 1.013 and the question is depth.

**Metric.** Per cell, the f16 arm's median per-run decode rate over the f32 arm's, pooled over both orders
(`docs/measurements/cuda-f16-kv-2026-10/grade-speed.py`, unchanged); each order alone is printed beside it.

**Bar** (as registered). PASS: every cell >= 0.98. PARKED (ambiguous): the worst cell in 0.95 to 0.98. REGRESSION: any cell below 0.95. Prediction,
carried over: 7B at 8000 reads >= 1.10, no cell under 0.98.

**Stopping rule.** Fixed: both orders, every cell, no early stop. A pass refused for load, a void cell (a missing arm, a CPU path, a KV not
delivered) or a cell started under heavy load is listed and re-run on the next night, never averaged in. The run is not repeated for a better read.

**Outcomes.**
- PASS: the override becomes a pass; F16-speed closes and the matrix's "Measured cost" notes gain the deep-cell figures.
- PARKED: `auto` stays f16. The figures are published as a cost, and Francis decides whether any family's `auto` changes.
- REGRESSION: the slow cells are published as a cost of `auto`, with a mechanism sought before any re-run. Whether `auto` goes back to f32 for the
  slow family is Francis's call: a default moves only on his word.

**Night recipe** (nobara, after L5's day A/B; about 70 minutes, the timing lock exclusive for each pass). `<sha>` is origin/main when it is
queued; the script is taken from that commit, so the nobara checkout's working tree is not touched:

    cd ~/mycode/goinfer && git fetch -q origin && mkdir -p ~/goinfer-bench/f16-speed && git show <sha>:docs/measurements/f16-speed-2026-11/run-f16-speed.sh > ~/goinfer-bench/f16-speed/run.sh
    env REV=<sha> DRY=1 bash ~/goinfer-bench/f16-speed/run.sh
    python3 scripts/night.py add f16-speed --est 70 --by "<who>, F16-speed" --doc docs/tasks/task-kv-precision-everywhere-2026-10.md -- env REV=<sha> bash ~/goinfer-bench/f16-speed/run.sh

The DRY run (seconds plus the build) checks the checkout, the four `~/models` paths and the build. Grade with
`python3 docs/measurements/cuda-f16-kv-2026-10/grade-speed.py <outdir>`, `<outdir>` being the dated `speed-` directory under `~/goinfer-logs/f16-speed`.
