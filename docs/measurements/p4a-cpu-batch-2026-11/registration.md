# P4a: CPU batched decode on f16 and int8 caches, not slower (registration)

Registered 2026-11-09, before the run. Ledger row P4, `docs/tasks/task-kv-precision-everywhere-2026-10.md`. Tier 1: P4a changes what runs
by default for multi-client CPU serving on an f16 or int8 cache. Bit-identity is held at tier 0, and this read checks speed.

- **Question:** with `-kv f16` or `-kv i8` on a model `-cpu-batch auto` batches, is #874 (the batcher reads the widened cache) at least as fast as
  main (each sequence decodes on its own)?
- **Instrument:** `run-p4a.sh` beside this file. nobara CPU, qwen2.5-7b-instruct Q4_K_M from `~/models` (over `cpuBatchAutoMinBytes`, so
  `auto` engages the batcher; the 1.5B is under it, and P4a does not change what it runs by default), serve defaults,
  `bench_w7_plain.py`'s 6 turns at 32 tokens, greedy, a fresh server per cell. A is main at #874's base, B is #874. A B B A.
- **Cells:** f16 and i8 at 4 clients, 10 pairs each (decisive); at 2 clients, 3 pairs each (reported).
- **Metric:** aggregate decode tok/s B/A per pair; the median per-client tok/s beside it; replies (`content_sha`) equal in every turn.
- **Bar** (`grade-p4a.py`): PASS when, in both decisive cells, replies are equal, the median aggregate ratio is at least 1.00, and B is at least as
  fast on at least 9 of 10 pairs. Otherwise FAIL.
- **Prediction:** about MC3c step 2's 4-client gain, 2.2× (`docs/measurements/concurrency-mc3c-step2-2026-09-27.md`), less the widening
  copy, which the per-sequence path already pays.
- **Stopping:** fixed, every pair once. A cell started on a busy box is waited out by the script's idle gate; a run that stops at the gate is
  re-run from the missing pairs, listed, never averaged in.
- **Outcomes:** PASS: #874 merges. FAIL: #874 stays open, and the reading and a mechanism go to Francis.
- **Wall:** about 55 minutes: 52 cells of the 7B at about a minute each (load, six turns of 32 tokens per client), plus the two builds.
