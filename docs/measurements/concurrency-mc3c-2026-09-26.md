# MC3c step 1 — concurrent decode on CPU: 4 clients at 1.86–1.97× the serialized aggregate, p99 halved, output byte-identical (2026-09-26)

MC3c of [`task-concurrency-2026-09.md`](../tasks/task-concurrency-2026-09.md), step 1 (the owner chose "cpu first"
2026-09-26): `serve -max-concurrent N` lets one CPU model run N generations at once, each on its own session KV.
MC2 measured the mechanism in-process ([`concurrency-mc2-2026-09-26.md`](concurrency-mc2-2026-09-26.md): J8's 4
workers, 2.00–2.48×). This is the serving-path version, graded on the W7 plain workload against gates registered
before the build.

**Result: all five gates pass.**
- At 4 clients, `-max-concurrent 4` reads **86.4 / 79.2 / 84.3 tok/s** against the same binary serialized at 43.9 /
  42.6 / 44.0: **1.97×, 1.86×, 1.91×**.
- p99 per-turn latency under that load falls from 12.8–13.4 s to **6.5–7.0 s**.
- Each conversation's output is byte-identical to serving it alone, and reuse is exactly the serialized run's.
- `-max-concurrent` **stays default 1** until the owner decides a CPU default: it trades a lone request's latency for
  throughput.

## What was built

- **Admission** (J1) is N-wide and still strict FIFO (`admission.setCap`). A zero value admits one, so every existing
  admission test is unchanged and passes.
- **Session check-out** (`sessionLRU.busy`):
  - `acquire` never hands out, evicts, demotes or saves a session another generation holds;
  - it never reads that session's tokens either, because its generation is writing them;
  - `drive` checks the session back in once the stream has drained (the generation reconciles the session before
    closing it).

  `sessMu` is now held around LRU operations only, not across a generation. A comment in `main.go` had already
  claimed that; it is now true.
- **Eligibility** (`loadedModel.setConcurrency`): N = `-max-concurrent`, but 1 unless `decoder.Model.CPUConcurrentSafe`
  holds (not GPU-resident, no expert or layer pager) and there is no vision tower. N is capped by `-kv-sessions`,
  because each running generation holds a session. The queue holds N running plus `-max-queue` waiting.
- **Prefill memory:** each request must fit in 1/N of the safety margin (`AdmitPrefillMemoryShare`). Concurrent
  prefills share available memory that each one's check reads before any has allocated.
- **Banner:** a "concurrency:" line with the count, naming a `--kv-sessions` cap or why a model runs one.
  **Docs:** `server.md`.

## Gates

| gate (registered before the build) | result |
|---|---|
| 1. each conversation's response served concurrently is byte-identical to it served alone | **pass.** `TestConcurrentSessions_matchAlone` (decoder: 4 conversations × 3 turns on one `Model`, llama-tiny and qwen2.5-coder-0.5b int4). `TestServe_maxConcurrentMatchesAlone` (the real `tryEnter` → `drive` → `exit` path, 4 conversations × 3 turns × 24 tokens on the 0.5B at `-max-concurrent 4`: identical text every turn, reuse on every turn). |
| 2. `-race` clean | **pass**: both tests above, `TestAdmission_*`, `TestSessionLRU_checkout` and the adapter tests under `-race` ([`serve-race.log`](concurrency-mc3c-2026-09-26/serve-race.log)) |
| 3. reuse unchanged: every turn's prompt − reused equals the 1-client run's | **pass**: 26, 27, 24, 20, 27 on every client, in every cell |
| 4. W7 4-client aggregate ≥ 1.2× the serialized build's (N = 4) | **pass: 1.97×, repeats 1.86× and 1.91×** |
| 5. p99 per-turn latency at 4 clients no worse than serialized at 4 clients | **pass: 6.5–7.0 s against 12.8–13.4 s** |

Unit tests: `TestAdmission_capN` (N holders, FIFO hand-off) and `TestSessionLRU_checkout` (a busy session is never
handed out; with every session busy and the LRU full, a transient session is served; checkin makes it reusable; save
skips a busy one). `TestBanner_concurrency`.

## Measurement

- M1 Pro 16 GB, macOS 26.6.2, CPU backend.
- qwen2.5-coder-1.5b-instruct q4_k_m, `-quant int4`, `-kv-sessions` 4.
- `scripts/bench_w7_plain.py` (gains `--serve-args`): 6 turns per client, 128 tokens, greedy, a fresh server per cell.
- Arms:
  - **old** = the MC1 build, serialized (`serve-metal-mc1`);
  - **n1** = the MC3c build at the default `-max-concurrent 1`;
  - **n4** = the MC3c build at `-max-concurrent 4`.
- One session, idle-gated per run. Raw: [`concurrency-mc3c-2026-09-26/`](concurrency-mc3c-2026-09-26/).

| arm | clients | aggregate tok/s | p99 turn | notes |
|---|---:|---:|---:|---|
| old | 1 / 2 / 4 | 40.4 / 43.2 / 43.3 | 3.9 / 6.6 / 12.8 s | pre-MC3c |
| n1 | 1 / 2 / 4 | 43.7 / 43.9 / 43.9 | 3.2 / 6.4 / 12.9 s | the default path: unchanged |
| **n4** | 1 / 2 / 4 | 43.2 / **69.3** / **86.4** | 3.3 / 4.0 / **6.5 s** | 1.58× at 2 clients, 1.97× at 4 |
| n1 repeats | 4 | 42.6, 44.0 | 13.4, 12.8 s | idle-gated per cell, alternating |
| **n4 repeats** | 4 | **79.2, 84.3** | **7.0, 6.5 s** | 1.86× and 1.91× their paired n1 |

The first pair of repeats (n4 49.1, n1 31.0) is **void**. It ran while an external load, load average 8.6 with no
process of this session's, took the machine. Both arms fell, n1 by 30%, and they were re-run idle-gated per cell.

## Scope and what is next

- CPU models only. A GPU-resident, weight-streaming or vision model runs one generation at a time. A resident model
  gets its concurrency from MC1's KV slots (conversations stay warm) and, later, MC3.
- **Step 2 (batched decode behind the same admission)** is registered in the task doc and not started. It starts only
  if the `nobara` cells or larger client counts show batching beating the workers. On this Mac the workers already
  match it.
- **An owner decision is open: a default > 1 for CPU models.** Throughput doubles at 4 clients and p99 halves under
  load, but a lone request is unaffected only while it runs alone.
