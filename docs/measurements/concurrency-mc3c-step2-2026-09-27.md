# MC3c step 2 — batched CPU decode behind serve's admission: 4 clients 2.19× on a 7B, a lone request unchanged (2026-09-27)

MC3c step 2 of [`task-concurrency-2026-09.md`](../tasks/task-concurrency-2026-09.md): concurrent CPU generations of one
model join their decode tokens into one batched forward, instead of step 1's independent workers. The trigger was the
MC2 7B cell (batched B = 4 beat J8's N = 4 by 2.25–2.41×). The design and grading were pre-registered in `0bc06f90`
before any code; the code is `420d655b`, and the graded build is `2c1d89ec`, which adds the shutdown stats line.

**Result: all five hard gates pass, and it ships with `-cpu-batch auto` as the default.**
- 4 clients: **2.19×** the step-1 build's aggregate (5.44 → 11.92 tok/s; pairs 2.200 / 2.185 / 2.187).
- The p99 turn at 4 clients drops **0.48×**, from ~96 s to ~46 s.
- A lone request is unchanged: p50 **1.000×**, p99 **1.000×**.
- Every reply is identical to the step-1 build's, turn for turn, at every client count.
- 2 clients (reported): 1.47×.
- On the 0.5B, forcing `-cpu-batch on` is 0.73× the workers. That is why `auto` leaves a model that small on them.

## Setup

- Machine: `nobara`, Ryzen 7 3700X (16 threads), kernel 7.2.0-202.nobara.fc44, CPU governor `schedutil`, CPU backend
  (`-backend cpu`, no GPU use).
- Model: qwen2.5-7b-instruct q4_k_m from `~/models` at int4, with serve defaults (`-max-concurrent` 4, `-kv-sessions`
  4). On the new build `-cpu-batch auto` batches the 7B, and the banner says so.
- Builds, each built once from a clean worktree (`vcs.modified=false`):
  - **old** = `serve` @ `0bc06f90`, step 1 (independent workers);
  - **new** = `serve` @ `2c1d89ec`, step 2.
- Workload: `scripts/bench_w7_plain.py --backend cpu --fixed-nonce`, 6 turns × 128 greedy tokens per client, and a
  fresh server per cell.
- Idle gate before every cell: load1 ≤ 1.0 and no other go, test-binary or serve process.
- Cells, in the registered order:
  - 4 clients old/new × 3 (old new new old old new);
  - 1 client the same;
  - one 2-client pair;
  - the 0.5B at 4 clients with the new build, `-cpu-batch off` then `on`.
- One session, 2026-09-27 11:55–13:42 PDT.
- Raw: [`concurrency-mc3c-step2-2026-09-27/`](concurrency-mc3c-step2-2026-09-27/):
  - `w7.json`, `w7-05b.json`;
  - `w7-run.log`, and `w7-servers.log` (every banner, and each new server's CPU-batch stats line);
  - `gates.py`, `gates-output.txt`;
  - `run-w7.sh`;
  - the correctness logs.

## Results (qwen2.5-7b-instruct)

| build | clients | aggregate tok/s (cells) | prompt − reused, turns 2–6 (every client) | slowest turn | batched steps per cell (stats) |
|---|---:|---|---|---:|---|
| old | 1 | 4.76, 4.76, 4.76 | 26, 27, 24, 20, 27 | 27.9–28.0 s | — |
| old | 2 | 5.28 | 26, 27, 24, 20, 27 | 49.8 s | — |
| old | 4 | 5.41, 5.46, 5.46 | 26, 27, 24, 20, 27 | 95.6–96.2 s | — |
| **new** | 1 | 4.75, 4.75, 4.75 | 26, 27, 24, 20, 27 | 28.0 s | 0 (768 solo tokens) |
| **new** | 2 | **7.74** | 26, 27, 24, 20, 27 | 35.1 s | 756, all 2 wide |
| **new** | 4 | **11.90, 11.93, 11.94** | 26, 27, 24, 20, 27 | 46.0–46.2 s | 768 (737–744 of 4, the rest 3) |

The server-side stats line (at shutdown, `2c1d89ec`) confirms the mechanism:
- At 4 clients almost every step carried all 4 sequences. The solo tokens (24–31 per cell) are, most likely, turn
  boundaries, where one generation is prefilling or finishing and fewer than two are decoding.
- A lone request ran no batched step at all, so it took production's own forward, which is why its timings match the
  old build's to the millisecond.
- No run was straggler-started.

## Gates (pre-registered in `0bc06f90`; `gates-output.txt`)

| gate | result |
|---|---|
| correctness before timing (hard) | **pass.** The production step is bit-identical to `m.forward` on every committed fixture it admits (30 fixture/quant cells). Concurrent `Session.Generate` matches alone, greedy and sampled, on llama-tiny, the 1.5B (`on`) and the 7B (`auto`, which engages by itself). int8-KV caches bypass the batcher, and `-race` is clean. |
| 1 identity (hard) | **pass:** every turn's `content_sha` is equal between old and new, at 1 and at 4 clients, in every cell |
| 2 reuse (hard) | **pass:** every turn prefills the same tokens in both builds |
| 3 aggregate, 4 clients, new ÷ old, median of 3 ≥ 1.2× | **2.187×** (2.200, 2.185, 2.187) |
| 4 p99 turn under load, new ÷ old, median ≤ 1.0 (hard) | **0.481×** (0.480, 0.483, 0.481) |
| 5 solo guard: 1-client p50 and p99, new ÷ old, median ≤ 1.05× (hard) | **p50 1.000×, p99 1.000×** |

Decision, by the registered rule: all hard gates pass and the aggregate is ≥ 1.2×, so it **ships with `auto` as the
default**. It is already the build's default; nothing else changes.

Reported, not gated:
- **2 clients:** 1.466× (5.28 → 7.74 tok/s), p99 turn 49.8 → 35.1 s.
- **Under load vs alone:** new's 4-client p99 is 1.65× its lone request's.
- **The 0.5B, why `auto` stops at 2 GiB:** `-cpu-batch on` reads 48.39 tok/s against `off`'s 66.31 (0.73×), with
  every reply identical (0 of 24 turns differ). On a small model the workers keep the cores busier than one batched
  forward does. This matches MC2's 0.64–0.75× on the 0.5B.

## Against the trigger

MC2's kernel-level cell measured batched B = 4 at 2.59–2.73× serial on the 7B, with the workers at 1.13–1.15×. End to
end, serve's workers reach 5.44 tok/s at 4 clients, 1.14× a lone request's 4.76. That is the same saturation. Batching
reaches 11.92, 2.50× a lone request. W7's plain turns sit mostly at depths 150–900, between MC2's two cells. The
end-to-end gain of 2.19× over the workers is the trigger's 2.25–2.41× less the per-turn costs batching does not touch:
prefill of the new user turn, sampling and HTTP.

## Scope, stated

- **CPU backend only.** A GPU-resident model's concurrency is MC3's (Metal) or none (CUDA).
- **Eligible families** are the plain gated dense ones (`cpuBatchModelEligible`). MoE, recurrent, hybrid, own-forward,
  non-gated and sandwich-norm families run as step 1's workers.
- **Eligible caches** are f32 append-forever ones. int8 KV, a sliding-window ring, an adapter session or a tree mask
  bypass the batcher.
  - The design registered "a run mixing eligible and ineligible tokens batches the eligible ones". In the build, an
    ineligible cache never enters the batcher at all, so no run mixes them.
- **`auto` is off on darwin** until a Mac 7B cell shows batched B = 4 ÷ J8 N = 4 ≥ 1.15×.
- **The batched numbers are a floor.** The batched path's M = 1 costs 0.965× production on the 7B; a tuned small-M
  amd64 kernel is a follow-on.
