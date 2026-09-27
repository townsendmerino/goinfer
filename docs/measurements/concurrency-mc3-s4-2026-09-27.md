# MC3 S4 — the 7B's batched-step cost: 2 clients 1.121× on the 7B and 1.059× on the 1.5B, every reply identical; encode-ahead parked (2026-09-27)

S4 of MC3 in [`task-concurrency-2026-09.md`](../tasks/task-concurrency-2026-09.md). On the 7B a batched step cost
~1.9× a token at any B ≤ 4, so 2 clients gained only 1.8%
([`concurrency-mc3-7b-w7-2026-09-27.md`](concurrency-mc3-7b-w7-2026-09-27.md)). This record finds why, ships the one
saving the diagnosis allows, and parks encode-ahead on its measured headroom.

**Result: all five pre-registered gates pass.**
- **7B, 2 clients:** aggregate **1.121×** the previous build (pairs 1.093–1.152×), p99 turn 0.74–0.90×.
- **1.5B, 2 clients:** **1.059×** (1.053–1.103×). It was registered only as a no-regression guard.
- **7B, 4 clients and a lone request:** unchanged (0.998× / p99 0.999×; 1.001× / 0.998×).
- Every reply and every turn's reuse is identical.

## 1. Why a 7B step costs ~1.9× a token (exploratory)

- **The step's matmul kernels have a fixed cost.** `mc3_bt` and `mc3_btd` run four 8×8×8
  `simdgroup_multiply_accumulate` per 32-k group over all 8 token columns, whatever B is.
  - On the 7B's large shapes, production's GEMV is bandwidth-bound: gate/up 0.433 ms, ~157 GB/s.
  - The fragment is ALU-bound at 1.17 ms. It costs 2.3× the GEMV on qkv, 1.9× on o, 2.7× on gate/up and 1.7× on down
    ([`concurrency-mc3-s0-2026-09-26.md`](concurrency-mc3-s0-2026-09-26.md)'s table).
  - Those four shapes are ~56 ms of the 64 ms one-sequence step.
- **No bit-identical kernel family does better at B ≥ 3.**
  - `TestMC3SmallMProbe`, re-run on this build ([`smallm-probe.log`](mc3-7b-lever-2026-09-27/smallm-probe.log)):
    S0's M-row GEMV costs more than M sequential GEMVs on the 7B. For gate/up at M = 2 it is 1.02–1.17 ms against
    2 × 0.433. At M = 4 it is 2.99 ms.
  - At B = 4 the fragment's half-empty columns cannot be refilled, because its columns are token vectors.
- **So the one saving is B = 2:** qkv and gate/up as two production GEMVs, which beat one fragment on both models.

## 2. Encode-ahead: parked, by its headroom (exploratory)

`TestMC3StepWallVsGPU` (`metal/mc3_step_test.go`), in sequence at depth 128, median of 20 steps:

| | B = 2 | B = 4 | B = 8 | production token: sync → pipelined |
|---|---:|---:|---:|---:|
| 1.5B, non-GPU ms per step | 1.26 (6.8%) | 1.39 (7.1%) | 1.66 (8.1%) | 0.83 → 0.45 |
| 7B, non-GPU ms per step | 1.41 (2.1%) | 1.59 (2.3%) | 2.18 (3.1%) | 0.92 → 0.50 |

- Production's own encode-ahead recovers about half of its token's non-GPU time.
- So a step executor would buy ~4% on the 1.5B and ~1.3% on the 7B. It would also need a guess at the next run's
  makeup, because the batcher regroups sequences between steps. That is at or under the park line.
- What the served gap mostly is, estimated from W7's own numbers and not measured: newcomer suffix prefills, which run
  in exclusive sections and take about a third of a 4-client 1.5B cell. That is the per-pass prefill cost lead.
- Logs: [`wall-vs-gpu-1.5b.log`](mc3-7b-lever-2026-09-27/wall-vs-gpu-1.5b.log),
  [`wall-vs-gpu-7b.log`](mc3-7b-lever-2026-09-27/wall-vs-gpu-7b.log).

## 3. The build (`7fa344b2`)

- In `forwardMulti`, qkv and gate|up run as B per-row production GEMVs below a calibrated batch size, and as the
  fragment above it.
  - The per-row path is production's own kernel and dispatch (`saRowsPick`), fed each row's int8 activations and
    scale and writing that row's output. So it is production's arithmetic by definition.
- `calibrateRows` times both arms once at build, on the first layers' real weights, one layer per dispatch so the cache
  does not serve them.
  - Arms are interleaved rep by rep; each keeps its fastest command buffer after 3 warm-ups.
  - A first version (median, arms in sequence) flipped between loads: once B ≤ 3 on the 7B's qkv, once 0 on the
    1.5B's. The interleaved minimum chose B ≤ 2 for both shapes on both models in 6 of 6 loads, and in all 12 W7
    servers.
  - The choice is logged at load (`metal: batched step: qkv as per-row GEMVs at B <= 2 …`).

## 4. Correctness, before any timing

- `TestMC3Step_rowsPathBitIdentical` forces the per-row path at every batch size, then the fragment at every size, at
  B = 2, 3 and 4: **0 logits differ** from production's single-token forward, on the 1.5B and the 7B.
- In the calibrated configuration, `TestMC3Step_bitIdentical`, `TestMC3Step_drawsMatchForwardSample` and both
  concurrent-vs-alone tests pass on both models.
- The tagged Metal suite: 183 pass, 0 fail.
- Logs: [`s4-identity-1.5b.log`](mc3-7b-lever-2026-09-27/s4-identity-1.5b.log),
  [`s4-identity-7b.log`](mc3-7b-lever-2026-09-27/s4-identity-7b.log),
  [`metal-suite-tagged-s4.log`](mc3-7b-lever-2026-09-27/metal-suite-tagged-s4.log).

## 5. In sequence (reported)

`TestMC3Step_throughput`, the same test as the pre-S4 7B run in
[`concurrency-mc3-7b-w7-2026-09-27.md`](concurrency-mc3-7b-w7-2026-09-27.md) §4:

| | B = 2 step, before → after | B = 2 aggregate | B = 4 step |
|---|---:|---:|---:|
| 7B, depth 128 | 65.29 → **58.76 ms** | 1.016× → **1.135×** | 66.28 → 66.33 ms |
| 7B, depth 512 | 68.48 → **62.07 ms** | 1.029× → **1.131×** | 72.53 → 72.86 ms |
| 1.5B, depth 128 | 17.20 → 16.87 ms | | |

Projected before the build: −8.8 ms on the 7B's B = 2 step. Measured: −6.5 ms.
Logs: [`s4-throughput-7b.log`](mc3-7b-lever-2026-09-27/s4-throughput-7b.log),
[`s4-throughput-1.5b.log`](mc3-7b-lever-2026-09-27/s4-throughput-1.5b.log).

## 6. The W7 grading

As registered (`915bf6fc`, committed before the build):
- M1 Pro 16 GB, macOS 26.6.2.
- *old* = `serve-metal` at `915bf6fc` (pre-S4); *new* = at `7fa344b2`.
- Serve defaults, `scripts/bench_w7_plain.py`, 6 turns × 128 greedy tokens, `--fixed-nonce`, from `~/models`.
- 7B at 2, then 4, then 1 client(s); then the 1.5B at 2 clients. Each is 3 interleaved pairs (old new new old old new).
- Idle-gated per cell (load1 ≤ 2.0). The gate held the start ~8 minutes, during interactive use.
- 2026-09-27 12:51–13:15 PDT.
- Swap stood at 1.87 GB throughout, left over from the morning, and no server's swap guard tripped.

Raw: [`w7-s4-7b.json`](mc3-7b-lever-2026-09-27/w7-s4-7b.json), [`w7-s4-15b.json`](mc3-7b-lever-2026-09-27/w7-s4-15b.json),
[`w7-s4-run.log`](mc3-7b-lever-2026-09-27/w7-s4-run.log), [`w7-s4-servers.log`](mc3-7b-lever-2026-09-27/w7-s4-servers.log),
[`run-w7-s4.sh`](mc3-7b-lever-2026-09-27/run-w7-s4.sh). Gates by [`gates-s4.py`](mc3-7b-lever-2026-09-27/gates-s4.py), output in
[`gates-s4-output.txt`](mc3-7b-lever-2026-09-27/gates-s4-output.txt).

| gate (pre-registered) | result |
|---|---|
| 1. identity and reuse, new vs old, every turn, 24 cells (hard) | **pass** |
| 2. 7B 2-client aggregate new ÷ old, median ≥ 1.05× | **pass: 1.121×** (1.152, 1.093, 1.121); p99 turn 0.736–0.895× |
| 3. 7B 4-client aggregate ≥ 0.98× and p99 turn ≤ 1.02× (hard) | **pass: 0.998× / 0.999×** |
| 4. 7B lone request p50 / p99 ≤ 1.05× (hard) | **pass: 1.001× / 0.998×** |
| 5. 1.5B 2-client aggregate ≥ 0.98× (hard) | **pass: 1.059×** (1.103, 1.053, 1.059) |

**Decision, by the registered rule:** all pass, so S4 ships. It is on by default, calibrated per model at load.

**Reading.**
- The 7B's 2-client gain end to end (1.12×) is close to the in-sequence 1.13×. The prefills between steps dilute it
  little at 2 clients.
- The 1.5B's 5.9% came out larger than the ~3% projected from per-kernel costs. Two effects could explain it: the
  per-row path also skips the qkv and gate/up packing dispatches, and the served mix runs many B = 2 steps. Neither
  was separated.
- On the 7B, 2 clients now read 28.5–29.5 tok/s against a lone request's ~26.1: 1.09–1.13× running one at a time.
  Before S4 it was 1.02×.
- The 4-client figure is untouched, because B = 4 keeps the fragment. On this GPU no bit-identical kernel family found
  beats it there.
