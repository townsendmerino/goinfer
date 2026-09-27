# MC3 on the 7B, end to end: 4 clients at 1.785× the serialized build, p99 turn 0.592×, a lone request unchanged, every reply identical; 2 clients +1.8% (2026-09-27)

The open "7B end to end" item of MC3 in [`task-concurrency-2026-09.md`](../tasks/task-concurrency-2026-09.md). Every
W7 grading of MC3 so far ran the 1.5B:
- MC3 itself: [`concurrency-mc3-2026-09-26.md`](concurrency-mc3-2026-09-26.md);
- S2: [`concurrency-mc3-s2-2026-09-27.md`](concurrency-mc3-s2-2026-09-27.md);
- S3: [`concurrency-mc3-s3-2026-09-27.md`](concurrency-mc3-s3-2026-09-27.md);
- chunked prefill: [`chunked-prefill-2026-09-27.md`](chunked-prefill-2026-09-27.md).

On the 7B, MC3 had been measured only below serve: S0's kernels, and S1's step in sequence
([`concurrency-mc3-s1-2026-09-26.md`](concurrency-mc3-s1-2026-09-26.md)). This grading runs the whole current stack
against the pre-MC3 serialized build.

**Result: all five pre-registered gates pass.**
- 4 greedy clients read **45.5 tok/s against 25.4–25.6** (1.785×, pairs 1.778–1.790×).
- p99 per turn falls from 20.85–20.99 s to 12.35–12.37 s (0.592×).
- A lone request: p50 1.002×, p99 1.014×.
- Every turn's reply and prefix reuse are identical, in all 18 cells.

**2 clients gain 1.8%**, where the 1.5B gained 8–11%. The in-sequence step (§4) shows why: on the 7B a step of two
sequences costs 1.95–1.97× one token.

## 1. Correctness, run before any timing

Run on the 7B's `.int4.metal.giw` with weights aliased, as S1 ran it. The `.gguf`'s own int4 load path needs
13.4 GB and does not fit this Mac. 2026-09-27 08:17–08:25 PDT.

| test | result |
|---|---|
| `TestMC3Step_bitIdentical`: 4 sequences × 12 steps, depths 5 / 23 / 40 / 300 | **0** logits differ from production's single-token forward |
| `TestMC3Step_bitIdenticalDeep`: depths 1936 / 1530 / 100 / 1600, mixing attention_fa and per-head rows in one step | **0** differ |
| `TestMC3Step_drawsMatchForwardSample`: 24 device draws, 24 logits rows | **0** differ |
| `TestMC3_concurrentMatchesAloneOnMetal`: 4 conversations × 3 turns through `Model.Generate`, greedy | ids and reuse identical; 96 steps served 379 tokens |
| `TestMC3_concurrentSampledMatchesAloneOnMetal`: the same at T = 0.8, fixed seeds | ids identical; 38 steps served 129 tokens |

Log: [`mc3-7b-identity.log`](concurrency-mc3-7b-w7-2026-09-27/mc3-7b-identity.log).

The first launch (08:15) pointed the tests at the `.gguf`. The fit guard refused every load, and nothing was
bypassed ([`mc3-7b-identity-attempt1-fitguard.log`](concurrency-mc3-7b-w7-2026-09-27/mc3-7b-identity-attempt1-fitguard.log)).

That launch also showed the sampled concurrent test passing *on the 1.5B*. It had hard-coded the 1.5B and ignored
`GOINFER_METAL_MC3_MODEL`. Both concurrent tests now read it, plus `GOINFER_METAL_MC3_TOKENIZER`, since a `.giw`
bundle is named with its `.gguf` for the tokenizer (`21e88ae4`).

## 2. The W7 grading

As registered (`21e88ae4`, committed and pushed before any timing):
- M1 Pro 16 GB, macOS 26.6.2.
- *old* = `serve-metal` at `9efc3185`: MC1 plus R18b plus the slot-pricing fix. A Metal resident serialises there.
- *new* = at `cc5f8c2c`: MC3, S2, S3 and chunked prefill. `21e88ae4`, the tree at the run, adds only docs and a test
  file to it.
- Both ran at serve's defaults (`-max-concurrent` 4, `-kv-sessions` 4, context 4096).
- All 18 servers logged the expected concurrency line: old's 9 *"one generation at a time"*, new's 9 *"4 generations
  at once … decode tokens batched"*. Each kept 4 conversations resident, and none logged an error or a clamp.
- `scripts/bench_w7_plain.py` with `BENCH_W7_MODEL` = qwen2.5-7b-instruct q4_k_m from `~/models`. Serve loads its
  `.int4.metal.giw`, weights aliased (4020 MB bound in place).
- 6 turns × 128 greedy tokens per client, `--fixed-nonce`, a fresh server per cell.
- 4, then 1, then 2 clients, each old/new × 3 pairs in the order old new new old old new.
- Idle-gated per cell (load1 ≤ 2.0; readings 0.50–1.89). Only the first cell waited, 30 s.
- 2026-09-27 08:26–08:45 PDT (15:26–15:45 UTC).
- The desktop was in interactive use (a browser, the editor). The cells repeat within ±0.3% at 4 clients.
- The bench header reads `goinfer_dirty: true`. The dirty entries were untracked docs: another session's prompt file,
  and this record's run script.

Raw: [`w7-7b.json`](concurrency-mc3-7b-w7-2026-09-27/w7-7b.json), [`w7-7b-run.log`](concurrency-mc3-7b-w7-2026-09-27/w7-7b-run.log),
[`w7-7b-servers.log`](concurrency-mc3-7b-w7-2026-09-27/w7-7b-servers.log), [`run-w7.sh`](concurrency-mc3-7b-w7-2026-09-27/run-w7.sh).
The gates are computed by MC3's own [`gates.py`](concurrency-mc3-2026-09-26/gates.py), output in
[`gates-output.txt`](concurrency-mc3-7b-w7-2026-09-27/gates-output.txt).

| gate (pre-registered) | result |
|---|---|
| 1. identity: every turn's `content_sha`, new vs old, every cell (hard) | **pass** (18 cells) |
| 2. reuse equal (hard) | **pass** |
| 3. 4-client aggregate new ÷ old, median of 3 ≥ 1.2× | **pass: 1.785×** (1.785, 1.790, 1.778) |
| 4. 4-client p99 turn new ÷ old, median ≤ 1.0 (hard) | **pass: 0.592×** (0.589, 0.592, 0.592) |
| 5. lone request p50 / p99, new ÷ old, median ≤ 1.05× each (hard) | **pass: 1.002× / 1.014×** |
| reported: 2 clients | aggregate 1.018 / 1.018 / 1.017×; p99 10.45–10.46 → 10.35–10.40 s |
| reported: new's 4-client p99 ÷ its lone-request p99 | 2.30× |

**Decision, by the registered rule:** all pass, so MC3 is recorded as paying on the 7B. The open "7B end to end"
item closes.

## 3. Per turn

| clients | build | aggregate tok/s | p50 turn | p99 turn |
|---:|---|---:|---:|---:|
| 4 | old | 25.44–25.59 | 19.71–20.04 s | 20.85–20.99 s |
| 4 | new | 45.51–45.55 | 10.99–11.01 s | 12.35–12.37 s |
| 2 | old | 25.56–25.59 | | 10.45–10.46 s |
| 2 | new | 26.02–26.05 | | 10.35–10.40 s |
| 1 | old | 25.49–25.53 | 4.947–4.958 s | 5.284–5.306 s |
| 1 | new | 25.16–25.50 | 4.956–4.962 s | 5.302–5.486 s |

- At 4 clients a turn takes 2.30× a lone request's, against 1.92× on the 1.5B after S3. Serialized, it took 3.9×.
- The lone request's aggregate reads up to 1.4% lower on new (25.16 in one cell). Its p50 turn ratios are
  1.001–1.002×. The p99 ratios are 0.999–1.038×, inside the guard. A lone request's tokens run production's own
  per-sequence path.

## 4. The step on the 7B (exploratory, after the grading)

`TestMC3Step_throughput` on the same `.giw`, in sequence (no serve, no prefill between steps): median GPU step time,
7 reps, arms interleaved. 08:46–08:50 PDT. The before column is S1's 7B run, before S2 and S3 changed the step
([`identity-throughput-7b.log`](concurrency-mc3-s1-2026-09-26/identity-throughput-7b.log)).

| depth | B | step before (S1) | **step now** | step ÷ one token now | aggregate vs one stream, now |
|---:|---:|---:|---:|---:|---:|
| 128 | 1 | 64.89 ms | 64.37 ms | 1.94× | 0.515× |
| 128 | 2 | 68.03 | 65.29 | 1.97× | **1.016×** |
| 128 | 4 | 73.03 | 66.28 | 2.00× | 1.998× |
| 128 | 8 | 83.62 | 68.51 | 2.07× | 3.865× |
| 512 | 2 | 73.08 | 68.48 | 1.95× | 1.029× |
| 512 | 4 | 81.31 | 72.53 | 2.06× | 1.942× |
| 512 | 8 | 99.92 | 80.57 | 2.29× | 3.495× |

Production's single-token decode: 33.1 ms at depth 128, 35.2 ms at depth 512.

Log: [`mc3-7b-step-throughput.log`](concurrency-mc3-7b-w7-2026-09-27/mc3-7b-step-throughput.log).

**Reading.**
- **On the 7B, a step has a floor of ~1.9× a token.** A step of one sequence already costs 1.94× production's token.
  Each added sequence costs ~0.5–0.9 ms at depth 128, and ~2 ms at depth 512, where attention grows.
  - The floor is most likely the batched matmuls. On the 1.5B, S3's breakdown put them at 14.4 of a 22.6 ms step. The
    7B's step was not broken down.
- **So B = 2 is break-even on this size, and 2 clients read 1.018×.** Nothing in the scheduler loses it. Two clients
  gain only if the batched matmuls, at small M, cost less than two production GEMVs on the 7B's shapes. S0 measured
  the fragment kernels at 1.0–2.7 GEMVs across the 1.5B's and 7B's shapes.
- **4 clients reach 1.785× end to end, against 1.94–2.00× in sequence.** The difference is what serve adds between
  steps: each newcomer turn's prefill of a ~150-token suffix runs in an exclusive section, and tokens wait in the
  straggler window. It was not separated further.
- **The lever for this size is the batched matmul's own cost, not per-row work.** A hybrid B = 2 (production's path
  for two sequences) would read ~1.0× here, since two production tokens are 2.0× a token as well.
