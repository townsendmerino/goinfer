# MC3 S2 — sampled tokens join batched steps: 4 sampled clients at 1.62× the shipped MC3, which gave them nothing (2026-09-27)

S2 of MC3 in [`task-concurrency-2026-09.md`](../tasks/task-concurrency-2026-09.md), an owner decision of
2026-09-27. MC3 ([`concurrency-mc3-2026-09-26.md`](concurrency-mc3-2026-09-26.md)) batched greedy tokens only:
- A temperature-only token is drawn on the device (`ResidentSample.ForwardSample`), and under MC3 each such token ran
  as its own per-sequence call.
- So a plain `temperature > 0` chat request, the default shape of most chat clients, was never batched.
- S2 lets a step row carry that draw.

**Result: all five pre-registered gates pass.**
- 4 sampled clients (temperature 0.8) read **119.7–120.1 tok/s against the shipped MC3's 72.6–74.2**: 1.623×.
  p99 per turn falls from 7.25–7.70 s to 4.62–4.63 s (0.635×).
- A lone request is unchanged (1.002× p50, 1.005× p99).
- Every sampled reply is identical across builds, including the pre-MC3 reference.
- The reference cell, 74.1 tok/s, confirms that the shipped MC3 had given sampled requests nothing.

## The change (`4954f978`)

- `ResidentBatchSeq.Draw` carries (temperature, seed, draw). `StepBatch` returns a `ResidentBatchOut` per row: the
  drawn id for a drawing row, else its logits.
- The Metal step runs ForwardSample's two gumbel dispatches on each drawing row of the batch logits, with that row's
  uniforms. It also takes ForwardSample's two fallbacks: an infinite 1/T and an all-`-inf` row both fall back to the
  argmax. A drawing row copies no logits.
- `generateInto` takes the sampler's draw once per token, before submitting, in both modes, so the RNG stream
  advances exactly as it did before S2. A step-served drawn token continues on `fastNext`, as a ForwardSample token
  does.

## Correctness, run before any timing

- **`TestMC3Step_drawsMatchForwardSample`:** 4 rows (two drawing at T = 0.8 and 1.3, two returning logits) × 12
  teacher-forced steps, against ForwardSample / ForwardEmb on twin slots. 24 draws and 24 logits rows checked;
  **0 differ**.
- **`TestMC3_concurrentSampledMatchesAloneOnMetal`:** 4 conversations at T = 0.8 with fixed seeds, through
  `Model.Generate` on the 1.5B. Ids are identical concurrent and alone, and 163 tokens were served in 48 steps.
- **`TestMC3_sampledConcurrentMatchesAlone`**, on the fake resident: red when the batched draw uses the wrong counter.
- The tagged Metal suite ran 182 pass, 0 fail; the decoder suite 650 pass, bar the manifest, refreshed
  goldens-proven (38 passed, 0 failed); serveapp ok.

## The W7 grading

Pre-registered in `16c27bd7`, committed before the S2 build:
- *old* = `serve-metal` at `d2225ec4`, the shipped MC3.
- *new* = `serve-metal` at `4954f978`.
- *ref*, reported only = `9efc3185`, pre-MC3.
- The MC3 W7 workload at `--temperature 0.8 --seed 1000 --fixed-nonce`: a fixed seed per client per turn, since
  serve draws a random one otherwise (M-03).
- M1 Pro, qwen2.5-coder-1.5b int4 from `~/models`, serve defaults, a fresh server per cell, idle-gated;
  2026-09-27 03:55–04:01 PDT.
- The server logs show the MC3 banner in all 12 old and new cells, and the pre-MC3 build in the ref cell.

Raw: [`w7-s2.json`](concurrency-mc3-s2-2026-09-27/w7-s2.json), [`w7-s2-run.log`](concurrency-mc3-s2-2026-09-27/w7-s2-run.log),
[`w7-s2-servers.log`](concurrency-mc3-s2-2026-09-27/w7-s2-servers.log), [`run-w7.sh`](concurrency-mc3-s2-2026-09-27/run-w7.sh);
gates by MC3's [`gates.py`](concurrency-mc3-2026-09-26/gates.py) → [`gates-output.txt`](concurrency-mc3-s2-2026-09-27/gates-output.txt).

| gate (pre-registered) | result |
|---|---|
| 1. identity: every turn's reply equal, new vs old, every cell | **pass**: 13 cells, the pre-MC3 ref included |
| 2. reuse equal | **pass** |
| 3. 4-client aggregate, new ÷ old, median of 3 ≥ 1.2× | **pass: 1.623×** (1.614, 1.623, 1.649) |
| 4. 4-client p99 turn, new ÷ old, median ≤ 1.0 | **pass: 0.635×** (0.602, 0.635, 0.637) |
| 5. lone request p50 and p99, new ÷ old, median ≤ 1.05× | **pass: 1.002× / 1.005×** |

| cell | ref (pre-MC3) | old (shipped MC3) | new (S2) |
|---|---:|---:|---:|
| 4 clients, aggregate | 74.1 tok/s | 72.6 / 74.2 / 74.0 | **119.7 / 119.7 / 120.1** |
| 4 clients, p99 turn | 7.36 s | 7.70 / 7.25 / 7.27 s | **4.63 / 4.62 / 4.62 s** |
| 1 client, p50 turn | — | 1.733 / 1.700 / 1.712 s | 1.708 / 1.704 / 1.721 s |

**Reading.** Sampled traffic now batches as well as greedy: 1.62× against greedy's 1.59× in MC3's own grading. The
two runs are a session apart, so the ratios are not directly comparable.

Before S2, the shipped MC3 gave sampled requests nothing: 72.6–74.2 tok/s against the pre-MC3 build's 74.1. A
temperature-only token's device draw ran as its own per-sequence call, one after another, so concurrency bought
nothing. By the code path each such call also rebinds the resident's KV slot, draining its pipelined executor, when
the conversations alternate. That cost was not measured separately.
