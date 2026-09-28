# MC3 on CUDA — batched multi-request decode ships: 4 clients 1.38× (1.5B) / 1.83× (7B), every reply identical (2026-09-27)

This is the W7 grading of MC3 on CUDA (`7a44a58e`), against the design, correctness gates and decision table
pre-registered in [`task-concurrency-2026-09.md`](../tasks/task-concurrency-2026-09.md) ("MC3 on CUDA", `01bcb640`)
before any code was written. The S0 that motivated it is
[`concurrency-mc3-cuda-s0-2026-09-27.md`](concurrency-mc3-cuda-s0-2026-09-27.md).

**Verdict: all five gates pass and the aggregate clears 1.2×, so MC3 on CUDA ships on under serve's defaults.** The
defaults are `-max-concurrent` 4, `-kv-sessions` 4 and `-prefill-chunk` 512.

| gate | result | bar |
|---|---|---|
| 1, identity: `content_sha` of every turn, old = new | PASS (1, 2 and 4 clients, every cell) | equal |
| 2, reuse: prompt − reused of every turn, old = new | PASS | equal |
| 3, 4-client aggregate, new ÷ old, median of 3 pairs | **1.380×** (1.381 / 1.376 / 1.380; 218.0 → 300.6 tok/s) | ≥ 1.2× |
| 4, 4-client p99 turn, new ÷ old, median | **0.755×** (2.68 → 2.02 s) | ≤ 1.0 |
| 5, 1-client p50 / p99 turn, new ÷ old, median | **1.004× / 0.999×** | ≤ 1.05× each |

## Setup

- Machine: `nobara`, RTX 2070 SUPER 8 GB, NVIDIA driver 595.91.07.
- Builds: `serve-cuda` from clean trees, named by hash.
  - *old* = `01bcb640`: MC1, with KV slots but one generation at a time.
  - *new* = `7a44a58e`: MC3 on CUDA.
- Workload: `scripts/bench_w7_plain.py --engines goinfer --backend cuda --fixed-nonce --server-log`, 6 turns × 128
  greedy tokens per client, a fresh server per cell, at serve's defaults.
- Models: qwen2.5-coder-1.5b-instruct q4_k_m (gated), and qwen2.5-7b-instruct q4_k_m (reported), both from `~/models`.
- Cell order (4 and 1 clients): old new new old old new.
- Idle gate before every cell: load1 ≤ 2.0; no CUDA compute process beyond the one present at the start (kwin);
  GPU memory within 256 MiB of the 639 MiB baseline.
- Run 2026-09-27 18:59–19:04 PDT.
- Raw data, driver script, scorer and logs: [`concurrency-mc3-cuda-2026-09-27/`](concurrency-mc3-cuda-2026-09-27/)
  (`gates-output.txt` is the scorer's output, verbatim).

## Correctness, before any timing (all hard, all pass)

1. **Step identity** (`TestCUDAStepBatch_matchesForward`): at B = 2, 3 and 4 sequences, each on its own slot at a
   different depth, every `StepBatch` row equals that sequence's own `Forward` bit for bit. The last row of each step
   carries a `Draw` and equals `ForwardSample`'s id.
   - Fixtures: llama-tiny and mistral-tiny-window, the latter for the window.
   - The 1.5B and the 7B at depths {120, 2200, 700, 64}. The 2200-key sequence runs the flash-decode lane inside the
     step, which is not bit-identical to the exact path, and still matches its own decode exactly.
2. **Through the production path** (`TestCUDAStepBatch_concurrentMatchesAlone`): 4 concurrent conversations × 3 turns
   under `EnableResidentConcurrency(4)` emit exactly the ids each emits alone, greedy and sampled.
   - Conversation 0's prompt is long, so chunked prefill runs while the others decode.
   - A control on `ResidentBatchStats` requires batched steps to have run.
   - Tiny fixture, 1.5B and 7B.
3. **Decode unchanged:**
   - the tagged CUDA suite (`cuda goinfer_testhooks`) passes, 0 FAIL;
   - the heavy real-checkpoint decode parity (`TestRealForwardParity`) passes;
   - vet, gofmt and CI's pinned staticcheck are clean.

## W7 cells

**1.5B, gated** (from `gates-output.txt`):

| cell | old aggregate | new aggregate | ratio | p99 turn old → new | p50 turn old → new |
|---|---:|---:|---:|---|---|
| 4 clients, pair 1 | 217.8 | 300.7 | 1.381× | 2.68 → 2.02 s | 2.26 → 1.68 s |
| 4 clients, pair 2 | 218.0 | 299.9 | 1.376× | 2.68 → 2.01 s | 2.27 → 1.62 s |
| 4 clients, pair 3 | 218.2 | 301.1 | 1.380× | 2.67 → 2.02 s | 2.26 → 1.61 s |
| 1 client, pair 1 | 215.4 | 214.5 | — | 0.670 → 0.669 s | 0.566 → 0.570 s |
| 1 client, pair 2 | 214.3 | 214.3 | — | 0.671 → 0.674 s | 0.567 → 0.570 s |
| 1 client, pair 3 | 215.2 | 215.1 | — | 0.673 → 0.672 s | 0.566 → 0.567 s |

**Reported, not gated:**
- 1.5B, 2 clients, one pair: 217.6 → 236.2 tok/s = **1.085×**; p99 turn 1.33 → 1.21 s.
- 7B, 4 clients, one pair: 74.03 → 135.16 tok/s = **1.826×**.
  - p99 turn 7.22 → 4.14 s (0.573×); p50 6.84 → 3.75 s.
  - All 24 turns identical in content and prefilled tokens (`identity-7b.txt`).
  - The 7B starts at 4984 tokens with 4 slots (MC1 on CUDA).
- Under load, new's 4-client p99 is **3.00×** its lone request's p99 (1.5B).

**`ResidentBatchStats` per cell** (new's shutdown line, `w7-servers.log`; the old build has no batcher):

| cell | runs | batched steps (tokens) | solo tokens | steps by size 1 / 2 / 3 / 4 |
|---|---:|---:|---:|---|
| 1.5B 4 clients (3 cells) | 771–775 | 768 (3065–3069) | 3–7 | 0 / 0 / 3–7 / 761–765 |
| 1.5B 1 client (3 cells) | 768 | 0 | 768 | — |
| 1.5B 2 clients | 768 | 768 (1536) | 0 | 0 / 768 / 0 / 0 |
| 7B 4 clients | 771 | 768 (3069) | 3 | 0 / 0 / 3 / 765 |

At 4 clients, 99.8% of decode tokens ran in 4-row steps. No run waited out the straggler window. A lone request never
entered a step, which gate 5 confirms from the other side.

## Reading

- **The 7B lands where the S0 said it would; the 1.5B lands below.** The S0 put a 4-row step, with every row's
  logits, at 2.04× (7B) and 1.75× (1.5B) cheaper than 4 decodes. End to end they read 1.83× and 1.38×.
  - The 7B keeps ~90% of its step-level gain. The 1.5B keeps ~79%, and only 1.085× at 2 clients (step: 1.30×).
  - Metal's MC3 kept 1.59 of 1.73–1.84× on the same 1.5B workload.
  - The per-step fixed cost that does not shrink with the model weighs more on the 1.5B.
- **Candidate levers, estimated, not measured.** None is needed for the ship decision.
  - `prefillCore` allocates its scratch per call. A step calls it once per token, where a prefill calls it once per
    prompt.
  - The head runs per row: 4.2 of the 7B's 27.0 ms at B = 4 in the S0. A batched head is worth it only if it is
    proven bit-identical.
  - The 2-client cell is the most exposed to both.
- **Scope, as pre-registered.** Dense families whose batched prefill engages.
  - MoE (including C′), MLA, Gated DeltaNet and the other recurrent families, and a model without a batched prefill
    report `BatchStepRange` hi = 0 and keep one generation at a time.
  - Speculation, adapters and vision take the resident exclusively, as on Metal.
