# Vetted-checkpoint peer cells, nobara-pc (pre-registered 2026-10-07)

**Status: PRE-REGISTERED 2026-10-07, before any timed run.** Night queue job `peer-vetted-nobara`. Results and verdicts go
below *Results*. A bar that turns out wrong gets a dated amendment with its mechanism, not an edit.

**Question.** goinfer.dev/models has empty speed cells for three vetted checkpoints. This night fills nobara's: gpt-oss 20B
on CUDA, and Phi-3 mini, gpt-oss 20B and Gemma 4 26B-A4B on the CPU. Granite 4.0-H Tiny is out of scope: it runs on the CPU
on both GPU machines, and no CPU number is compared with a GPU number.

## Cells

| cell | checkpoint (docs/capability-matrix.json) | backend | goinfer | Ollama v0.32.5 |
|---|---|---|---|---|
| 1a | Phi-3 mini 4k, `Phi-3-mini-4k-instruct-q4.gguf` (sha256 `8a83c7fb…`) | CPU | `--backend cpu`, its default quant (q4k) | `p3m-local`, CPU-forced (`num_gpu: 0`) |
| 1b | gpt-oss 20B, `gpt-oss-20b-MXFP4.gguf` (sha256 `27cd6c43…`) | CPU | `--backend cpu`, default (int4mix) | `g20`, CPU-forced |
| 1c | Gemma 4 26B-A4B, the QAT `gemma-4-26B_q4_0-it.gguf` (sha256 `3eca3b8f…`) | CPU | `--backend cpu`, default (int4; the MoE loads from its own one-time sidecar, the default CPU path) | `g4-26b`, CPU-forced |
| 2 | gpt-oss 20B, as 1b | CUDA (RTX 2070 SUPER, 8 GB) | `--backend cuda -moe-cache-experts`, default quant (int4mix) | `g20`, its own partial offload |

- **Every cell:** depth 128, greedy, 3 runs (pairs), 8 completions × 64 tokens a run, goinfer against Ollama.
- **Context:** cell 2 runs at context 2048 (`BENCH_CTX=2048` on both engines), as cell c of
  [`peer-sweep-2026-09-29.md`](peer-sweep-2026-09-29.md) does for the 26B on the same card. The CPU cells run at the
  harness's default context.
- **Phi-3 and the 26B on CUDA already have cells** (cell b, cell c of 2026-09-29) and are not re-run here.
- **The 26B is the vetted QAT file on both sides,** under its own harness key `G26Q`. It is not cell c's requantized
  Q4_K_M. Cell c's history is left alone.

## Rules, fixed before the run

- **The same weights on both sides.** goinfer loads the vetted file. Each Ollama tag was checked on 2026-10-07 with
  `scripts/gguf_same_weights.py`:
  - `g20` and `g4-26b`: SAME WEIGHTS. Their model blobs are the vetted files byte for byte (blob digest = the file's
    sha256).
  - `p3m-local`: 195/195 tensors identical (2026-09-25).
- **The same engine path on both sides:** CPU against CPU-forced Ollama, GPU against GPU. No cell compares a CPU number
  with a GPU number.
- **GPU residency.** Cell 2 is VOID unless goinfer's decode path is the GPU's (`cuda-resident`), read from its serve log.
  - **The by-day smoke, 2026-10-07, untimed:** `cuda-resident (int4mix)`, the C′ expert cache capped at 17 slots per
    layer (5.8 GB) by free VRAM, and a correct reply.
  - The CPU smokes read `cpu (q4k)`, `cpu (int4mix)` and `cpu (int4)`.
- **Each engine at its defaults,** as in cell c. On the 8 GB card that means goinfer with `-moe-cache-experts` and
  Ollama with its own partial offload, which is stated on the cell.
- **Swap.** One engine is loaded at a time (the harness restarts servers per cell). `BENCH_SWAP_VOID_MB=0`: any growth
  in system swap over a cell, from before its server starts to its last completion, voids that engine's arm.
- **Bands and voids are the 2026-09-29 sweep's, unchanged** (its `grade.py`):
  - r = goinfer ÷ Ollama per pair;
  - AHEAD (all pairs > 1.03), LEVEL (all in 0.97–1.03), BEHIND (all < 0.97), AMBIGUOUS-HIGH or -LOW otherwise, every
    pair deciding;
  - the 5% spread cap;
  - the harness's proportional token gate;
  - the decode-path gate.
- **The two registered additions** (`peer-vetted-2026-10-07/grade.py`):
  - an arm the harness refused, including a swap void, is VOID with the harness's reason;
  - a valid goinfer arm beside a VOID Ollama arm is reported GOINFER-ALONE (speed, no ratio). This happens on this box
    only by accident: no Ollama arm here is expected to swap.
- **Provenance:**
  - goinfer `65b2c22a`, serve binaries built from a clean worktree at it;
  - Ollama v0.32.5 at `~/ollama-0325`;
  - NVIDIA driver 595.91.07 (another driver stops the run);
  - checkpoints from `~/models`.

**Decision rule.** Nothing is decided by this run.
- **A valid cell** becomes a `claims.json` entry: `runs` / `peer_runs` exactly as printed, the medians, `ratio_raw` this
  record's median r, the date and this file as the source.
- **A VOID cell** stays "not measured", with its reason here.
- **A loss stays a loss.**

**Cost.** By night only, under the queue's timing lock: four cells, two engines, about 30 minutes of cells and loads.
Queued at 50 minutes.

## Results

### Night 1, 2026-10-07 23:31-23:58 PDT (graded 2026-10-08)

- **The run:** night job `peer-vetted-nobara` at the pinned `65b2c22a`.
  - Serve sha256: `11fdea29…` (cuda) and `524b8866…` (cpu).
  - NVIDIA driver 595.91.07, Ollama v0.32.5.
  - Load 0.89 at the start.
- **Raw:** `~/goinfer-bench/peer-vetted-2026-10-07/results/` on nobara-pc (`cpu.json`, `cuda-g20.json`, `grade.jsonl`,
  serve logs).
- **Graded** with `peer-vetted-2026-10-07/grade.py`, the rules above unchanged.

| cell | goinfer runs (tok/s) | median | Ollama runs | median | pairs r | median r | outcome |
|---|---|---:|---|---:|---|---:|---|
| 1a cpu phi3-mini @128 greedy | 8.2 / 8.2 / 8.2 | 8.2 | 10.8 / 10.7 / 10.8 | 10.8 | 0.763 / 0.763 / 0.762 | 0.763 | BEHIND |
| 1b cpu G20 (gpt-oss 20B) @128 greedy | 7.5 / 7.4 / 7.5 | 7.5 | 10.7 / 10.7 / 10.7 | 10.7 | 0.696 / 0.693 / 0.696 | 0.696 | BEHIND |
| 1c cpu G26Q (Gemma 4 26B-A4B) @128 greedy | — | — | — | — | — | — | **not run** (harness crash, below) |
| 2 cuda G20 @128 greedy, ctx 2048 | — | — | 26.2 / 26.2 / 26.2 | 26.2 | — | — | **VOID** (goinfer's swap grew +14.4 MB) |

- **1a and 1b:** goinfer's decode paths were `cpu (q4k)` and `cpu (int4mix)`; swap growth was 0.0 MB on every arm.
- **1c did not run:**
  - `bench_peer.py` raised `KeyError: 'G26Q:128'` when building the cell's prompt. The `G26Q` key was added without a
    calibrated prompt in `scripts/prompts.json`, which the by-day smokes did not exercise (they loaded serve directly).
  - The crash came after 1a and 1b, the step's last cell.
  - Fixed 2026-10-08 by calibrating `G26Q:128` with `bench_prompts_calibrate.py`: 129 tokens, the same text as
    `M26:128`, the same tokenizer family.
- **2 is VOID by the registered swap rule:**
  - goinfer's arm decoded `cuda-resident (int4mix)`, but system swap grew 14.4 MB during its cell (from before the
    server started to its last completion), and any growth voids the arm.
  - The harness drops a voided arm's runs, so no goinfer figure is reported.
  - Ollama's arm (26.2 tok/s, swap +0.0 MB) has nothing to pair with.
  - The void stands for this night.

### The re-run, registered 2026-10-08 before it runs (owner: "reschedule nobara and kick it off")

- **The cells:** the two that produced no graded cell, 1c and 2. They run again as a second, separate measurement,
  started by day on the owner's word.
- **What is unchanged:** cells, rules, bands, void conditions and binaries (the same `65b2c22a` serve builds).
  - Night 1's VOID for cell 2 is not re-graded. If the re-run is valid, it is the reported cell, with this history
    stated beside it.
- **One change, the fix above:** the harness runs from a worktree with `G26Q:128` in `scripts/prompts.json`.
- **Script:** `peer-vetted-2026-10-07/run-nobara-2.sh`, results in `~/goinfer-bench/peer-vetted-2026-10-07/results-2/`.
