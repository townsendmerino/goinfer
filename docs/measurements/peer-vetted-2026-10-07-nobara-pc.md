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

*(not yet run)*
