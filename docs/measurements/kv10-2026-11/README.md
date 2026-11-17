# KV10 on CUDA: the long-context fidelity harness and the families with no fused prefill kernel (registered 2026-11-16 13:30 PST, before any harness code or run; a 16-core amd64 box, 62 GB RAM, one 8 GB RTX 2070 SUPER)

Ledger row KV10 (`docs/tasks/task-kv-architecture-2026-10.md`, "Long context and lower precision"; lever 4's third step). Francis set the CUDA half, 2026-11-16. Metal's and WebGPU's ceilings and K8V4 are not in this item.

## What the fit checks say before any design (read, 2026-11-16, the server's own plan, `fit-checks.md`)

- **Qwen2.5-Coder-1.5B** (`.gguf`) declares a 32,768 window, so `--ctx 131072` is clamped to it; a 64K or 128K reading on it needs a YaRN-configured copy, and the GGUF loader parses no rope scaling for `qwen2`.
- **Fit at 128K on this card, resident:** `qwen3.5-0.8b` (a hybrid: 6 of its 24 layers attend, head dim 256): **fits at 131,072 in both precisions** (f16: 2,662 MiB on the card; f32: 4,198 MiB). `lfm25-2.6b` declines a resident build (the CPU path). `smollm3-3b` (head dim 128) at 65,536: needs 9.81 GB of KV, 5.61 GB free, **declined**.
  `qwen2.5-vl-3b` (head dim 128) at 65,536: declined for 2.57 GB of KV plus a 3.46 GB companion reserve beside 5.58 GB free.
- **So on this card the one real checkpoint that holds 64K and 128K resident in both KV precisions is `qwen3.5-0.8b`, a family with no fused prefill kernel (head dim 256).** The fused attention kernel (head dims 64 and 128) cannot be read at 64K or more on a real checkpoint here; that is stated as a limit, not worked around.

## KV10-a: the families with no fused prefill kernel (tier 0, no device)

The rule is in code: `backend/cuda/prefill.go:cudaResident.useAttnFused` serves a layer with the fused kernel only when fast attention is on, there is no attention softcap, the pass has at least `attnFusedMinRows` rows, it is above the fast-prefill floor, and `attnFusedFor` has a pipeline for the layer's head dim, which is **64 or 128** only (`backend/cuda/attn_fused.cu`).
Every other layer runs the exact `attn_batched`. The census applies that rule to the `config.json` of every checkpoint under `~/models` (70 configs, 41 groups of model type, head dim and softcap): `fused-census.md`.

| | |
|---|---|
| **Instrument** | `docs/measurements/kv10-2026-11/fused_census.py` (reads configs, applies the rule) and `backend/cuda/kv3_longprefill_test.go:TestKV10_fusedPrefillRule` (a resident loaded on the card: `useAttnFused` answers yes for head dims 64 and 128 and no for 32, 80, 96, 192 and 256, and no under a softcap) |
| **PASS** | the list is written with its rule and sources; the test passes; a plant (the test with `attnFusedFor` answering for 96) is red |
| **Not claimed** | tiny fixtures' head dims (16 and 32) say nothing about a real family; the list is by real configs. A family whose config is not on this box is not listed |

## KV10-b: the 64K/128K fidelity harness (tier 1: a real checkpoint at depth, the lowest tier that answers it)

| | |
|---|---|
| **Question** | At depth, does the engine's lower-precision KV (f16) stay as faithful to its f32 KV as it does at the depths already graded (KV4b's G4: 3,900 and 8,000)? Is anything NaN, or refused, at 64K and 128K? |
| **Instrument** | `backend/cuda/kv3_longprefill_test.go:TestKV10_longContextFidelity` (heavy tier, `GOINFER_HEAVY_TESTS=1`): `~/models/qwen3.5-0.8b` (local NVMe, int4 resident), text of this repository's `decoder/*.go` as the prompt (real tokens, no cycling), prefill depth - 16 tokens, then the **last 16 rows' logits** by `PrefillLastN`, once with `--kv f32` and once with `--kv f16`, fresh load each. Depths **3,900 (the control, graded depth)**, **65,536** and **131,000**; chunking at the build's default |
| **Metric** | over the 16 rows: min cosine of f16's logits against f32's; the argmax agreement count, and over the rows where f32's top-1 margin is at least 0.5 logit; the mean KL(f32 &#124;&#124; f16) in nats; any NaN or Inf |
| **HOLDS** | min cosine at least 0.999, argmax agrees on every row whose f32 margin is at least 0.5, no NaN or Inf, at the depth |
| **DEGRADES (parked for Francis)** | min cosine in [0.99, 0.999), or one or more disagreeing rows with margin at least 0.5, or mean KL above 2 x the control's |
| **FAILS** | min cosine under 0.99, a NaN or Inf, or a crash or a refusal that the fit check did not predict |
| **Reading the control** | the 3,900 cell is the instrument checking itself against a known direction (hybrids' f16 `auto` stays f32 because Qwen3.5-9B's G4 read KL ratio 1.21 at 3,900): a control that reads HOLDS and a deep cell that reads DEGRADES is a depth effect; both DEGRADES is the known hybrid effect, and the deep cells are read against the control's KL |
| **Stopping rule** | all six cells (three depths x two precisions), once; no repeat unless a cell is void |
| **Fit check first** | each load by the server's own plan at its depth (`fit-checks.md`): 131,072 f32 is 4,198 MiB on the card; a refusal is NOT RUN, not a pass |
| **Void if** | a timed lock not held (shared lock: not a timed run), the model read from `/srv/models`, a skip counted as a pass, a bar moved after a reading |
| **Cost, stated before the run** | a prefill of 131,000 ids on the 0.8B at about 1,000 ids a second plus the quadratic attention term: **about 3 to 6 minutes a cell at 131,000, 1 to 2 at 65,536, seconds at 3,900; about 15 to 25 minutes in all**, started and finished times written to the status file. Over 2 hours would stop it |
| **Means** | HOLDS at all depths: the f16 path is as faithful at 128K as at the graded depths on this family. DEGRADES: Francis decides what depth f16 may be offered at. FAILS: the harness's finding, with the cell |

**Not read (stated limits):** the fused attention kernel at 64K or more on a real checkpoint (nothing with head dim 64 or 128 and a native window of 64K or more fits resident beside its KV on this card, above); a YaRN-configured checkpoint (the long end of a model that needs scaling; the candidates that carry YaRN here, `gpt-oss-20b-hf` and `mellum2`, do not fit at 128K, and the GGUF loader parses no rope scaling for qwen2); K8V4; Metal and WebGPU.

## Readings, 2026-11-16

Build `5ecc0d73d`'s tree plus the two tests (`backend/cuda/kv3_longprefill_test.go:TestKV10_fusedPrefillRule`, `TestKV10_longContextFidelity`). Driver 595.112.07 as anchored. All loads from `~/models` on NVMe, under the shared timing lock (a correctness read, not a timed one). The registration above did not change after its commit except a typo in its fit-check bullet and the two tests' file path in the instrument rows (they live in `backend/cuda/kv3_longprefill_test.go`, a new CUDA test file importing `decoder` being refused by `internal/layout`'s ratchet).

### KV10-a: the census (tier 0): PASS

`fused-census.md` (`fused_census.py` over 66 attention-decoder configs under `~/models`). **No fused prefill kernel:** `gemma`, `gemma2` (head dim 256, and a softcap), `gemma3_text`, `gemma4_text`, `qwen3_5_text`, `qwen3_5_moe_text` and `spark2_5` (256), `nemotron_h` 96 and `phi3` (96), `mistral` as `tinymistral-248m` (32). **Fused (head dim 64 or 128, no softcap):** `llama`, `qwen2`, `qwen3`, `qwen3_moe`,
`qwen2_moe`, `qwen2_5_vl`, `qwen3_vl*`, `cohere`, `cohere2`, `gpt_oss`, `granite`, `internlm2`, `laguna`, `mellum`, `ministral3`, `olmo3`, `olmo_hybrid`, `smollm3`, `glm_ocr_text` and the rest listed there. **The rule does not read:** MLA (`deepseek_v2`, `deepseek_v3`), the Mamba-2 hybrids (`nemotron_h`, `granitemoehybrid`: their prompts prefill token by token) and `lfm2` (not CUDA-resident).
`TestKV10_fusedPrefillRule` passes on a resident whose fused module loaded: `useAttnFused` answers yes for exactly head dims [64 128] of every width from 1 to 512, no for 128 under a softcap, no for a pass under `attnFusedMinRows`. The registered plant ("`attnFusedFor` answering for 96") has no code seam; the exhaustive set equality is the guard (a kernel added for another head dim turns it red).

### KV10-b: the harness (tier 1): the verdicts below are by the registered bands

`fidelity-run1.log`, `fidelity.jsonl`. `qwen3.5-0.8b` int4 resident, this repository's text as 131,000 real tokens, the last 16 rows' logits by `PrefillLastN`, `--kv f32` against `--kv f16`, a fresh load each. A calibration of the 3,900 cell alone (14 s, 13:33 PST) preceded the run, which started 13:45 PST and ended 14:43.

| depth | min cosine (f16 vs f32) | argmax agrees (all rows / rows with a 0.5+ margin) | mean KL(f32 &#124;&#124; f16), nats | NaN or Inf | verdict | cell wall time |
|---|---|---|---|---|---|---|
| 3,900 (the control) | 0.993464 | 15 of 16 / 14 of 14 | 1.491e-02 | no | **DEGRADES** | 13 s |
| 65,536 | 0.985323 | 16 of 16 / 12 of 12 | 1.241e-02 (0.83 of the control's) | no | **FAILS** (cosine under 0.99) | 704 s |
| 131,000 | 0.989244 | 15 of 16 / 14 of 14 | 1.344e-02 (0.90 of the control's) | no | **FAILS** (cosine under 0.99) | 2,771 s |

- **The control reads as the registration said it would:** DEGRADES at the graded depth, the known hybrid f16 effect (hybrids' `auto` stays f32).
- **The two deep cells are FAILS by the registered cosine bar, and the registered measures disagree.** Cosine falls under 0.99, but KL is below the control's at both depths, the argmax agrees on every row that has a margin, and nothing is NaN or Inf or refused. This reading does not say the deep cells are worse than the control by KL; it says the cosine bar, set from the control's 0.9935, is crossed.
  The cosine is over the full 248,320-wide logit row; KL weights the rows by f32's probability mass. Which of the two matters for a use is Francis's call; neither was changed after the reading.
- **The registered cost estimate was wrong.** It assumed about 1,000 ids a second and said 15 to 25 minutes; the 65,536 prefill took 326 s (f32) and 371 s (f16), and the 131,000 took 1,301 s and 1,462 s (about 100 ids a second at depth: the exact attention kernels, head dim 256, no fused kernel). The run took 58 minutes, under the 2-hour line; the status file carried the revised estimate from the 65,536 cell on.
- **Fits held as planned:** every load was fit-checked (`fit-checks.md`) and none was refused; 131,072 in f32 is 4,198 MiB on the card.

### Limits, stated

- **Not read: the fused kernel at 64K or more on a real checkpoint** (nothing with head dim 64 or 128 and a 64K+ window fits resident here: `smollm3-3b` and `qwen2.5-vl-3b` are declined at 65,536; `lfm25-2.6b` is not resident). A tiny fixture with a real head dim, or a larger card, would read it.
- **Not read: a YaRN-configured checkpoint.** The candidates that carry YaRN here (`gpt-oss-20b-hf`, `mellum2`) do not fit at 128K, and the GGUF loader parses no rope scaling for `qwen2`.
- **One checkpoint, one prompt, 16 rows a depth:** a first harness, not a graded set. The tail rows are the last 16 of a repository text, not held-out prompts.
- **K8V4, Metal's and WebGPU's ceilings** are not in this item.
