# KV10 on CUDA: the long-context fidelity harness and the families with no fused prefill kernel (registered 2026-11-16 13:30 PST, before any harness code or run; a 16-core amd64 box, 62 GB RAM, one 8 GB RTX 2070 SUPER)

Ledger row KV10 (`docs/tasks/task-kv-architecture-2026-10.md`, "Long context and lower precision"; lever 4's third step). Francis set the CUDA half, 2026-11-16. Metal's and WebGPU's ceilings and K8V4 are not in this item.

## What the fit checks say before any design (read, 2026-11-16, the server's own plan, `fit-checks.md`)

- **Qwen2.5-Coder-1.5B** (`.gguf`) declares a 32,768 window, so `--ctx 131072` is clamped to it; a 64K or 128K reading on it needs a YaRN-configured copy, and the GGUF loader parses no rope scaling for `qwen2`.
- **Fit at 128K on this card, resident:** `qwen3.5-0.8b` (a hybrid: 6 of its 24 layers attend, head dim 256): **fits at 131,072 in both precisions** (f16: 2,662 MiB on the card; f32: 4,198 MiB). `lfm25-2.6b` declines a resident build (the CPU path). `smollm3-3b` (head dim 128) at 65,536: needs 9.81 GB of KV, 5.61 GB free, **declined**.
  `qwen2.5-vl-3b` (head dim 128) at 65,536: declined for 2.57 GB of KV plus a 3.46 GB companion reserve beside 5.58 GB free.
- **So on this card the one real checkpoint that holds 64K and 128K resident in both KV precisions is `qwen3.5-0.8b`, a family with no fused prefill kernel (head dim 256).** The fused attention kernel (head dims 64 and 128) cannot be read at 64K or more on a real checkpoint here; that is stated as a limit, not worked around.

## KV10-a: the families with no fused prefill kernel (tier 0, no device)

The rule is in code: `backend/cuda/prefill_kernels.go:cudaResident.useAttnFused` serves a layer with the fused kernel only when fast attention is on, there is no attention softcap, the pass has at least `attnFusedMinRows` rows, it is above the fast-prefill floor, and `attnFusedFor` has a pipeline for the layer's head dim, which is **64 or 128** only (`backend/cuda/attn_fused.cu`).
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
| **Means** | HOLDS at all depths: the f16 path is as faithful at 128K as at the graded depths on this family. DEGRADES: Francis decides what depth f16 may be offered at. FAILS: the harness's finding, with the cell. **Francis, 2026-11-22: a logged reading (option b).** The bands label each depth's reading, which the test prints and `gate gpu` reports. A cosine FAILS does not fail the test or the gate; a NaN or Inf does (see "What FAILS means", below) |

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

## The frozen text (Francis, 2026-11-22, option A)

**Why.** Until now `kv10Text` built its prompt from the live tree: the non-test `decoder/*.go`, `docs/*.md` and `backend/cuda/*.go`, sorted, first N tokens. So the verdict moved with any edit to those files. At v0.25.0 (`docs/releases/v0.25.0-checklist.md` §3), one `36dbfbaf1` test binary read 0.991232 at 65,536 on `e0d661712`'s text and 0.985122 on `36dbfbaf1`'s. The two texts are byte-identical up to byte 213,566, where one line of `decoder/backend_view.go` changed (`Dims()`'s result list became `ResidentDims`), about 60 tokens before the 16 scored rows (`docs/measurements/release-v0.25.0/kv10-text/`).
The 0.99 bar sits inside that spread, and every deep reading so far lies on one side or the other of it: 0.9851, 0.9853, 0.9892, 0.9909, 0.9912.

**What.** The prompt is now the text `kv10Text` built at `157e2ca29`, the commit of the first reading above. That text is identical to `5ecc0d73d`'s, since the diff between the two touches only the test file. It is chosen because it is what the registration graded, not because of a verdict.
`backend/cuda/testdata/kv10_text_157e2ca29.txt.gz` holds the whole files `decoder/actquant_hazard.go` through `decoder/fitguard.go`: 78 files, 526,468 bytes, sha256 `1cd7d3327d3488527f5c77789af8c4eacdf0d82ef9aa34fbc3939e1b3ec7ed0d`, 157,831 bytes gzipped. `freeze_text.py` rebuilds it from git byte for byte. It is a byte prefix of the full 3,258,173-byte text, and its first 131,000 and 150,000 tokens equal the full text's (HF tokenizers 0.23.1 on the checkpoint's `tokenizer.json`).
`kv10TextBytes` reads only that file, `kv10Text` declares the cell VOID on any other hash, and `TestKV10_textIsFrozen` (no device) holds the hash. The bar (0.99) and the bands did not change.

**What the earlier readings now mean.** The registration's readings at `157e2ca29` (0.993464 at 3,900, 0.985323 at 65,536, 0.989244 at 131,000) are on this exact text. So a re-read at a later build differs from them only by code.

### Reading on the frozen text, 2026-11-22: FAILS at 65,536, the registration's reading reproduced exactly

The run was `run-frozen.sh` at `3145faeea`, from a clean worktree, under the exclusive timing lock. It started 09:24:21 PST and ended 09:36:12. Load average 2.28 at the start, 63.1 GB available. Driver 595.112.07 as anchored. The card held only the desktop compositor's context (`kwin_wayland`, 18 MiB). `qwen3.5-0.8b` was read from `~/models` on NVMe.
Records are in `frozen-2026-11-22/` (`run.log`, `fidelity-frozen.jsonl`). The bar and the bands are the registered ones.

| depth | min cosine (f16 vs f32) | argmax agrees (all rows / 0.5+ margin) | mean KL, nats | NaN or Inf | verdict | the registration's reading at `157e2ca29` |
|---|---|---|---|---|---|---|
| 3,900 (the control) | 0.993464 | 15 of 16 / 14 of 14 | 1.491e-02 | no | DEGRADES | 0.993464, 1.491e-02, DEGRADES |
| 65,536 | **0.985323** | 16 of 16 / 12 of 12 | 1.241e-02 (0.83 of the control's) | no | **FAILS** (cosine under 0.99) | 0.985323, 1.241e-02, FAILS |

- **Both cells equal the registration's to the last float digit** (`fidelity.jsonl`: min cosine 0.9853234643651604, mean KL 0.012412510429789178 at 65,536; 0.9934641567702915 and 0.01490808953608066 at 3,900). This says three things. The frozen text is the registered input: the harness tokenizes it with goinfer's own tokenizer and gets the same cells. The code path this harness reads computes the same logits at `3145faeea` as at `157e2ca29`. And the v0.25.0 red (0.985122 on `36dbfbaf1`'s text) was not a regression.
- **The registered bar on the registered text reads FAILS, as it did on its first reading.** The pre-flight's pass on `e0d661712`'s text (0.991232) was that text's last 16 tokens. By the other registered measures the deep cell is no worse than the control: KL below it, every margin row agreeing, no NaN or Inf. That is the disagreement the first reading already recorded.
- **What FAILS means for this harness** was left to Francis on this reading; he decided it the same day (next section).

## What FAILS means: a logged reading (Francis, 2026-11-22, option b)

**The decision.** KV10 is a logged reading, not a release gate. The 0.99 bar and the bands did not change; they now label the reading. A cosine FAILS does not fail the test or `gate gpu`.
A NaN or Inf in the logits still fails both. That part is the coordinator's reading of Francis's word, stated in the PR: a NaN is a correctness defect, not a fidelity reading.

**The wiring.**
- `kv10Judge` labels each depth by the registered bands and fails only on a NaN or Inf. `TestKV10_judge` (no device) holds both halves: the frozen text's 65,536 reading (0.985323) is labelled FAILS and passes, and a NaN fails at any cosine.
- Each depth prints `GATE_READING KV10 depth N: ... : <verdict>` to stdout.
- `gate gpu` prints every `GATE_READING` line of a heavy process as `<group> reading: ...` beside its memory line, and keeps them in the process's record row (`readings`). `TestHeavyProcLines_readings` holds this through the same `go test -json` parse the run and the replay use.
- A check of the real stream (depth 3,900 alone, the exclusive lock, 2026-11-22 10:19 PST, `frozen-2026-11-22/gate-reading-event-3900.jsonl`) gave the event `{"Action":"output","Test":"TestKV10_longContextFidelity/depth3900","Output":"GATE_READING KV10 depth 3900: min cosine 0.993464, ... DEGRADES; 13 s\n"}` and a pass.

**A NaN fails even from finite logits.** `kv10Compare` keeps a NaN cosine (a row with a zero norm) as the minimum; `c < minCos` alone
had dropped it, so the depth read as its other rows. `kv10Judge` treats a NaN or Inf cosine or KL as a NaN in the logits.
`TestKV10_judge` and `TestKV10_compareKeepsANaNCosine` hold both, red first. `kv10Judge` returns the widened flag. The report (`kv10Depth`) then prints `NaN/Inf true` on such a depth, gives no drift notice for it, and writes its non-finite figures to the JSONL as the strings `"NaN"`, `"+Inf"` or `"-Inf"`, with the marshal error checked (`TestKV10_depthReportOnAMetricNaN`).

**A drift is a notice, not a gate.** `kv10Drift` compares each depth with the reading recorded on this text, at the printed precision (cosine to six places, KL to four figures). The recorded readings are 3,900 and 65,536 from the table above, and the registration's 131,000. It checks only the default checkpoint.
When they differ, the test prints a `GATE_NOTICE` line. `gate gpu` shows it as `kv10 notice: ...` beside the reading, keeps it in the process's record row and the run record (`notices`), and lists it in the verdict block under "Notices". The notice never fails the test, the run or its exit code (`TestGradeGPU_noticesDoNotChangeTheVerdict`).

**So a move shows.** The reading on the frozen text is recorded in `frozen-2026-11-22/fidelity-frozen.jsonl` and in the table above: 0.993464 / 1.491e-02 at 3,900 and 0.985323 / 1.241e-02 at 65,536, the same as `157e2ca29`'s `fidelity.jsonl`. The text is fixed and the output was bit-identical across six days of code, so a `gate gpu` reading that differs from these numbers is a code change. It is worth reading even though it does not fail the gate.
