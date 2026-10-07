# Multimodal support: every image and audio model on every backend, at a usable speed (2026-10)

**Status: PLANNED 2026-10-06. Nothing below has started.** Each phase writes its own gates into this doc, and commits
them, before its first measurement (CLAUDE.md, "Pre-registration").

The owner, 2026-10-06, after the multimodal plan (`docs/multimodal.md`) was finished and aikit v1.58.0 shipped:
"we want better overall support." Three axes:
- **Which models** take images and audio.
- **Which backends** run them on the GPU (CPU, CUDA, Metal, WebGPU).
- **Whether they are fast enough** to use interactively.

The last phase puts the answer where users look first, the README, with a check that keeps it true.

## Where it stands (2026-10-06)

"Tower / decoder": where the image or audio encoder runs, then where the language model runs after it.

| | CPU | CUDA | Metal | WebGPU |
|---|---|---|---|---|
| Gemma 3 | CPU / CPU | GPU / GPU | CPU / GPU (read, never run) | GPU / GPU |
| Gemma 4 E2B, E4B | CPU / CPU | CPU / GPU (since 2026-10-07, S1 on CUDA) | GPU / CPU | CPU / CPU |
| Gemma 4 26B, 31B | CPU / CPU | CPU / GPU | GPU (checked on E2B only) / GPU (read) | CPU / CPU |
| Qwen2.5-VL | CPU / CPU | CPU / GPU | CPU / GPU (read, never run) | CPU / GPU |
| Qwen3.5+ dense | CPU / CPU | CPU / GPU | CPU / CPU | CPU / CPU |
| GLM-OCR | CPU / CPU | CPU / GPU | CPU / CPU | CPU / staged |
| EmbeddingGemma 2 (text, image, audio) | CPU | CPU | GPU | CPU |

**Gaps in coverage:**
- **Models:**
  - Qwen3.5+ MoE images have never been run.
  - Gemma 4 E4B and 31B were never validated (no checkpoint on either box).
  - Qwen3-VL is text only.
  - Ministral 3 (Pixtral), LFM2.5-VL and North have no tower.
- **Audio** is EmbeddingGemma 2's embeddings only. Gemma 4 E2B/E4B audio into the model is not wired, though its tower
  (aikit `audio`) exists.
- **Release binaries:** WebGPU needs cgo and is in no release binary.

**Speed** (the most recent read of each, exploratory unless marked):
- **Usable:**
  - Gemma 3 on CUDA: a 4.1 s tower (recorded 2026-09-21).
  - Qwen3.5 on CUDA: 3.6 s for a new image, 0.3 s to resend one.
  - EmbeddingGemma 2 on Metal: 1.2-2 s an image, 70-180 ms an audio clip.
  - The Gemma 4 tower on Metal: 1.3-2.3 s.
- **Slow:**
  - Gemma 4 E2B image chat on a Mac: 14-16 s for 32 tokens, mostly CPU decode.
  - Qwen3.5 on the CPU: 7.7 s TTFT on a new image.
- **Very slow:**
  - Gemma 3's CPU tower: 31 s (recorded 2026-09-08).
  - GLM-OCR's tower: 29 s at 1 MP, minutes at its full size.

## Rules for every phase

- **Gates first.** Each phase writes its gates here and commits them before measuring.
  - The correctness bar is the one its neighbours use: bit-identical where the arithmetic allows; otherwise cosine
    >= 0.9999 per row against the CPU path or HF, with an ambiguous band parked for the owner.
  - Plus identical greedy tokens on a served request where a decoder is involved.
  - Plus planted defects that turn the gate red. Randomise any tiny fixture whose weights are all ones (memory:
    gemma4-vision-tiny); check one-ulp effects at the float level, not only in uint8 (F1 of `docs/multimodal.md`).
- **Speed** is a record, not a gate, unless the phase pre-registers a bar. By day it is exploratory and labelled so;
  timed A/Bs and peer comparisons go on the night queue (CLAUDE.md, "Run budget").
- **aikit:** new API is batched into one release per cycle. A phase that needs it develops against a local aikit branch
  in a separate worktree and does not push goinfer main until the tag exists (the 2026-10-06 pattern: memory
  eg2-gpu-branch-worktree).
- **Checkpoints** come from `~/models` (`models-pull`), never the archive, for anything timed. The Mac has about 26 GB
  free after the 2026-10-06 cleanup, so a large download needs space first.

## Phases, in order of payoff

### S1 — Gemma 4 E2B/E4B decode on the GPU (Metal, then CUDA)

The largest single speedup. It is not only for images: E2B/E4B text is CPU-only on every backend today. No backend
declares `FeatGemma4EModel` (`decoder/features.go`), so these sizes fall back to the CPU rather than silently drop:
- **PLE:** a per-layer input embedding.
- **Shared KV:** later layers reuse an earlier layer's keys and values.
- **A per-layer FFN width.**

`docs/tasks/task-gpu-paths-2026-09.md` lists the gap. This task builds it, and that row points here once started.

- **Metal first.** EmbeddingGemma 2's Metal encoder already runs a PLE block (`metal/embeddinggemma2.go`,
  `eg2_gelu_mul_ple`).
  - Shared KV is resident-cache indexing: a layer reads another layer's slot.
  - Per-layer FFN width is per-layer scratch sizing.
  - Then the resident decode bridge for E-models, and image turns through it (`GenerateGemma4VL`).
- **Gates to write before starting:**
  - Resident against CPU logits on the tiny E-model fixture, every position, with PLE, shared KV and the varying FFN
    width all exercised.
  - Real E2B text: identical 32 greedy tokens against the CPU on a fixed prompt set.
  - Real E2B image chat: identical served replies against the CPU decoder (the F2b shape).
  - Planted defects: PLE skipped, the shared-KV source layer off by one, one FFN width.
- **Speed (night):** E2B decode tokens per second, resident against CPU, same-session interleaved.
- **Size:** M-L per backend. No aikit API expected.

**S1 Gate 0, read 2026-10-06** (a desk map with file:line citations:
`docs/measurements/multimodal-support-2026-10/s1-gate0.md`).
- **Two existing bugs in shipped Gemma 4 GPU decode**, each checked against the code and against transformers 5.19
  before this was written:
  - **(a) Metal never applies the per-layer output scalar on a dense Gemma 4 layer.** It does so only in the MoE join
    (`metal/gemma4_moe.go`). CUDA and WebGPU apply it on dense layers.
  - **(b) No GPU backend applies `v_norm` on a layer that owns its own `v_proj`.**
    - HF applies the scale-less `v_norm` on every layer that owns its K/V (`modeling_gemma4.py`), and so does
      goinfer's CPU path (`decoder/forward_gemma4.go`).
    - Metal and CUDA apply it only on K=V layers, and WebGPU has only a K=V kernel.
    - This affects the sliding layers of the 12B/26B/31B now running resident, and would affect every E2B layer.
  - **Why the gates missed both:** they are loose (pos0 >= 0.97 on the scaled dense test, a 0.90 backstop on the
    two-geometry test, 0.88 on the 26B), and the post-attention norm partly hides a V scale.
- **What the E-model needs on Metal:**
  - **PLE** needs the token id, which the resident interface does not carry. Plan: the host computes the per-layer
    inputs with the CPU's own code and appends them to the embedding row (`[H || L*P]`), with strict length checks at
    every Metal entry. The per-layer branch is built from kernels Metal already has.
  - **Shared KV** is buffer aliasing (Metal's KV is linear). The traps: the K/V store must be skipped on shared layers,
    the slot copies, `UploadKV`, and the KV-memory estimate.
  - **The per-layer FFN width:** Metal sizes one width for the model, from layer 0 (6144). E2B's 20 shared layers are
    12288, so admitting E2B as it stands would overflow the gate/up scratch silently.
  - **The fixture:** no tiny E-model fixture exists, and safetensors PLE loading is refused, so one is built first
    (S1.1).

**S1 gates, written 2026-10-06 before any S1 measurement:**
- **S1.0, the two fixes, on Metal:**
  - Gemma 4's resident-vs-CPU parity tests (the scaled dense and two-geometry fixtures) are read before and after the
    fix, with each test's own metrics.
  - **PASS:** no metric moves the wrong way after the fix, and re-dropping each fix through a test seam reproduces
    its before-fix numbers.
  - Each existing bar is then tightened in a separate, recorded amendment. The fix is the mechanism; the new bar sits
    between the before and after readings. The bars are never loosened.
  - The 26B is re-checked on the night queue.
  - CUDA's and WebGPU's `v_norm` fix runs as its own pass on nobara, under the same rule, before S1 reaches CUDA.

  **S1.0 read 2026-10-06, 19:14 PDT, on the Mac: PASS.**
  - **The fix:** `metal/model.go` gains the dense layer scalar (a `layer_scale` kernel at each dense Gemma 4 layer's
    end) and `v_norm` on every Gemma 4 layer that owns its K/V. The test seams `GOINFER_S10_DROP=scalar|vnorm|both`
    re-drop either fix.
  - **The readings,** each test's own summary metric, before (both dropped) → after:

    | test | before | after |
    |---|---|---|
    | scaled dense, pos0 | 0.982297 | **0.999148** |
    | scaled dense, mean | 0.858647 | **0.926917** |
    | two-geometry localize, layer 0 (local, real V) | 0.987958 | **1.000000** |
    | two-geometry localize, layer 1 (global, K=V) | 0.980768 | **0.999940** |
    | two-geometry resident, min | 0.981251 | **0.999761** (max abs 0.21 → 0.024) |
    | MoE localize, worst layer | 0.999644 | **0.999982** |
    | MoE resident, min | 0.958904 | **0.999809** |

    - Argmax agreement is unchanged (15/16 dense, 8/8 two-geometry, all MoE).
    - Each fix alone moves its own tests: the scalar alone lifts the scaled-dense mean to 0.880, `v_norm` alone to
      0.906.
    - Two individual scaled-dense positions read lower after (pos 2 0.918→0.907, pos 12 0.884→0.836). At those
      positions the CPU's own int4 path is at 0.70 and 0.45 against f32, so they are quantization noise on a
      low-agreement position, not a regression. Every summary metric and every two-geometry and MoE position rose.
  - **A correction:** `TestGemma4TwoGeom_f16ScaleConfound` attributed its ~0.98 residual to "the broader resident
    quant path". It was these two bugs; matched to f16 scales, it now reads 0.999761.
  - **The amendment** (the fix is the mechanism; each bar sits between its before and after readings):

    | bar | old | new |
    |---|---|---|
    | scaled dense pos0 | 0.97 | 0.99 |
    | scaled dense mean | none | 0.90 |
    | two-geometry localize, each layer | 0.95 | 0.999 |
    | two-geometry f16-scale and resident, min | 0.95 / 0.90 | 0.995 |
    | MoE localize | 0.90 | 0.9999 |
    | MoE resident, min | 0.60 | 0.995 |

    Under the new bars, the fixed code passes all six tests, both fixes dropped fail all six, the scalar alone
    dropped fails two, and `v_norm` alone dropped fails five.
  - **The Metal snapshot golden moved,** on the Gemma 4 dense-scaled rows only (3 of 10 checkpoints). With the fixes
    dropped it matches the old golden 10/10, byte for byte, so the fix is exactly what moved them. It is re-baked
    (same M1 Pro, macOS 26.6.2).
  - **Raw:** `docs/measurements/multimodal-support-2026-10/s10-*.log`.
  - **Still owed:** the 26B on Metal (night queue), and the same `v_norm` fix on CUDA and WebGPU (nobara).
  - **The 26B re-check's design, fixed 2026-10-06 before it runs.** Every 26B Metal parity test compares against a CPU
    26B forward, which the Mac must not run (the M26 rule: Metal only, paged, guards on). So it is split across the
    two machines:
    - **Mac, night queue** (`run-s10-26b-dump.sh`, `TestGemma4_26B_s10Dump`): Metal's full logits at every position of
      three G3 prompts plus 16 greedy tokens, in two arms — fixes in, then `GOINFER_S10_DROP=both` teacher-forced over the
      same sequences — under the swap kill-watch. A load-and-run smoke passed by day (24 positions, RSS 3.8 GB).
    - **nobara, afterwards:** the CPU 26B forward over the same sequences, from the same `.int4.metal.giw`, is the
      reference for both arms.
    - **PASS (S1.0's rule):** with the fixes in, neither the mean Metal-vs-CPU logit cosine nor the argmax agreement
      over all positions is lower than with them dropped. Only `v_norm` can move on the 26B: its layers are all MoE,
      and the MoE join already applied its own layer scalar.
    - **Mac half done, night queue 2026-10-06 22:16 PDT (binary at `b1e7c9f6`):** both arms PASS, 117 positions each
      over three prompts, paged (46 and 56 slots/layer), peak RSS 5.9 / 6.7 GB, swap flat under the kill-watch. The dump
      (`seqs.json`, `logits-fixed.f32`, `logits-drop-both.f32`, 117 MB each) is in the Mac's
      `~/goinfer-bench/s10-26b/run-2026-10-06/`, not committed; the grading half is nobara's (`docs/prompts/nobara-s1-2026-10-07.md`).
      Raw log: `docs/measurements/multimodal-support-2026-10/s10-26b-dump-night.log`.
- **S1.0's 26B re-check, graded 2026-10-07 on nobara: PASS.** The Mac's dump (Metal, `gemma4-26b-int4-v14st.metal.giw`, 117
  positions over three prompts, both arms' dump runs `--- PASS`, swap flat, the kill-watch silent; provenance in the dump
  directory, binary at `b1e7c9f6`) was graded against the CPU 26B forward over the same sequences, from the same `.giw`
  (`TestGemma4_26B_s10Grade`, 158 s, CPU, from the archive: a correctness run).

  | arm | mean Metal-vs-CPU logit cosine | argmax agreement |
  |---|---|---|
  | fixes in | **0.990848** | 82/117 |
  | `GOINFER_S10_DROP=both` (fixes dropped) | 0.940854 | 71/117 |

  - **PASS (S1.0's rule):** with the fixes in, neither number is lower than with them dropped; both are higher (+0.050 in
    mean cosine, +11 positions in argmax). Only `v_norm` can move on the 26B (all its layers are MoE, whose join already
    applied its own layer scalar), so this is the `v_norm` fix on a real checkpoint.
  - **A caveat the rule does not test:** 70% argmax agreement (82/117) is low next to the 94-95% the dense models score
    in G3, even with the fix. This pairs Metal's W4A8 MoE against the CPU's int4 MoE over 117 positions of a 26B whose
    router flips top-k under tiny input noise (memory: a bit-identical router still flips 779 of 3200 decisions under ~0.5%
    noise), so I expect it is that floor and not a defect, but this run does not separate the two: it has no
    near-tie rule and no per-position margins. If the owner wants that, it is a follow-up with the G1 near-tie rule over
    these logits.
  - Raw: `docs/measurements/multimodal-support-2026-10/s10-26b-grade.log`. The 245 MB dump itself is not committed.
- **S1.0, the `v_norm` fix on CUDA and WebGPU, read 2026-10-07 on nobara (RTX 2070 SUPER, driver 595.91.07): PASS.**
  - **The fix:** every Gemma 4 layer that owns its K/V now applies the scale-less `v_norm`, K=V or not. CUDA: a per-layer
    `vNorm` flag (`cuda/backend.go`), the decode launch in `segA` and the batched-prefill launch in `cuda/prefill.go`
    now key on it, and the unit weight is allocated when any layer has it. WebGPU: `runLayer.vNorm`, and a non-K=V
    layer runs the existing `vNorm` shader on its `v_proj` output into a fresh buffer (one buffer cannot be both the
    shader's read and write binding), after the LoRA hooks. The test seam `GOINFER_S10_DROP=vnorm|both` re-drops it on
    both backends. The dense layer scalar was already right on both, as the brief said.
  - **The readings,** each test's own metric, before (`v_norm` dropped) → after:

    | test | before | after |
    |---|---|---|
    | CUDA scaled dense, pos0 / mean | 0.996182 / 0.910545 | **1.000000 / 0.999634** (argmax 15/16 → 16/16) |
    | CUDA two-geometry, min | 0.977972 (max abs 0.20) | **0.999942** (max abs 0.012) |
    | CUDA scaled MoE, pos0 / mean | 0.998601 / 0.960917 | **1.000000 / 0.996078** |
    | CUDA tiny MoE resident, pos0 / mean | 0.999581 / 0.947512 | **1.000000 / 1.000000** |
    | CUDA MoE localize, worst decision (wgt, x1, x2) | 0.999731, 0.999125, 0.999387 | **1.000000** on all three |
    | WebGPU two-geometry, min | 0.977930 (max abs 0.20) | **1.000000** (max abs 1.8e-07) |
    | WebGPU scaled dense, pos0 / min | 0.996182 / 0.800099 | **1.000000 / 0.975300** (argmax 15/16 → 16/16) |

    - **PASS (S1.0's rule):** no metric moved the wrong way, and re-dropping the fix through the seam reproduces every
      before-fix number exactly (the same digits, on the same binary).
    - **The recorded device baselines moved, on the Gemma 4 rows only.** `cuda/testdata/prefill_tails_baseline.json`
      and `keqv_copy_baseline.json` are bit patterns recorded from before earlier changes; with `v_norm` re-dropped both
      tests pass against the old files (15 tail outputs and 4 runs bit-identical), so the fix is exactly what moved them,
      and they were re-recorded. The `qwen3moe-tiny-k3` rows did not change.
  - **The amendment** (separate commit; each bar sits between its before and after readings, none loosened):

    | bar | old | new |
    |---|---|---|
    | CUDA scaled dense pos0 / mean | 0.97 / relation to the int4-vs-f32 floor only | 0.999 / 0.99 added |
    | CUDA two-geometry min | 0.97 | 0.995 |
    | CUDA scaled MoE pos0 / mean | 0.97 / floor only | 0.9995 / 0.98 added |
    | CUDA tiny MoE resident pos0 / mean | 0.97 / floor only | 0.9999 / 0.99 added |
    | CUDA MoE localize, each branch | 0.99 | 0.9999 |
    | WebGPU two-geometry min | 0.97 | 0.995 |
    | WebGPU scaled dense pos0 / min / argmax | 0.97 / none / 15 of 16 | 0.999 / 0.9 / ~~16 of 16~~ 15 of 16 restored (see the correction below) |

    Under the new bars the fixed code passes all seven tests and `v_norm` re-dropped fails all seven
    (`s10-cuda-amended-*.log`, `s10-gpu-*.log`).
    - **Correction, 2026-10-07 (from the Mac, which read it on its own GPU):** the WebGPU argmax bar of 16/16 was a single device's reading.
      The Apple GPU's WebGPU reads **15/16 with the fix and 15/16 without it**, so `TestGemma4DenseScaled_webgpuParity` was red on the Mac with
      nobara's changes removed too. An argmax count is a near-tie flip, not a continuous metric, and I tightened it on one device's before/after pair.
      The bar is back at its pre-S1.0 value of 15/16 (a restoration of the old bar, not a loosening of a tightened one that two devices agree on);
      the fix's signature stays in the cosine bars, which fail with `v_norm` re-dropped on this box (pos0 0.996182 < 0.999, minCosine 0.800099 < 0.9,
      two-geometry 0.977930 < 0.995) and which the Mac passes. Lesson: tighten a count only where the before and after readings differ on every device
      that runs the test.
  - **Not covered:** the real 26B/31B on CUDA (the dump grading below is the Metal side), and the CUDA/WebGPU
    batched-prefill path is exercised through the prefill baselines and the two-geometry/MoE tests, not a separate
    before/after pair. Raw: `docs/measurements/multimodal-support-2026-10/s10-cuda-*.log`, `s10-gpu-*.log`.
- **S1.1, the fixture and safetensors PLE loading, done 2026-10-06 on the Mac.** No bar was pre-registered for this
  step; it builds what G1 runs on. It was held to the f32 tiny-golden convention (`gemma4_moe_forward_test.go`):
  argmax equal and cosine >= 0.99999 against HF, here at every position rather than only the last.
  - **The loader:** `decoder/weights.go` loads the model-level PLE tensors (the token table streamed a row at a time,
    as GGUF does) and each layer's gate, projection and post norm. It refused any PLE checkpoint before.
  - **The fixture:** `scripts/pin_gemma4_emodel_tiny.py` → `testdata/gemma4-emodel-tiny` (gitignored, 9.4 MB, so
    fingerprinted in `testdata/fixture_identity.json`) and `testdata/gemma4_emodel_tiny_golden.json`.
    - Six layers `[s, s, f, s, s, f]`, PLE P=32, the last two layers KV-shared (layer 4 reads 3; layer 5 reads 2, not
      the layer before it), double-wide FFN on those two.
    - One KV head, head_dim 32 local and 64 global, `k_eq_v` off, prompt 12 against a sliding window of 4.
    - All 42 norm weights and the 6 layer scalars are drawn away from identity.
  - **A loader gap found on the way:** transformers 5.16 writes `per_layer_config` (`{"2": {"head_dim": 64}}`) in
    place of `global_head_dim`, so a checkpoint saved by it loaded every layer at the local geometry. goinfer now
    reads that map and refuses overrides it cannot represent (`Config.gemma4GlobalGeometry`).
  - **Result:** `TestGemma4EModel_safetensorsParity`, worst cosine 1.00000000 over all 12 positions, argmax 12/12,
    6-token greedy continuation identical.
  - **Can it go red:** scaling the PLE projection norm by 1.3 → worst 0.9977; scaling the post-PLE norm by 1.3 →
    0.959 with an argmax flip at position 1.
  - **`gate quick` over the change:** 2100 tests passed, 41 forward goldens green; the one red was
    `TestParityManifest_fresh`, cleared by the goldens-gated `scripts/refresh_parity_hashes.sh`
    (`docs/measurements/multimodal-support-2026-10/s11-gate-quick.log`).
- **S1.2, the resident embedding row, done 2026-10-06 on the Mac.**
  - The CPU's per-position PLE-input code moved verbatim into `Model.gemma4PLEInputs`; the CPU forward and
    `embedResidentInto` both call it. For an E-model the resident row is `[h || L*P PLE inputs]`,
    `ResidentEmbedLen()` long; for every other model it is H, as before.
  - **Check:** `TestGemma4EModel_residentEmbedRow`. The head equals the scaled embedding exactly, and the tail matches
    HF's own `get_per_layer_inputs` + `project_per_layer_inputs` (now recorded by the pin script; the checkpoint
    re-pinned byte-identical) to max |diff| 1.19e-6 over all 12 positions. A reused buffer gives the same row.
  - Metal reads the tail since S1.5 (below).
- **S1.3–S1.6, the E-model on Metal, done 2026-10-06 on the Mac.**
  - **S1.3, per-layer FFN width:** each dense layer carries its width and uniform (`residLayer.ffnI`/`uFFNI`); the
    gate|up and down scratch is sized to the widest layer, and the R18 row picks and the staged-down guard hold for
    every width. Byte-identical for every other model: the Metal snapshot golden still matches 10/10.
  - **S1.4, shared KV:** a KV-shared layer (`residLayer.kvShared`) projects Q only, runs `qk_norm` and RoPE over the Q
    heads, skips `v_norm` and the K/V store, and attends over its source's cache, which `r.kc[l]`/`r.vc[l]` and every
    MC1 slot alias. Its geometry must equal its source's or the build refuses. `UploadKV` refuses a shared layer, the
    decoder's prefill bridge skips it, and `ResidentKVBytes` no longer prices one.
  - **S1.5, the PLE branch:** `encodePLE` after the FFN residual and before the layer scalar: `quant_vec` of the raw
    residual, the PLE gate GEMV, a new `ple_gelu_mul` (the clamped `glu_act` GELU-tanh times this layer's input),
    `quant_vec`, the projection GEMV, `rmsnorm_f32` with the post norm, the residual add. Every entry point stages the
    row through one `loadEmb` and checks its length against `H + L*P` (`metalResident.embLen`, refused at build if it
    differs from the decoder's `ResidentEmbedLen`). `ForwardN` runs an E-model token by token; `ForwardSample` gained
    the length check the others had.
  - **S1.6:** Metal declares `FeatGemma4EModel`; CUDA and WebGPU still decline it (admission tests updated). The f16
    prefill declines it (not in `prefillFeatures`), and the MC3 batched step declines it by name.
  - **G1, read 2026-10-06 on the Mac (M1 Pro): PASS.** 18 positions (the golden's prompt and continuation), int4 both
    sides: argmax 17/18 exact, the one mismatch a 0.32% near-tie, no gap over 3%; mean Metal-vs-CPU(int4) cosine
    0.998510 (min 0.995866) against the CPU's own int4-vs-f32 0.777695 (`TestGemma4EModel_metalResidentParity`).
  - **G2, same run: PASS for the six defects that can be seen.** Each alone turns G1 red on the argmax rule:

    | defect | exact argmax | gaps > 3% | mean cosine |
    |---|---|---|---|
    | (1) PLE branch skipped | 2/18 | 16 | 0.374693 |
    | (2) token-identity term zeroed (resident row only) | 9/18 | 7 | 0.965641 |
    | (3) shared-KV source one owning layer off (layer 4 reads 1, not 3) | 7/18 | 9 | 0.887059 |
    | (4) one FFN width for the model | 7/18 | 11 | 0.845776 |
    | (6) layer scalar dropped | 12/18 | 4 | 0.969368 |
    | (7) `v_norm` dropped on non-K=V layers | 9/18 | 8 | 0.921480 |

    Note the mean-cosine envelope alone would have passed every one of them (all above 0.777695); the argmax rule
    caught them. On this fixture the envelope is a crater detector, not a sensitive one.

  **G2 amendment, 2026-10-06, after the G2 run (a mechanism, stated with its evidence):** defect (5), the K/V store not
  skipped on a shared layer, is a byte-for-byte no-op in this design, so no fixture can turn it red. A shared layer's
  Q-only GEMV writes only the Q region of the qkv scratch; its K/V slots still hold what its source (the last owning
  layer of that attention type) stored earlier in the same token, because the other type's writes end below that slot
  (fixture: sliding writes floats 0-191, the global K slot starts at 256; E2B: 0-2559 against 4096). The unskipped
  store re-writes the source's own bytes. Measured: with it planted, Metal's logits are bit-identical to the clean
  run at all 18 positions. That bit-identity is now the check (`TestGemma4EModel_sharedStoreIsNoOp`): if a layout
  change ever breaks it, an unskipped store would corrupt the source cache, and that test goes red first. The skip
  stays in production. This is flagged for the owner, since it was decided after the run.
  - **Raw:** `docs/measurements/multimodal-support-2026-10/s1-g1-g2.log` (G1, G2 and the no-op check) and
    `s13-16-gate-quick.log` (`gate quick` over the change: 2105 passed, 0 failed, lint 17/17).
- **G1, tiny E-model, Metal resident against the CPU, every position, int4 on both sides:**
  - argmax identical, a first divergence where the CPU's top-1/top-2 margin is under 3% counting as a near-tie (the
    two-geometry rule);
  - and mean Metal-vs-CPU(int4) cosine >= mean CPU(int4)-vs-CPU(f32) cosine (the scaled-dense envelope).
- **G2, planted defects, each red on G1:**
  - (1) the PLE branch skipped;
  - (2) PLE's token-identity term zeroed, or its 1/sqrt(2) dropped;
  - (3) the shared-KV source one layer off;
  - (4) one FFN width for the model;
  - (5) the K/V store not skipped on a shared layer;
  - (6) the layer scalar dropped;
  - (7) `v_norm` dropped on non-K=V layers.

  A defect that stays green means the fixture is degenerate along that axis. Fix the fixture, not the bar.

  **G2 amendment, 2026-10-06, before any G2 run (a mechanism, not a reading):** the second half of (2), the
  1/sqrt(2) dropped, cannot go red on logits for any checkpoint. The factor multiplies the whole per-layer input;
  the PLE branch is linear in that input (gelu(gate.h) x input, then a projection) and ends in
  `post_per_layer_input_norm`, an RMSNorm, which removes a uniform scale except through its eps. Measured on the CPU
  against HF while building S1.2: dropping it moves the worst logit cosine from 1.00000000 to 0.99998923. So that
  half is checked where it is visible, on the embedding row's PLE tail against HF's own per-layer inputs
  (`TestGemma4EModel_residentEmbedRow`, red at max |diff| 1.02 with it dropped), and G2's (2) keeps only "the
  token-identity term zeroed". This is the axis being invisible by construction, not the fixture being degenerate,
  so the rule above does not apply to it.
- **G3, real E2B text:**
  - 8 fixed prompts x 32 greedy tokens, Metal resident against the CPU, both int4 from the same GGUF.
  - **PASS:** identical tokens, or a first divergence at a near-tie, on every prompt; and teacher-forced argmax
    agreement >= 99% over all positions.
  - **Ambiguous (parked for the owner):** 97-99%. **Fail:** under 97%.
  - **G3's prompt set and procedure, fixed 2026-10-06 before any G3 run.** The model is
    `~/models/gemma-4-e2b-gguf/gemma-4-E2B_q4_0-it.gguf` (local disk), loaded `Quant: int4` twice: once for the CPU,
    once for Metal's resident (each from its own sidecar of that GGUF). Each prompt is one user turn rendered by the
    GGUF's own chat template (`chat.Detect`), no system message, encoded as the chat CLI does with a template
    (`Encode(prompt, false)`: the template writes its own `<bos>`; corrected 2026-10-06 before any run, the first wording
    said the tokenizer adds it, which would have doubled it). The eight prompts:
    1. `Explain why the sky is blue in two sentences.`
    2. `Write a haiku about autumn leaves.`
    3. `What is 17 multiplied by 23? Show your work.`
    4. `Translate "Where is the train station?" into French and German.`
    5. `List three differences between Python and Go.`
    6. `Summarize the plot of Romeo and Juliet in one paragraph.`
    7. `Write a Go function that reverses a string.`
    8. `What are the main causes of inflation?`
    - **Free-running:** each side prefills token by token and decodes 32 greedy tokens (no early stop on EOS). Same
      tokens, or a first divergence where the CPU's logit gap between its top-1 and Metal's pick is under 3% of
      |top-1| (the two-geometry rule), passes that prompt.
    - **Teacher-forced:** the CPU's sequence (prompt + its 32 tokens) is fed to Metal from a reset cache; agreement is
      Metal's argmax equal to the CPU's at every position that predicts a next token, pooled over the eight prompts.
    - Memory: Metal's own fit guard (`residentFitsMemory`) must admit the model; the test refuses otherwise.
  - **G3 run 1, 2026-10-06 20:53 PDT, on the Mac: FAIL as registered, with a confound in the procedure.**
    - Free-running: 7/8 prompts pass (prompt 3 first diverges at generated token 19, CPU gap 3.61%; 5, 6, 7, 8 diverge
      at near-ties of 2.03%, 0.34%, 1.55%, 1.13%). Teacher-forced agreement 409/435 = **94.02%**, under the 97% fail line.
      Metal's text is coherent throughout and tracks the CPU's.
    - **The confound:** the run met the load-time fit guard by loading each side's pre-built sidecar, and the CPU's is
      the `e4h` one: its embedding/LM-head and PLE token tables are int4 (the CPU default since 2026-09-28; the loader's
      own help puts it at ~2.3 pts top-1), while Metal's sidecar keeps them at the int8 pin. So the run compared two
      different quantizations of the logit-critical tables, not Metal against the CPU on the same weights, which is what
      "both int4 from the same GGUF" was written to mean.
    - Raw: `docs/measurements/multimodal-support-2026-10/g3-run1-e4h-cpu.log`.
  - **G3 amendment, after run 1 (flagged for the owner, since it follows a failing run):** both sides load the same
    sidecar, Metal's (`.int4.metal.giw`, int8-pinned tables); nothing else changes — prompts, rule and bands stand. A
    control arm, reported and not graded, runs the same teacher-forced comparison CPU against CPU across the two
    sidecars (int8 tables against `e4h`), to measure how much of run 1's 94.02% the table precision alone explains.
    Run 2's verdict stands as the G3 reading only if the owner accepts this amendment.
  - **G3 run 2, 2026-10-06 20:56 PDT (amended procedure): FAIL as registered.** 7/8 prompts pass (prompt 3 first
    diverges at generated token 19, CPU gap 5.91%); teacher-forced agreement 413/435 = **94.94%**. Deterministic: a
    re-run that logs each disagreement gives the same 413/435. Of the 22 disagreements, 16 are near-ties under 3% and
    6 exceed it (7.63, 5.91, 4.00, 3.52, 3.49%), four of those six at prompt positions.
    - **Control (CPU int8 tables vs CPU e4h tables, same procedure): 93.56%.** A table-precision change alone, on the
      CPU, disagrees as often as Metal does: run 1's confound was real but was not what failed it.
    - **Localization (exploratory), the residual after every layer, Metal vs CPU, at three positions including the
      worst (prompt 6 pos 18):** smooth drift from 1.00000 at layer 0 to 0.986-0.999 at layer 34; no layer drops more
      than ~0.002, and nothing changes at layer 15 (the first KV-shared layer) or any PLE layer. That is the shape of
      accumulated quantization noise, not of a defect.
    - **Calibration (exploratory, the same `g3Run` on Qwen2.5-Coder-1.5B, whose Metal path is validated): 7/8 prompts,
      94.22%.** A known-good Metal path fails G3's bands by the same margin, so the 99%/97% bands were set without
      calibrating the instrument: Metal's W4A8 (per-tensor int8 activations, f16 scales) against the CPU's int4 path
      flips about 5-6% of argmax positions on these prompts, mostly at near-ties.
    - **Status: G3 FAIL as first registered; the owner re-registered it (below), and it passes that.** Not re-graded under the first rule. One candidate re-registration, for the
      owner to accept or reject: non-inferiority against the calibration (E2B's agreement at least the validated
      model's minus a stated margin, measured in the same session), plus the free-run rule.
    - Raw: `g3-run2.log`, `g3-control.log`, `g3-localize.log`, `g3-calibration-qwen15b.log` in
      `docs/measurements/multimodal-support-2026-10/`.
  - **G3 re-registered (owner decision, 2026-10-06), written before the run that grades it.** Disclosed: the rule was
    written after the owner and I had seen run 2 (94.94%) and the calibration (94.22%); the run below re-measures both.
    - **Instrument:** one test process (`TestGemma4EModel_realE2BNonInferiority`) runs `g3Run` (unchanged: the same
      eight prompts, both sides on one Metal sidecar, int4, a 512-token pinned context) first on Qwen2.5-Coder-1.5B,
      whose Metal path is validated (the reference), then on E2B.
    - **Margin:** 2.0 percentage points of teacher-forced agreement, about two binomial standard errors at ~435
      positions and p near 0.94.
    - **PASS:** E2B agreement >= reference agreement - 2.0 points, and E2B's free-run prompt passes >= the reference's
      - 1.
    - **Ambiguous (parked for the owner):** agreement 2.0-4.0 points below the reference, free-run condition met.
    - **FAIL:** agreement more than 4.0 points below the reference, or E2B's free-run passes 2 or more short.
  - **G3 read 2026-10-06 21:17 PDT on the Mac (M1 Pro), the re-registered rule: PASS.** Reference Qwen2.5-Coder-1.5B
    7/8 prompts, 375/398 = 94.22%; E2B 7/8 prompts, 413/435 = 94.94%; delta **+0.72 points** against the -2.0 margin,
    free-run passes equal. Both numbers reproduce the earlier runs exactly (the procedure is deterministic). Raw:
    `docs/measurements/multimodal-support-2026-10/g3-noninferiority.log`.
- **G4, real E2B image chat:**
  - The F2b request through the Metal serve binary, resident decode against CPU decode.
  - **PASS:** identical reply, or a first divergence at a near-tie.
  - **G4's procedure, fixed 2026-10-06 before any G4 run.**
    - **What changes first:** E2B's image turns did not reach the resident at all: `GenerateGemma4VL` bridged a CPU
      prefill to resident decode only for the bidirectional 26B/31B class. The gate now also admits the E-model class
      (only Metal has a resident for it), and `Generation.DecodeResident` plus one serve log line ("vision: decoded N
      tokens on the resident path") say where a turn decoded. A causal non-E Gemma 4 is unchanged.
    - **Request:** F2b's (`testdata/glm_ocr/table.png`, "What does this image show? Answer briefly.", `max_tokens` 32,
      temperature 0), `--model ~/models/gemma-4-e2b-gguf/gemma-4-E2B_q4_0-it.gguf --vision ~/models/gemma-4-E2B-unq`.
    - **Arms:** one `metal/cmd/serve` binary built at the commit under test. The resident arm, `--backend metal`, must
      report the resident decode path at load and "on the resident path" for the turn, or the run is void. The CPU arm
      is `--backend cpu`; F2a/F2b showed the two towers agree (cosine 1.000000000) and give identical replies, so the
      arms differ in decode.
    - **PASS:** identical reply text. A divergence passes only as a near-tie: the CPU's sequence is replayed in-process
      to the first differing token and the CPU's logit gap between its token and the resident's must be under 3% of
      |top-1| (the G1/G3 rule).
  - **G4 read 2026-10-06 21:30 PDT on the Mac (M1 Pro), `metal/cmd/serve` at `3f198af6`: PASS.**
    - Resident arm: `decode path: metal-resident (int4)`, tower on Metal, "vision: decoded 32 tokens on the resident path".
    - CPU arm: `decode path: cpu (int4)`, tower on the CPU, "on the cpu path". (Serve's CPU arm loads its `e4h` sidecar,
      int4 embedding/LM-head tables against Metal's int8; the replies matched regardless.)
    - Both replies, byte-identical: "This image shows **Table 2: Quarterly unit sales by region (in thousands)** for the
      fiscal year 2025 (FY2025". Wall time 15.3 s and 14.9 s per request (exploratory: each is dominated by the
      286-token CPU prefill and the tower).
    - Raw: `g4-serve-{metal,cpu}.log`, `g4-reply-{metal,cpu}.txt` in `docs/measurements/multimodal-support-2026-10/`.
- **Speed (night):** E2B decode tokens per second, resident against CPU, same-session interleaved, plus the host's
  PLE milliseconds per token. A record; not a gate.
  - **S1.9's procedure, fixed 2026-10-06 before the run:** `docs/measurements/multimodal-support-2026-10/run-s19-speed.sh`
    on the Mac's night queue, from binaries pre-built at `888d1d4f` (`~/goinfer-bench/s19/`).
    - `bench_peer.py`, goinfer only, model `E2B` (added to its table; depth-128 prompt calibrated to 129 tokens),
      Phase A at one depth, backends `cpu` and `metal` from the one serve binary, the harness's defaults (64 tokens x 8
      completions x 2 runs, greedy, instant idle gate).
    - The host's per-token cost of the embedding row plus PLE inputs (`TestGemma4EModel_realE2BPLEHostCost`, 512
      tokens after 32 warm-up), under the timing lock.
    - Recorded as numbers with their provenance; no bar.
  - **S1.9 recorded, Mac night queue 2026-10-06 22:10 PDT (M1 Pro, macOS 26.6.2, AC power, binaries at `888d1d4f`,
    model `~/models/gemma-4-e2b-gguf/gemma-4-E2B_q4_0-it.gguf`, int4, greedy, depth 128 (129 prompt tokens), 64 tokens x
    8 completions x 2 runs per cell, instant idle gate, no thermal warning):**

    | backend | decode tok/s (runs) | decode path | peak RSS |
    |---|---|---|---|
    | CPU | 38.0 (37.92, 37.99) | `cpu (int4)`, row4 layout, `e4h` sidecar | 1.40 GiB |
    | Metal resident | 46.1 (46.09, 46.10) | `metal-resident (int4)` | 0.38 GiB |

    - Metal resident is 1.21x the CPU on E2B decode. A record, not a gate. The CPU arm runs its own sidecar (int4
      embedding/LM-head tables), the shipped default for each backend.
    - The host's embedding row + PLE inputs: mean 0.315 ms/token (median 0.304, p90 0.347, 512 tokens), about 1.5% of a
      Metal token (21.7 ms): not where the time goes.
    - Raw: `docs/measurements/multimodal-support-2026-10/s19-e2b-speed-night.log` (and `s19-bench-e2b.json`).

**S1 on CUDA: plan and gates, written 2026-10-07 on nobara before any CUDA E-model code or measurement.** A desk map of the CUDA
resident against Metal's version (every claim checked against the tree at `24c16cf2`; one research pass, no run) fixes the
plan. Where it says INFERRED, the step names the fallback and what decides it.

- **What is already known, from the code:**
  - **No new CUDA kernel and no PTX regeneration for PLE.** `glu_quant` computes `act(g[k]) * u[k]` plus int8 quant with
    separate `g`/`u` pointers and offsets (`cuda/glue.cu`), and `ACT_GELU_TANH` is 0. The PLE multiply is
    `glu_quant(g=pleG, u=pleIn, uOff=l*P, I=P, act=0)`; every other PLE step reuses an existing launch (`quant_vec`, `doG`
    for the gate and projection, `normF32`, the residual add, `scale_vec`). The audited glue/moe/gemv_fwd PTX trio stays
    pinned at NVRTC 12.6.85, untouched. If a kernel turns out to be needed after all, its PTX is regenerated at the pinned
    NVRTC and that is recorded here before the gate that depends on it.
  - **No new kernel for shared KV.** `rope_kv` with `nKV=0` rotates Q only and skips the K/V store; `fused_rms_qkv` with
    `kvDim=0` never reads Wk/Wv; `qk_norm` with `nKV=0` launches no K blocks.
  - **Three silent failures if E-models were admitted unchanged** (so the steps below remove them in the same change, not
    afterwards): (i) scratch is sized from the model-level FFN width (`FFNPerLayer[0]`, 6144 on E2B) while 20 layers are 12288,
    so `fused_rms_gu` and `glu_quant` would write past `gO`/`uO`/`dq`, a device-memory overrun and not a decline; (ii) the
    batched prefill copies each row with `copy(xhost[m*hidden:(m+1)*hidden], e)`, which silently drops a PLE tail; (iii)
    `BuildResident` packs and validates K/V and K-norm weights that a shared layer does not have (it already errors on them, so
    today CUDA declines E-models by naming the missing feature).
- **Steps, in order. Each is one commit, and each is checked on the tiny fixture before the next starts:**
  - **C0, the fixture on this box.** `testdata/gemma4-emodel-tiny` is gitignored and absent here. Regenerate it with
    `scripts/pin_gemma4_emodel_tiny.py` (`~/.venv-vl`, transformers 5.16.1 as the golden records) and require it to match
    `testdata/fixture_identity.json` and `TestGemma4EModel_safetensorsParity` on the CPU (cosine >= 0.99999 at every position, as
    S1.1). If it does not reproduce, stop and say so: every CUDA reading below is against that fixture.
  - **C1, per-layer FFN width.** Gate/up scratch, `glu_quant` scratch and the launches take `Ly.g.N` / `Ly.d.K` per layer, scratch
    sized to the widest layer. Byte-identical for every non-E-model: the device baselines (`prefill_tails_baseline.json`,
    `keqv_copy_baseline.json`) and the Gemma 4 / dense parity tests must not move.
  - **C2, shared KV.** A KV-shared layer (`Ly.kvShared`, `Ly.kvSrc` from `KVSrcAtResident`) leaves its own `r.kc[l]`/`r.vc[l]` and
    every MC1 slot's zero-valued, projects Q only (unfused and fused), runs `qk_norm` and `rope_kv` with `nKV=0`, skips `v_norm`
    and the K/V store, and attends over its source's cache at every launch site (split-KV, flash-decode, decode attention). Its
    `kEqV` and `vNorm` are false (checked before the source's). `UploadKV` refuses a shared layer by name. `kvBytesForCap` and
    `decoder.cudaKVBytes` stop pricing it, together, and `TestResidentKVBytes_matchesCUDAAllocation` and
    `kvbytes_agreement_test.go` gain the E-model fixture. The build refuses a shared layer whose geometry differs from its
    source's (as Metal does). Aliasing the buffers (Metal's choice) is the fallback only if a launch site cannot be made to
    resolve the source.
  - **C3, PLE.** The resident row is `H + L*P` long; `launchToken` (the funnel every single-token entry point goes through)
    refuses any other length by name. The tail lives in `r.x`'s own allocation (`pleIn = r.x.At(H*4)`) so the existing upload
    carries it; if that does not work on the device, a second buffer and a second upload (INFERRED either way, the first run
    decides, and the choice is recorded). Per layer, after the FFN residual and before the layer scalar: `quant_vec` of the raw
    residual, the PLE gate GEMV, `glu_quant` (GELU-tanh times this layer's input), the projection GEMV, `normF32` with the post
    norm, the residual add. It sits in `segB`, so CUDA graphs capture it; every operand is a fixed-address buffer and `l*P` is a
    per-layer argument. `graphsSelfTest`'s hidden-sized row becomes the augmented length with a non-zero tail.
  - **C4, declines.** `prefillStaticDecline` declines E-models by name (so every batched entry point, MC3 `StepBatch`,
    `ForwardN`'s batched arm and `PrefillPath` report it and the sequential fallbacks take over); `AttachDrafter` refuses an
    E-model target by name; a LoRA adapter on an E-model is declined by name (its `r.inter` K is wrong on wide layers, and nothing
    verifies it). Each decline has a test that asserts the named reason.
  - **C5, declare it.** `FeatGemma4EModel` goes into CUDA's feature set only once C1-C4 pass; WebGPU keeps declining. The
    admission tests, `docs/hardware-matrix.md` (generated), `docs/multimodal.md` and `docs/gpu-residency-coverage.md` move with it.
- **Gates (CUDA's G1-G4, registered now; each reads the resident against the CPU on the same weights):**
  - **G1c, the tiny E-model, every position, int4 on both sides** (the golden's prompt and continuation, 18 positions). Metal's
    G1 rule unchanged: a first mismatch is benign only where the CPU's gap between its top-1 and the resident's pick is under 3% of
    |top-1|; PASS when every mismatch is benign and the mean CUDA-vs-CPU(int4) cosine is at least the mean CPU(int4)-vs-CPU(f32)
    cosine. The test first asserts the resident took the shape (PLE width, two shared layers, the two FFN widths).
    The minimum cosine is reported. After the first reading a separate recorded amendment may TIGHTEN the bars between the
    planted-defect and clean readings, as S1.0 did; they are never loosened.
  - **G2c, planted defects, each alone red on G1c:** (1) PLE skipped; (2) PLE token-identity term zeroed (the decoder seam);
    (3) the shared-KV source one owning layer off; (4) one FFN width for the model; (6) the layer scalar dropped; (7) `v_norm`
    dropped on non-K=V layers; and, CUDA-specific, (8) the PLE per-layer input offset off by one layer (`uOff` = `(l+1)*P`).
    Defect (5), the K/V store not skipped on a shared layer, is registered with its outcome open: on Metal it is a byte-for-byte
    no-op and is checked as such. On CUDA the shared layer's `kB`/`vB` scratch may hold a different layer's K/V (layer 5 is
    full-attention with source layer 2 while the scratch holds layer 3's sliding K/V), so it may go red here. Either way it is
    recorded: red means it is a gate; green means a bit-identity check like Metal's (`..._sharedStoreIsNoOp`) replaces it, and that
    is stated as a mechanism with its evidence, not as a pass. A defect that stays green otherwise means the fixture is degenerate
    on that axis: fix the fixture, not the bar.
  - **G2c-graphs and accounting, same fixture:** a graphs-on run is bit-identical to a graphs-off run (a prompt longer than the
    sliding window, non-zero PLE tail, `GOINFER_CUDA_GRAPHS_UNSAFE` because this box's DEFAULT compute mode declines graphs); the
    KV bytes the planner prices equal the bytes allocated; every decline in C4 returns its named reason and a greedy generation
    through the sequential fallbacks equals the CPU's tokens.
  - **G3c, real E2B text.** Metal's re-registered G3 rule, adopted unchanged and written before any CUDA run: one test process
    runs the eight G3 prompts x 32 greedy tokens (same prompts, same free-run and teacher-forced procedure, 512-token pinned
    context) first on Qwen2.5-Coder-1.5B on CUDA (its CUDA path is validated; the reference), then on E2B. **PASS:** E2B's
    teacher-forced agreement >= the reference's - 2.0 points and its free-run prompt passes >= the reference's - 1. **Ambiguous
    (parked for the owner):** 2.0-4.0 points below, free-run met. **FAIL:** more than 4.0 below, or 2 or more free-run passes short.
    Lessons from Metal's G3 run 1 are applied up front, not as an amendment: both sides run the SAME table precision (the CPU arm
    loads the sidecar matching CUDA's embedding/LM-head and PLE table precision, and the log prints each side's), and a control arm
    (CPU against CPU across the two sidecars, reported, not graded) states how much table precision alone moves the number. Model:
    `~/models/gemma-4-e2b-gguf/gemma-4-E2B_q4_0-it.gguf` (local disk), int4. The fit guard must admit it on the 8 GB card or the
    test refuses.
  - **G4c, real E2B image chat through the CUDA serve binary.** F2b's request (`testdata/glm_ocr/table.png`, "What does this image
    show? Answer briefly.", `max_tokens` 32, temperature 0), `--vision ~/models/gemma-4-E2B-unq`; resident arm `--backend cuda`, CPU
    arm `--backend cpu`, one binary built at the commit under test. The resident arm must report the CUDA-resident decode path at
    load and "decoded N tokens on the resident path" for the turn, or the run is void. **PASS:** identical reply, or a first
    divergence replayed on the CPU at a near-tie (gap under 3%). The tower stays on the CPU (S4 is separate).
- **Speed (night, a record, no bar):** E2B decode tokens per second, CUDA resident against CPU, same-session interleaved through
  `bench_peer.py` (goinfer only, the `E2B` model entry, depth 128), plus the host's PLE milliseconds per token under the timing
  lock. Queued after G3c/G4c pass; nothing is quoted before then.
- **By day / by night:** C0-C5, G1c, G2c, the graphs and accounting checks and G4c are seconds to a few minutes: by day. G3c loads
  E2B twice and runs about 870 teacher-forced positions on the CPU: a guess of 5-10 minutes (no per-position time measured yet), restated
  from a measured time before it starts and queued if it exceeds 10. The speed record is a night job.
- **Open on purpose (INFERRED in the desk map, decided by the first run, each recorded where it is decided):** the `r.x` tail
  against a second upload; whether `gemv_w4a8` handles the fixture's PLE projection at K=32; whether defect (5) is detectable
  on CUDA; whether LoRA touches an E-model at all (declined by name regardless).

- **S1 on CUDA, read 2026-10-07 on nobara (RTX 2070 SUPER, driver 595.91.07, int4): G1c, G2c, graphs, accounting, G3c, G4c PASS.** Pre-registered above; implemented as C0-C5
  in commit `d1cc93c0`. Nothing below is a speed claim; the speed record is owed on the night queue.
  - **What was built, and where it departs from the plan.**
    - **C0:** the tiny E-model fixture regenerated on this box with transformers 5.12.0 (the golden was pinned with 5.16.1): the HF logits differ
      from the committed golden by at most 8.7e-6, argmax and continuation identical, `testdata/fixture_identity.json` and
      `TestGemma4EModel_safetensorsParity` green (worst cosine 1.00000000 over 12 positions). The directory is gitignored and stays untracked.
    - **C1:** per-layer FFN width (`cudaLayer.ffnI`), scratch sized to the widest layer. Byte-identical for every existing model: both device
      baselines (`prefill_tails_baseline.json`, `keqv_copy_baseline.json`) unmoved.
    - **C2, a departure:** KV sharing **aliases the buffers** (`r.kc[l]`, `r.vc[l]` and every MC1 slot's, set to the source's), which the plan named
      only as the fallback. The desk map counted ten launch sites that index `r.kc[l]` (split-KV, flash-decode, decode attention, the rest); aliasing
      needs none of them touched, and it is Metal's choice. A shared layer projects Q only, runs `qk_norm` and `rope_kv` with `nKV=0` (no K/V
      store), and attends over the aliased cache with its source's geometry (checked at build). `kvBytesForCap` and `decoder.cudaKVBytes` skip
      shared layers; `UploadKV` refuses one by name. E-models take the **unfused** QKV chain (the fused kernels need K/V on every layer), so
      `fuseQKV` is off for them: slower, correct, and an optimization for later.
    - **C3:** the resident row is `[hidden || 35*256]` for E2B (`r.x` is allocated that long and the single upload carries the tail; each layer's
      `glu_quant` reads its slice with an element offset, so no buffer view is needed: the open question about `Buffer.At` never arose). The PLE
      branch is `quant_vec`, the gate GEMV, `glu_quant` (act 0, GELU-tanh), the projection GEMV, the post-norm, the residual add, between the FFN
      residual and the layer scalar, in `segB` (so graphs capture it). **No new kernel and no PTX regeneration.** `gemv_w4a8` handled the fixture's
      K=32 projection (G1c passes), which was one of the open questions. `launchToken` refuses a row of the wrong length by name.
    - **C4:** `prefillStaticDecline` declines E-models by name (every batched entry point, MC3, `ForwardN`'s batched arm, and the serve banner reports
      "prefill path: sequential — Gemma 4 E-model ... has no batched prefill"); `AttachDrafter` and `SetAdapter` refuse by name.
    - **C5:** CUDA declares `FeatGemma4EModel`; WebGPU still declines. The admission tests, `docs/hardware-matrix.md` (regenerated; its note had
      said no backend declares the feature, which Metal had already made false) and `docs/gpu-residency-coverage.md` moved with it.
  - **G1c (`TestGemma4EModel_cudaResidentParity`), tiny E-model, 18 positions, int4 both sides: PASS.** Argmax 18/18 exact, mean CUDA-vs-CPU(int4)
    cosine **0.999931** (min 0.999346), against the CPU's own int4-vs-f32 mean 0.777674 (Metal's reading of the same instrument: 0.777695).
    The test first asserts the resident took the shape: P=32, two shared layers, FFN widths 256 and 512.
  - **G2c (`TestGemma4EModel_plantedDefects`), each defect alone: all eight RED**, on the argmax rule (the mean-cosine envelope alone would have passed
    most of them, as on Metal):

    | defect | exact argmax | gaps > 3% | mean cosine |
    |---|---|---|---|
    | (1) PLE branch skipped | 2/18 | 16 | 0.371194 |
    | (2) token-identity term zeroed (decoder seam) | 10/18 | 6 | 0.966228 |
    | (3) shared-KV source one owning layer off | 7/18 | 9 | 0.887704 |
    | (4) one FFN width for the model | 8/18 | 6 | 0.952167 |
    | (5) K/V store not skipped on a shared layer | 5/18 | 11 | 0.763998 |
    | (6) layer scalar dropped | 10/18 | 6 | 0.972024 |
    | (7) `v_norm` dropped on non-K=V layers | 10/18 | 8 | 0.920343 |
    | (8) PLE per-layer input offset one layer off (CUDA-specific) | 0/18 | 18 | 0.194175 |

    - **Defect (5) was registered with its outcome open, and it goes red on CUDA** (it is a byte-for-byte no-op on Metal). The shared layer's K/V
      scratch holds another layer's data, so the unskipped store writes it through the alias into the source's cache. It is therefore a real gate
      here, not a bit-identity check.
  - **Graphs (`TestGemma4Graphs_bitExact_emodel`, `GOINFER_CUDA_GRAPHS_UNSAFE=1`, 40 positions past the sliding window): replay is bit-identical to the
    live launches**; the tautology guard confirms graphs were captured. **KV accounting** (`TestResidentKVBytes_matchesCUDAAllocation` with the
    E-model fixture): allocated bytes equal `kvBytesForCap` equal `ResidentKVBytes("cuda")`, and the shared layers' buffers are asserted to alias.
    **Declines and fallbacks** (`TestGemma4EModel_declinesAndFallbacks`): each decline returns its named reason, a hidden-sized row is refused by
    length, `UploadKV` to a shared layer is refused, and 8 greedy tokens through the sequential fallbacks equal the CPU's.
  - **G3c (`TestGemma4EModel_realE2BNonInferiority`, real E2B and the Qwen2.5-Coder-1.5B reference, 8 prompts x 32 tokens, both sides loading the same
    GGUF with the same options): PASS.** E2B 7/8 prompts, teacher-forced agreement 411/435 = **94.48%**; reference 7/8 prompts, 374/398 = **93.97%**;
    delta **+0.51 points** against the -2.0 margin, free-run passes equal (7 and 7). 144 s end to end. (Metal's reading of the same rule: 94.94% and 94.22%.)
    - **Run 1 was flawed and is superseded.** The CPU side of the reference loaded the CUDA `e4h` sidecar (int4 embedding/LM-head tables) while the
      CUDA side loaded the GGUF with `EmbedInt4` unset (int8 pin): the confound Metal's G3 run 1 fell into, which the plan said to avoid up front. It
      read the reference at 89.20% (355/398) and E2B at the same 94.48%, a delta of +5.29 that came entirely from a depressed reference. Caught from the
      log before it was recorded; the procedure was corrected (both sides load the same file with the same options) and the run repeated under the
      same rule and bands. Raw: `s1c-g3c-run1-asymmetric-tables.log`, `s1c-g3c-run2.log`. E2B's own numbers did not move between the runs, because
      its two sides were loaded symmetrically both times.
    - The plan's table-precision control arm is not run: with both sides on one file and one set of options there is no table-precision difference to measure.
  - **G4c (the F2b request through one CUDA-built serve binary, `d1cc93c0`): PASS, byte-identical replies.** Resident arm `--backend cuda`:
    `decode path: cuda-resident (int4)`, tower on the CPU, "vision: decoded 32 tokens on the resident path". CPU arm: `decode path: cpu (int4)`,
    "on the cpu path". Both replies: "This image shows **Table 2: Quarterly unit sales by region (in thousands)** for the fiscal year FY2025." and
    a start of the next sentence. The CUDA reply differs from the Metal run's ("... fiscal year 2025 (FY2025"), which is a different backend's quantization
    on the same prompt; the gate compares each backend to the CPU. Raw: `s1c-g4c.sh`, `s1c-g4c-serve-{cuda,cpu}.log`, `s1c-g4c-reply-{cuda,cpu}.txt`.
  - **Not done, and not claimed:**
    - **The speed record** (E2B decode, CUDA against CPU, same-session interleaved, plus the PLE host cost) is a night job, not yet queued or run.
    - **E4B** was not run (no checkpoint on either box). The code path is the same shape, and nothing was measured on it.
    - **TTFT:** a long prompt prefills token by token (the batched prefill declines E-models), so time to first token on a long prompt is slow. The
      vision tower is still on the CPU (4.5 s here; S4).
    - **The heavy tier** (`gate gpu`) has not run on this tree; tonight's queued `gate-gpu-cuda` is the first run that exercises the rest of the CUDA suite
      with C1-C5 in.
  - **`gate quick` over the change (18 min, `~/goinfer-logs/gate-quick-s1c.log`): 2085 passed, 0 failed, 429 skipped; lint 16 of 17.** The one red lint step was
    `staticcheck` on `./cuda/...`: two unused fields (`resident`, `cpuInt4v`) I left in the new `gemma4_emodel_test.go`, the dead-field class CLAUDE.md warns
    about. Removed; `staticcheck` for `-tags 'cuda goinfer_testhooks'` and `-tags 'gpu goinfer_testhooks'` re-run clean, the full gate not repeated.

### S2 — GPU towers for Qwen3.5+ and GLM-OCR

On CUDA, after P26b, the Qwen3.5 tower is about 3.3 of the 3.6 s a new image costs. GLM-OCR's tower is the slowest one
in the tree.

- **Approach:** the Gemma 4 Metal tower's.
  - aikit exports the tower's float32 weights (a `Weights()` for `Qwen3VisionEncoder` and `GlmOcrVisionEncoder`; new
    aikit API, next release).
  - goinfer runs the blocks on the device.
  - aikit's tail (merger, projection) stays shared, like `FinishHidden`.
  - Metal and CUDA. The attention blocks, GEMMs and norms from `metal/gemma4_vision.go` largely carry over. Qwen's
    2-D RoPE and its bilinear position-table interpolation are new kernels.
- **Gates to write before starting:**
  - Device tower against aikit's CPU tower, every merged token at cosine >= 0.9999, on the P8a images and on GLM-OCR's
    O3 documents.
  - Served replies identical against the CPU tower.
  - Planted defects in the new kernels.
- **Speed (night):** the tower per image on both boxes, against the CPU tower.
- **Size:** L.

### S3 — Gemma 3 and Qwen2.5-VL on Metal: run what exists, then put the towers on the GPU

- **First, run it.** Metal's `UploadKV` and `ForwardMRoPE` tests have never been run, and no image turn has ever run on
  a Mac (`docs/measurements/multimodal-audit-2026-10-02.md`).
  - Pull `gemma-3-4b-it` and `qwen2.5-vl-3b` (disk first).
  - Run those tests, and one served image turn per family against the CPU decoder.
- **Then the towers.** aikit ships `gpu/visionmetal` (SigLIP) and `gpu/qwenmetal`, tagged and unused by goinfer.
  - Wire them through `EnableResident` (M-15 of `docs/audit-metal-2026-09-12.md`).
  - Gate them like S2, and run a night crossover against the CPU tower.
- **Size:** S to run, M to wire.

### S4 — CUDA towers for Gemma 4 and Qwen2.5-VL

- **Gemma 4:** port `metal/gemma4_vision.go` to `cuda/`. The registration seam is already backend-neutral
  (`multimodal.RegisterGemma4Tower`).
- **Qwen2.5-VL:** wire aikit's tagged `gpu/qwencuda` (a require plus `EnableResident`).
- **Gates:** as S2, on nobara.
- **Size:** M (Gemma 4), S-M (Qwen2.5-VL).

### S5 — Gemma 4 E2B/E4B audio into the model

aikit's `audio` package probably loads E2B's tower unchanged (its config and tensor names were checked on nobara,
2026-10-06; the load was not run). What is left:
- the decoder splice: audio rows unscaled, and PAD for PLE's token-identity term;
- the prompt layout: `<|audio>` n x `<|audio|>` `<audio|>`;
- `input_audio` parts on `/v1/chat/completions`, mirroring the embeddings route.

- **Gates:**
  - The tower's stages against HF on E2B's own weights, on a fixed clip.
  - The full-model logits on an audio prompt against HF `Gemma4ForConditionalGeneration`.
  - A served transcription-style request.
- **Faster with S1:** it is CPU decode until S1 lands.
- **Size:** M-L.

### S6 — Coverage that is cheap once the above exists

- **Qwen3.5+ MoE images:** never run. On nobara, at night; the checkpoint's tower is in the archive, so it is copied to
  `~/models` for anything timed.
- **Gemma 4 E4B and 31B:** download and validate, or scope the support claim to E2B/26B. Owner decision.
- **EmbeddingGemma 2 on CUDA:** text, image and audio. Optional. The Metal kernels show the shapes; the work is a CUDA
  twin.
- **WebGPU:** Gemma 4 and Qwen3.5 there stay CPU unless the owner wants WebGPU invested in. It is in no release binary.
  Owner decision.

### S7 — Measure every cell, once, at night

One night job per box re-measures each table cell on the current binary, with each figure's provenance:
- per-image (and per-clip) time;
- TTFT;
- decode tokens per second after the image.

It replaces the scattered and stale figures above: the 31 s, 29 s and 4.1 s, and the June ones.

- **Speed bar:** a cell is "usable" when a new image's TTFT is under 5 s at the default image budget, on the box's best
  backend. That bar is a proposal for the owner to confirm before S7 runs.
- **Output:** a dated record in `docs/measurements/`. S8 reads its numbers.

### S8 — The support table in the README, kept true

- **The README** gains a short "Images and audio" section: one compact table (family by backend, a check or "CPU"),
  one line on speed with a link, and the models that are not supported yet.
- **The full table** lives in `docs/multimodal.md`'s status block (tower and decoder per cell, with the S7 figures).
- **A drift check,** so neither goes stale silently: a test that reads both tables and checks each claim the code can
  answer.
  - A GPU-tower cell needs the tower registered for that backend (e.g. `multimodal.Gemma4Towers()` in a `metal`
    build).
  - A GPU-decoder cell needs a backend to declare the family's resident features.
  - A model listed as supported needs a serve loader.

  A cell the code cannot answer (a speed figure) carries a date instead.
- **Size:** S, once S7 has numbers.

## Decisions for the owner (none blocks S1)

- P11's audio options (`docs/measurements/multimodal-finish-2026-10-06/p11-audio-comparison.md`):
  - stop at Gemma 4;
  - a Whisper-style front end plus Qwen3-ASR;
  - plus a Whisper encoder (which serves Voxtral), toward pure-Go Whisper;
  - or Voxtral Realtime.
- Several images per message (today a 400).
- Which VL checkpoint `pull` recommends per box class (P9(d)).
- The S7 speed bar.
- WebGPU's future for multimodal.
