# Multimodal support: every image and audio model on every backend, at a usable speed (2026-10)

**Status: ACTIVE.** S1 is done on Metal and CUDA (2026-10-07; Metal and CUDA speed records taken or queued); S2 onward
not started. The phases were extended on 2026-10-07 (S9-S14, owner decisions below), and the order they run in is in
"Order of work". Each phase writes its own gates into this doc, and commits them, before its first measurement
(CLAUDE.md, "Pre-registration").

The owner, 2026-10-06, after the multimodal plan (`docs/multimodal.md`) was finished and aikit v1.58.0 shipped:
"we want better overall support." Three axes:
- **Which models** take images and audio.
- **Which backends** run them on the GPU (CPU, CUDA, Metal, WebGPU).
- **Whether they are fast enough** to use interactively.

The last phase puts the answer where users look first, the README, with a check that keeps it true.

## Where it stands (2026-10-06; the E2B/E4B row updated 2026-10-07 for S1)

"Tower / decoder": where the image or audio encoder runs, then where the language model runs after it.

| | CPU | CUDA | Metal | WebGPU |
|---|---|---|---|---|
| Gemma 3 | CPU / CPU | GPU / GPU | CPU / GPU (read, never run) | GPU / GPU |
| Gemma 4 E2B, E4B | CPU / CPU | CPU / GPU (since 2026-10-07, S1 on CUDA) | GPU / GPU (since 2026-10-06, S1) | CPU / CPU |
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

## Phases (numbered as added; the order they run in is "Order of work", below)

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
    - **The speed record** (E2B decode, CUDA against CPU, same-session interleaved, plus the PLE host cost) is queued for tonight as `s1c-e2b-speed`
      (`run-s1c-speed.sh`, pre-built binaries at `928f9a41`, est 30 min, the harness defaults, no bar) and has not been read. One exploratory smoke of the
      script (`BENCH_RUNS=1`, 2.5 min, not a result and not to be quoted) showed the plumbing works end to end: both cells passed the token gate, and the
      host's embedding row plus PLE inputs cost 0.33 ms/token (Metal's reading on its own CPU: 0.315).
    - **E4B** was not run (no checkpoint on either box). The code path is the same shape, and nothing was measured on it.
    - **TTFT:** a long prompt prefills token by token (the batched prefill declines E-models), so time to first token on a long prompt is slow. The
      vision tower is still on the CPU (4.5 s here; S4).
    - **The heavy tier** (`gate gpu`) has not completed on this tree. The first night run (2026-10-07, stopped by the owner at minute 35 of the heavy
      tier, `~/goinfer-logs/gate-gpu-night-2026-10-07/`) was red for three reasons, none a defect in the E-model path, all fixed and pushed:
      (1) **my** helper `mustResident` sat in a file tagged `cuda` alone but called a test hook, so the no-testhooks cuda test package did not build
      (group 2a; present since f95fabfd, invisible to CI, which cannot build cuda tests); (2) the gate's pinned worktree linked only the gitignored weights
      of `testdata/gemma4-emodel-tiny`, leaving `model.safetensors` without `config.json`, so 13 E-model tests failed to load (the script now links untracked
      fixture directories whole and excludes them from the dirty-tree check); (3) an **unpinned default resident context** that one KV slot overshoots by the
      build's own scratch declined the resident path to the CPU (`TestB2DenseFlagship` and four Gemma-3 tests; R19 raised the default to 16384 and says it
      is "shrunk to what the card holds", but only the multi-slot arm did), now trimmed to fit with a floor of 4096. The six tests pass again and the full
      non-heavy cuda suite is green. A control run with the heavy tier skipped (`GOINFER_GATE_SKIP_HEAVY=1`, pinned worktree of `6ceb11a6`) reads
      **PASS, 10 of 10 groups**, with the E-model tests running in 2b and 2d. The rest of the heavy tier (about 40 of 78 minutes were never reached) is
      re-queued for the owner to start.
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

**S2 Gate 0, the desk map (2026-10-07, Mac, aikit at v1.58.0 + 2 test-only commits):**
- **Neither tower has a device seam.** `vision.Qwen3VisionEncoder` (`vision/qwen3_encoder.go`) and
  `vision.GlmOcrVisionEncoder` (`vision/glm_ocr_encoder.go`) keep every field unexported, with no `Weights()` and no
  public tail; their tails (`merge`, `downsample`) are unexported. goinfer calls `enc.Forward` from `lm.qwenForward`
  (`internal/serveapp/vision_serve.go`); no backend reaches either loader. aikit's `gpu/qwenmetal` and `gpu/qwencuda`
  serve Qwen2.5-VL only (window plan, RMSNorm hard-coded at 1e-6) and are not reused.
- **Qwen3 (Qwen3.5+ tower), per block:** LayerNorm with weight and bias, eps 1e-6 (f64 accumulation); fused biased qkv
  `[3][heads][hd]`; 2-D RoPE, theta 1e4, `[row·f, col·f]` then `cat(row,row)`, NeoX rotate-half over the full head;
  full attention per image frame, scale `1/sqrt(hd)`; biased proj, residual; LayerNorm, biased fc1, GELU-tanh, biased fc2,
  residual. Before the blocks: a biased patch-embed matmul over `C·T·P·P`, plus a learned 48x48 position table
  interpolated bilinearly (align_corners=True, f32 coordinates, clamped taps, a fixed tap order). After: the merger
  (LayerNorm eps 1e-6; m² patches concatenated **patch-major**; fc1, **erf** GELU, fc2). Real tower
  (`~/models/qwen3.5-0.8b`): 12 layers, hidden 768, 12 heads x 64, patch 16, merge 2, out 1024.
- **GLM-OCR, per block:** RMSNorm (weight only, eps 1e-5); fused biased qkv, then **per-head RMSNorm on q and k before
  RoPE**; the same RoPE and attention; biased proj; RMSNorm; SiLU-gated MLP with biases. No position table. After the
  last block: post RMSNorm, a downsample conv done as a matmul over **channel-major** groups (`x[g][c·m²+j]`, the
  opposite of Qwen3's merger layout), then proj (no bias), LayerNorm eps 1e-5, erf GELU, SiLU-gated MLP. Real tower
  (`~/models/glm-ocr`): 24 layers, hidden 1024, 16 heads x 64, patch 14, merge 2, out 1536.
- **What Metal already has:** goinfer's tower base (`eg2Ops`: f32 GEMM, RMSNorm, per-head norm, rotate-half RoPE,
  attention built from matmuls in 256-row blocks, never T²) and aikit `gpu.ViT` (LayerNorm with bias, biased GEMM
  epilogues, GELU tanh and erf, SiLU-mul). **New:** an attention scale (the base hard-codes 1.0; Gemma 4 needs none), and
  Qwen3's interpolated position table, which depends only on the grid, so the host computes it with aikit's own code.
- **Split, mirroring Gemma 4:** aikit exports the f32 weights (aliasing, refusing a quantized tower) and a public tail
  that takes the last block's output; goinfer runs the patch embed and the blocks on the device; the merger (Qwen3) or
  post-norm, downsample and merger (GLM) stay aikit's on the host, as `FinishHidden` does for Gemma 4.

**S2 steps:** S2.0 the aikit exports (a local branch; released with the cycle's batch). S2.1 a backend-neutral tower
seam in goinfer's `multimodal` for both towers (Gemma 4's registry is the model). S2.2 the Qwen3 tower on Metal. S2.3 the
GLM-OCR tower on Metal. S2.4 serve: `--backend metal` picks the device tower, with named CPU fallbacks (int8 tower, not
registered, declined). The CUDA twins are nobara's, after S2.4, against the same gates.

**S2 gates, written 2026-10-07 before any S2 code or measurement:**
- **G-S2a, the exports:** in aikit, the exported weights plus the public tail reproduce `Forward` exactly (bit-identical:
  the same CPU code in a different order of calls) on the tiny towers (`qwen35vl-vision-tiny`, `glm-ocr-vision-tiny`),
  with norms and the position table randomised first (the all-ones trap); a quantized tower's export errors.
- **G-S2b, the device towers against aikit's CPU tower:** every merged token at cosine >= 0.9999, on the tiny towers at
  three grid sizes, and on the real towers (`qwen3.5-0.8b`, `glm-ocr`) on the four repo images F2a uses plus GLM-OCR's
  three O3 documents at the default cap. Ambiguous (parked): a worst token in 0.999-0.9999.
- **G-S2c, planted defects, each red on G-S2b's tiny-tower check:** (1) the attention scale dropped; (2) RoPE's row and
  column halves swapped; (3) Qwen3's position table added at a transposed grid; (4) GLM's per-head q/k norm skipped;
  (5) the patch-embed bias dropped. A defect that stays green means the fixture is degenerate along that axis.
- **G-S2d, served:** one image request per family through the Metal serve binary (qwen3.5-0.8b and glm-ocr), the tower
  on Metal against the tower on the CPU: identical reply text, or a first divergence at a near-tie under the G1 rule.
- **Speed (night):** tower time per image, Metal against CPU, on the same images. A record.

**S2 progress, 2026-10-07 (Mac), on goinfer branch `s2-towers` over aikit's unreleased `s2-tower-exports`:**
- **G-S2a: PASS.** aikit's exports recompose `Embed` and `Forward` bit for bit on both tiny towers, norms and the
  position table randomised (`TestQwen3Export_recomposesForward`, `TestGlmOcrExport_recomposesForward`); dropping GLM's
  post-norm from the tail turns it red.
- **G-S2b, tiny towers: PASS**, worst merged-token cosine 1.000000000 for both over four grid sets (three sizes and a
  two-image batch). **G-S2c: PASS**, every planted defect red: Qwen3.5 (scale 0.975, RoPE halves 0.997, transposed
  position grid 0.995, patch bias 0.994), GLM-OCR (scale 0.126, RoPE 0.698, q/k norm 0.119, patch bias 0.531). The GLM
  gate uses aikit's tiny GLM tower, copied into goinfer's testdata (goinfer's own `glm-ocr-tiny` has no vision weights).
- **G-S2b, real towers: PASS**, nine image/tower pairs through serve's preprocessing and caps: Qwen3.5-0.8B worst token
  0.999999212, GLM-OCR worst 0.999991618 (`formula.png`). Exploratory times: Metal 2-4x the CPU tower (GLM-OCR invoice
  12.3 s against 48.9 s; Qwen3.5 on the 896² image 1.2 s against 4.7 s). Raw: `s2-gs2b-real.log` on the branch.
- **G-S2d run 1 (serve binary at the branch's `fac4aa52` plus the serve wiring), table.png, 32 greedy tokens:**
  - **GLM-OCR: PASS**, byte-identical replies. Both arms decode on the CPU (Metal does not run GLM-OCR's decoder), so
    only the tower differed. Time to first token 16.5 s against 39.2 s (exploratory).
  - **Qwen3.5-0.8B: confounded, not graded.** Under `--backend metal` the 0.8B also decodes on Metal, so the two arms
    differed in the decoder as well as the tower. The replies agree for their first 15 tokens, then the Metal arm wrote
    "," where the CPU arm wrote "." — the kind of flip Metal-vs-CPU decode makes about 5% of the time (G3), which says
    nothing about the tower. The registered procedure did not hold the decoder fixed; GLM only did so by accident.
  - Raw: `docs/measurements/multimodal-support-2026-10/s2-gs2d-run1/`.
- **G-S2d amendment, after run 1, before the graded Qwen run:** both arms run `--backend metal`; the CPU-tower arm adds
  serve's new `-vision-device cpu` (a real flag: it keeps the tower on the CPU while the model keeps its backend), so only
  the tower differs. Pass rule unchanged.
- **G-S2d run 2, Qwen3.5-0.8B, both arms `--backend metal` (decode `metal-resident (int4)` in both), the CPU-tower arm
  with `-vision-device cpu`:** the replies differ.
  - Tower on Metal: "...region (thousands)**, which presents...". Tower on the CPU: "...region (in thousands)**.\n\nIt
    presents...". Every token before generated token 14 agrees, each at p ~ 1.000.
  - At token 14 both arms sit on a coin flip: the CPU-tower arm picks "in" at p 0.545 with "th" at 0.442; the Metal-tower
    arm picks "th" at 0.579 with "in" at 0.403. The CPU-tower arm's gap between its token and the Metal arm's is 0.21 in
    logits.
  - **Status: near-tie by every measure serve exposes; the registered form is not computed.** The rule compares that
    gap with 3% of the raw top-1 logit, which log-probabilities hide; it passes for any top-1 logit above 7. Accepting
    it on that basis, or building an in-process replay that reads the raw logit, is the owner's call.
  - **Owner decision 2026-10-07: accepted as a near-tie. G-S2d: PASS for both families.**
  - Raw: `docs/measurements/multimodal-support-2026-10/s2-gs2d-run2/` (serve logs, replies, top-3 logprobs per token).

### S3 — Gemma 3 and Qwen2.5-VL on Metal: run what exists, then put the towers on the GPU

- **First, run it.** Metal's `UploadKV` and `ForwardMRoPE` tests have never been run, and no image turn has ever run on
  a Mac (`docs/measurements/multimodal-audit-2026-10-02.md`).
  - Pull `gemma-3-4b-it` and `qwen2.5-vl-3b` (disk first).
  - Run those tests, and one served image turn per family against the CPU decoder.
- **Then the towers.** aikit ships `gpu/visionmetal` (SigLIP) and `gpu/qwenmetal`, tagged and unused by goinfer.
  - Wire them through `EnableResident` (M-15 of `docs/audit-metal-2026-09-12.md`).
  - Gate them like S2, and run a night crossover against the CPU tower.
- **Size:** S to run, M to wire.

**S3 progress and gates (2026-10-07, Mac):**
- **Run it, part 1: done on what is local.** Metal's `UploadKV` and `ForwardMRoPE` tests ran for the first time and pass
  (5/5 on `llama-tiny` and on the real Qwen2.5-Coder-1.5B). `TestVLImageTurn_metalResident` runs the first image turns on a
  Mac's GPU, from the tiny fixtures: Gemma 3 and Qwen2.5-VL decode `metal-resident` after their CPU prefill (8/8 and 6/8
  tokens agree with the CPU decoder; agreement is the served gate's question). The real checkpoints (`gemma-3-4b-it`,
  `qwen25vl-3b-instruct`) are downloading from Hugging Face into `~/models`; the archive has no Qwen2.5-VL, and nobara
  held its timing lock.
- **The wiring:** goinfer's `metal` module imports aikit's tagged `gpu/visionmetal` (SigLIP) and `gpu/qwenmetal`
  (Qwen2.5-VL); serve's `enableResidentTower` admits `metal`. SigLIP's Metal tower is int8 (it needs `LoadEncoder`
  quant=true, as on CUDA and WebGPU), so `towerInt8` treats Metal like them for that family; Qwen2.5-VL stays f32 (its
  export carries both). `-vision-device cpu` keeps either on the CPU, f32 by default.
- **Near-tie, defined for served comparisons (serve exposes log-probabilities, not logits; G-S2d):** at the first differing
  token, the reference arm's probability of the other arm's token is at least half its own top token's probability, read
  from serve's `top_logprobs`.
- **S3 gates, written 2026-10-07 before any S3 tower code or measurement:**
  - **G-S3a, device tower against aikit's CPU tower at the same quantization** (SigLIP int8 both, Qwen2.5-VL f32 both):
    every soft token at cosine >= 0.9999, on the tiny towers (`siglip-tiny`, `qwen25vl-tiny`) and on the real towers on the
    four images F2a uses; the encoder must report the resident attached. Ambiguous (parked): a worst token in
    0.999-0.9999.
  - **G-S3b, served, the tower isolated:** one image request per family through the Metal serve binary, both arms
    `--backend metal`; the reference arm adds `-vision-device cpu` (and `-vision-quant int8` for Gemma 3, matching the
    device tower). Identical reply, or a first divergence at a near-tie.
  - **G-S3c, served, the decoder (part 1's "against the CPU decoder"):** both arms with the tower on the CPU
    (`-vision-device cpu`): `--backend metal` against `--backend cpu`. Identical reply, or a first divergence at a near-tie.
  - **Speed (night):** tower time per image, Metal against CPU, and TTFT of the served turn. A record.
- **The wiring needed new aikit tags.** `gpu/visionmetal` and `gpu/qwenmetal` v0.1.0 date from 2026-08-12 and do not build
  against `gpu` v0.33.5; aikit's current source does. So the S3 tower work moved to the `s2-towers` branch (whose go.work
  uses the local aikit), with S2's release batch.
- **G-S3a, read 2026-10-07 11:30 PDT on the Mac: tiny PASS, real FAIL for SigLIP, real PASS for Qwen2.5-VL.**

  | tower | tiny | real, worst token over four images | time per image, Metal vs CPU (exploratory) |
  |---|---|---|---|
  | SigLIP (Gemma 3), `visionmetal`, int8 both | 1.000000000 | **0.180-0.461: wrong** | ~73 s vs ~17 s |
  | Qwen2.5-VL, `qwenmetal`, f32 both | 1.000000000 | 0.999995874 or better | 22-47 s vs 23-37 s |

  - The Metal SigLIP tower agrees exactly at the tiny fixture's size (hidden 32, 16 patches) and breaks at Gemma 3's
    (hidden 1152, 4,096 patches), so aikit's own tiny parity test could not see it. It is also four times slower than the
    CPU.
  - The Metal Qwen2.5-VL tower is correct but no faster than the CPU, and slower on the larger images.
  - **Neither is wired into serve:** the wiring commit is reverted on the branch; the test stays as the evidence
    (`metal/s3_towers_test.go` on `s2-towers`). Raw: `docs/measurements/multimodal-support-2026-10/s3-gs3a-real.log`.
  - **Proposed, for the owner:** build both towers on goinfer's own Metal tower base, as S2 did for Qwen3.5 and GLM-OCR
    (its kernels pass at real size and run 2-4x the CPU), from new aikit float32 exports added to the same unreleased
    batch; and report the `visionmetal` real-size defect to aikit. G-S3b waits for that decision.
- **G-S3c, read 2026-10-07 11:35 PDT on the Mac (serve binary from `s2-towers`, the tower on the CPU in every arm,
  table.png, 32 greedy tokens): Qwen2.5-VL FAIL; Gemma 3 not reached.**
  - **Qwen2.5-VL-3B, `metal-resident (int4)` against `cpu (int4)`: the replies differ from the first token.** Metal
    "Table 2. Quarterly unit sales by region (thousands)"; CPU "Quarterly unit sales by region (thousands)". At token 0
    the CPU arm picks "Quarter" at p 0.728 with "Table" at 0.208, under half its top: **not a near-tie, so G-S3c fails for
    Qwen2.5-VL.** The first token comes from the prefill, so the Metal image turn computes something the CPU does not
    on the real checkpoint (the tiny fixture's first token matched). To be traced.
  - **Gemma 3 4B: Metal declined the resident** (it needs 5.15 GB against a 4.20 GB budget on this 16 GB Mac), so both
    arms decoded on the CPU and G-S3c did not reach Metal. The two CPU arms still differ at token 7, not as a near-tie
    (the `--backend cpu` arm puts the other arm's token at p 0.284 against its top 0.687): a CPU decode after a declined
    Metal request is not the same computation as `--backend cpu` (its int4 layout differs). A separate finding, to be
    understood.
  - Raw: `docs/measurements/multimodal-support-2026-10/s3-gs3c/` (serve logs, replies, top-3 logprobs per token).
- **Owner decision 2026-10-07: the proposal is accepted.** Both towers get rebuilt on goinfer's Metal tower base, from new
  aikit float32 exports (SigLIP, Qwen2.5-VL) on the same unreleased `s2-tower-exports` branch; the `visionmetal` real-size
  defect is reported to aikit. G-S3a's bars apply unchanged to the rebuilt towers, then G-S3b.
- **G-S3a for the rebuilt towers, registered 2026-10-07 before their code runs:**
  - **Bars unchanged:** every soft token at cosine >= 0.9999 against aikit's CPU tower, on the tiny towers and on the four
    F2a images at real size; 0.999-0.9999 ambiguous (parked).
  - **One amendment: SigLIP is float32 on both sides,** not int8. The rebuilt tower runs float32 GEMMs, from aikit's new
    float32 export (`Encoder.Weights`, `FinishHidden`), so serve loads Gemma 3's tower float32 for it, as it already does
    on the CPU. Qwen2.5-VL stays float32 on both sides, from aikit's existing `GPUWeights` (its float32 form),
    `BuildWindowPlan` and `MergeHidden`.
  - **The tiny checks randomise every norm first** (the all-ones trap).
  - **Planted defects, each one alone, must turn the tiny check red:**
    1. the attention scale dropped (both towers);
    2. RoPE's row and column halves swapped (Qwen2.5-VL);
    3. every block attending its whole image instead of its windows (Qwen2.5-VL);
    4. the window reordering skipped, on the way in and on the way out (Qwen2.5-VL);
    5. the position table dropped (SigLIP);
    6. the patch-embed bias dropped (SigLIP).
  - A defect that stays green means the fixture is degenerate along that axis. That gets recorded, and the check is
    rerun on a fixture or grid that can see the defect, before any result is read.
  - **Then G-S3b, as registered.**
- **G-S3a for the rebuilt towers, read 2026-10-07 on the Mac (`s2-towers` at `77f20330`, aikit `s2-tower-exports` at
  `d7d5cf9`): PASS, tiny and real.**
  - **The code:** `metal/vl_towers.go`. Both towers run on the S2 base (`metal/grid_vision.go`, which gains tower kinds:
    no RoPE, windowed segments, a fixed position table, and a block split across command buffers when its segments would
    overflow the scalar arena). They register through aikit's own seams (`vision.RegisterResident`,
    `vision.RegisterQwenResident`); aikit's `gpu/visionmetal` and `gpu/qwenmetal` are no longer imported. aikit gains
    SigLIP's float32 export (`Encoder.Weights`, `FinishHidden`), its recomposition bit-exact on the tiny tower.
  - **Tiny: both towers exact (worst token 1.000000000).** The first run left five of the seven planted defects green,
    because the tiny fixtures are degenerate: every bias is zero, and Qwen2.5-VL's q/k/v and output projections are at
    init scale (0.02), so attention is nearly uniform and adds little to the residual. As registered, the fixture was
    sharpened before reading anything: random biases, q/k scaled by 6 (SigLIP) or 12 (Qwen2.5-VL), and v and the output
    projection scaled by 8 (Qwen2.5-VL), through the aliasing exports, which the CPU tower reads too. Then every defect
    is red:

    | defect | SigLIP | Qwen2.5-VL |
    |---|---|---|
    | (1) attention scale dropped | 0.998501 | 0.629849 |
    | (2) RoPE halves swapped | — | 0.750730 |
    | (3) whole-frame attention in the windowed blocks | — | 0.789726 |
    | (4) window reordering skipped | — | 0.726768 |
    | (5) position table dropped | 0.745353 | — |
    | (6) patch-embed bias dropped | 0.762383 | — |

  - **Real, the four F2a images, attached through `EnableResident` as serve attaches them:**

    | image | SigLIP worst token | Metal / CPU tower | Qwen2.5-VL grid | worst token | Metal / CPU tower |
    |---|---|---|---|---|---|
    | gemma3_preprocess_image.png | 0.999999928 | 9.2 / 26.1 s | 64x64 | 0.999996659 | 8.9 / 22.1 s |
    | qwen25vl_preprocess_image.png | 0.999999972 | 9.5 / 28.4 s | 4x6 | 0.999999999 | 1.8 / 0.9 s |
    | glm_ocr/formula.png | 0.999998073 | 9.6 / 25.8 s | 86x72 | 0.999999651 | 15.1 / 34.9 s |
    | glm_ocr/table.png | 0.999999857 | 9.6 / 26.6 s | 64x86 | 0.999999464 | 14.4 / 31.1 s |

    Times are exploratory: one reading each, by day, a compile running beside one of them. SigLIP's Metal tower is about
    2.8x the CPU tower here; aikit's was wrong and ~73 s. Qwen2.5-VL's is 2.1-2.5x on the large images, and slower on
    the tiny 4x6 one, where per-window dispatch overhead dominates. Raw:
    `docs/measurements/multimodal-support-2026-10/s3-gs3a-rebuilt-real.log`.
- **G-S3b, read 2026-10-07 12:00-12:04 PDT on the Mac** (serve binary from `s2-towers` at `0c66b18b`, table.png, 32 greedy
  tokens, both arms `--backend metal`, the tower on Metal against `-vision-device cpu`; `run-gs3c-served.sh` with arms
  `metal:auto,metal:cpu`): **Qwen2.5-VL PASS; Gemma 3 FAIL as registered, its control owed.**
  - **Qwen2.5-VL-3B: byte-identical replies** ("Table 2. Quarterly unit sales by region (thousands)"). Decoder
    `metal-resident (int4)` in both arms, the tower on Metal in one. Request time 38.1 s against 53.1 s (exploratory).
  - **Gemma 3 4B: the replies differ at generated token 10,** " presented" against " broken". The Metal-tower arm puts
    " broken" at p 0.195 against its top 0.795: not a near-tie. Both arms decoded on the CPU (Metal declined the decoder
    for memory again, 5.15 GB against a 4.3-4.9 GB budget) on the same load, so only the tower differed, and the tower
    matched the CPU tower to a worst token of 0.999999857 on this very image (G-S3a).
  - **That size of flip from that size of perturbation is surprising, so nothing is concluded yet.** The determinism
    control (the CPU-tower arm twice; the Metal-tower arm twice) was attempted at 12:04 and 12:05. Both times serve's
    fit guard refused to load the 4B: 7.6-7.9 GB was available by then, against the 5.9 GB it needs at a 70% margin. It
    is owed on a quieter machine. If the CPU-tower arm repeats itself byte for byte, the next step is the perturbation
    control: CPU-tower features plus random noise of the Metal tower's size. Shipped path is not ground truth (R2): an
    int4 W4A8 decoder quantizes activations per tensor, so a 1e-7 perturbation can cross an int8 rounding and change a
    choice the reference itself makes with p 0.8.
  - Raw: `docs/measurements/multimodal-support-2026-10/s3-gs3b/`.
- **G-S3c's failure, analysed 2026-10-07 (no new measurement): the two arms run different CPU kernels, so the gate does
  not isolate Metal.**
  - Metal has no resident m-RoPE prefill (only CUDA implements `PrefillMRoPELast`). So a Qwen2.5-VL image turn under
    `--backend metal` prefills on the CPU and `UploadKV`s, and **its first token comes from the CPU prefill**, as the
    `--backend cpu` arm's does.
  - The two CPU prefills still differ: the load keys the int4 layout on `Options.Backend` (`wantsCanonicalInt4`,
    `wantsRow4Fallback` in `decoder/weightmat.go`). Under `metal` the CPU holds canonical int4 and no row4; under `cpu`
    it holds the arm64 row4 repack only, for embed, gate/up and qkv (the serve logs' `int4 layout` line). Different
    kernels, different rounding.
  - With both arms' towers on the CPU in float32 (the same features), the token-0 split ("Table" against "Quarter")
    can only come from that layout difference, not from Metal. Gemma 3's G-S3c is the same thing measured directly:
    both arms decoded wholly on the CPU (Metal declined) and still differed beyond a near-tie.
  - **Proposed, for the owner:** re-register G-S3c so the reference shares Metal's CPU-side layout. For example, the
    `--backend metal` load with its resident detached, through a test hook, against the same load decoding on Metal.
    Then the gate grades Metal's decode and nothing else. Until then G-S3c stands as FAIL, as registered.
- **G-S3c on CUDA, a cross-check, registered 2026-10-07 before it runs (nobara):** the same two requests, rule and near-tie
  definition as G-S3c, one serve binary built from `s2-towers` with `-tags cuda`, both arms `-vision-device cpu`:
  `--backend cuda` against `--backend cpu`, plus a second `--backend cpu` run as a determinism control (its reply must be
  byte-identical to the first). It answers whether Qwen2.5-VL's first-token divergence is Metal's alone: CUDA runs its own
  resident m-RoPE prefill, while Metal runs the CPU prefill and then `UploadKV`. It does not stand in for Metal's G-S3c.
- **G-S3c on CUDA, read 2026-10-07 on nobara (RTX 2070 SUPER, driver 595.91.07; `cuda/cmd/serve` built from `s2-towers` @ `29dece6c`, `-tags cuda`; int4; greedy):**
  Qwen2.5-VL reads clean; Gemma 3 diverges at a late token, outside the near-tie definition. Raw: `docs/measurements/multimodal-support-2026-10/s3-gs3c-cuda/`.
  - **Qwen2.5-VL-3B (arms `cpu`, `cuda`, `cpu`, tower on the CPU in all three): all three replies are identical** ("Quarterly unit sales by region
    (thousands)"; the CUDA arm against the CPU arm, and the second CPU run against the first). Decode paths as asked: `cpu (int4)` (canonical layout),
    `cuda-resident (int4)` with CUDA's batched resident prefill, `cpu (int4)`. **CUDA does not reproduce Metal's first-token divergence** ("Table" against
    "Quarter"), so Metal's `UploadKV` prefill-then-upload route is the suspect, not the image prompt or the m-RoPE positions as the resident paths see them.
    (Whole-request wall, exploratory: CPU 55.6 s, CUDA 22.2 s.)
  - **Gemma 3 4B, first run: void as a CUDA comparison, for two reasons the log shows.** (1) The `cuda` arm's resident build failed on a device allocation
    (`requested cuda -> running on cpu`; 4 KV slots planned at a 4141-position context, then a 17 MB scratch buffer did not fit), so that arm decoded on the
    CPU; (2) **`-vision-device cpu` was not honored for Gemma 3's SigLIP tower**: the generic branch of `loadVisionTower` keyed the tower's int8 default and its
    resident attach on the model's own backend, so the `cuda` arm ran an int8 resident tower while the CPU arms ran f32 on the CPU. That run's difference
    (token 3, ' quarterly' p 0.777 against ' a' 0.223) is therefore confounded twice. It is, though, the same shape as the Mac's separate observation that a
    CPU decode after a declined GPU request differs from `--backend cpu` (the declined arm's log line has no `int4 layout: canonical` marker).
  - **Gemma 3 4B, second run (valid): `GS3C_EXTRA="--kv-sessions 1"`, the flag fixed on the branch** (`internal/serveapp/main.go`: under `-vision-device cpu` the
    int8 default and the resident attach use the CPU, as `towerBackend()` says; until now this branch used `cfg.load.Backend`). All three arms load the same
    `encoder f32` tower on the CPU; decode paths `cpu (int4)`, **`cuda-resident (int4)`**, `cpu (int4)`.
    - **CUDA against CPU: first differing generated token 10, CPU ' region' 0.711, CUDA ' geographic' 0.287; near-tie (p(other) >= half p(top) = 0.356): False.**
      Registered rule: not a near-tie. **The second CPU run is byte-identical to the first** (determinism control holds).
    - **How to read it:** the divergence is at the 10th generated token, not the first, with the alternative at p 0.29. That is the size of an ordinary int4 W4A8
      (CUDA) against int4 (CPU) disagreement; the Mac's G3 calibration found a validated Metal model at 94% teacher-forced agreement, and E2B on CUDA reads
      94.5%. It is NOT shown to be a defect, and the near-tie definition here is stricter than G3's logit-gap rule. A text-only control (the same prompt
      without the image, CUDA against CPU) or a teacher-forced run over the image turn would separate "decoder noise" from "image path"; not run.
  - **Two findings for the Mac and the branch:** (a) the `-vision-device cpu` fix above is on `s2-towers` (not pushed); (b) Gemma 3 4B with four KV slots does not
    fit the 8 GB card at serve's default and falls back to the CPU (it fits with `--kv-sessions 1`); pre-existing, not touched here.
  - **What changed in the script:** `run-gs3c-served.sh` takes `GS3C_EXTRA` (extra serve flags for every arm), backward compatible.

### S4 — CUDA towers for Gemma 4 and Qwen2.5-VL

- **Gemma 4:** port `metal/gemma4_vision.go` to `cuda/`. The registration seam is already backend-neutral
  (`multimodal.RegisterGemma4Tower`).
- **Qwen2.5-VL:** wire aikit's tagged `gpu/qwencuda` (a require plus `EnableResident`).
- **Gates:** as S2, on nobara.
- **Size:** M (Gemma 4), S-M (Qwen2.5-VL).

**S4 and S2's CUDA twins, combined (2026-10-07):** CUDA has no tower base yet (Gemma 3's SigLIP is goinfer's own,
`cuda/vision_encoder.go`, built on aikit's CUDA ViT ops; nothing else), so the Gemma 4 port is the base, and S2's Qwen3.5+
and GLM-OCR towers follow on it, as Metal's did. One brief: `docs/prompts/nobara-s2-s4-cuda-towers-2026-10-07.md`.
- **G-S4q, aikit's `gpu/qwencuda` at real size, registered 2026-10-07 before it runs:** G-S3a's test shape and bars on
  CUDA (f32 both, every soft token at cosine >= 0.9999 on the four F2a images; 0.999-0.9999 parked), times exploratory.
  It decides S4's Qwen2.5-VL route: correct and faster than the CPU tower → wire it, as written above; wrong, or not
  faster → build it on the CUDA base, as Metal now will. The Metal twin was correct and no faster.
  - **G-S4q read 2026-10-07 on nobara (RTX 2070 SUPER; `s2-towers` @ `29dece6c`, aikit `s2-tower-exports`; Qwen2.5-VL f32 both sides): aikit's `gpu/qwencuda`
    is CORRECT at real size and faster than the CPU tower. Verdict by the registered rule: wire it, as S4 says.** Raw and the test sources (kept as `.go.txt`
    so `go build ./...` does not compile them from `docs/`): `docs/measurements/multimodal-support-2026-10/s4-gs4q/`.
    - **Correctness:** the tiny towers at 1.000000000 (both grids). The real Qwen2.5-VL-3B on the four F2a images, every soft token against aikit's CPU tower:
      worst token cosine **0.999997994, 0.999999994, 0.999999901, 0.999999942**: all over the 0.9999 bar.
    - **Time, exploratory (the first run, the Qwen subtest alone on the box):** tower on CUDA against the CPU tower, by grid: [1 64 64] 7.97 s against 15.17 s;
      [1 4 6] 51 ms against 140 ms; [1 86 72] 15.29 s against 24.40 s; [1 64 86] 12.70 s against 21.28 s, so 1.6x to 2.7x. (The second run's times, 16.9 / 0.118 / 32.4 /
      26.9 s against 28.7 / 0.142 / 43.2 / 41.0 s, were taken while the SigLIP package ran in parallel on the same GPU and CPU and are not to be used.) 8-15 s for a
      large grid is faster than the CPU but not fast.
    - **aikit's `gpu/visioncuda` (SigLIP), the information item: WRONG at real size, like its Metal twin.** On the four images, worst token cosine **0.245, 0.269,
      0.097, 0.281**, with 4094 to 4096 of the 4096 tokens under 0.9999 (its tiny parity test passes). Times were taken under the parallel-run contention above, so
      they say nothing about speed (17-34 s on the CUDA tower against 21-37 s on the CPU's). For the aikit defect report together with the Metal reading.
    - **A trap that nearly made this reading wrong:** aikit's `visioncuda` and goinfer's own `cuda.VisionEncoder` (`cuda/vision_register.go`, Gemma 3's CUDA tower)
      both register through the one global `vision.RegisterResident`, and the LAST registration wins. Inside package `cuda`, goinfer's registers last, so a SigLIP
      test there measures goinfer's tower: `*cuda.VisionEncoder` was the attached type, with per-token cosines mostly 0.99+ against the CPU int8 tower but 4065 of 4096
      tokens under 0.999 and outliers as low as 0.18 on high-norm tokens (rows of norm 16 against 2-3). The first run's "siglip" lines are that tower, and the test
      now asserts the attached type. aikit's tower is read in its own package (`cuda/s4siglip`), which imports nothing from goinfer's `cuda`. goinfer's own tower is
      not gated here and the numbers above are not a verdict on it; the served Gemma 3 image replies with it were sensible.
  - **Wired (2026-10-07, on `s2-towers`, commit `e5e5dbdf`; not on `main`):** `cuda/vision_towers.go` imports `gpu/qwencuda` (and deliberately not `visioncuda`);
    serve's `qwenTowerPlacement` attaches it under `--backend cuda` for a float32 tower, names every CPU fallback (`-vision-quant int8`, "cuda declined"), and
    `-require-backend` refuses them; `-vision-device cpu` keeps the tower on the CPU. `TestQwenTowerPlacement` covers the ten cases.
  - **Served on Qwen2.5-VL-3B, the G-S2d shape (`--backend cuda`, tower on CUDA against `-vision-device cpu`, a second CPU-tower run as the control):** identical
    replies in all three ("Quarterly unit sales by region (thousands)"). The log lines say what each arm did: tower `(CPU; ...)`, `(CUDA; ...)`, `(CPU; ...)`,
    decode `cuda-resident (int4)` throughout. Whole request, exploratory: 22.2 s with the CPU tower, 13.5 s with the CUDA tower.
    Raw: `docs/measurements/multimodal-support-2026-10/s4-served-qwen/`.
  - **A limit the first served run found, and it is the card's:** with serve's default KV plan the resident decoder takes all the free VRAM, so the tower's
    19.6 MB upload failed (`CUDA_ERROR_OUT_OF_MEMORY`) and the tower ran on the CPU: the log said `CPU (cuda declined)` and the reply was still identical, so a
    run on the default plan reads as a pass for the wrong reason. The comparison above needed `--kv-sessions 1 -ctx 4096`. The resident KV planner reserves
    for a drafter (`extraBytes`) but not for a vision tower, so on an 8 GB card a device tower and a default-planned decoder do not coexist. That is a planner
    gap, not a tower defect, and it applies to goinfer's own Gemma 3 SigLIP tower too; not fixed here.
  - **The script:** `run-gs3c-served.sh` sets `-vision-device` per arm as `<backend>:<auto|cpu>` (plain `<backend>` means the CPU tower); the Mac added the same thing
    at the same time, and the merge kept its spelling. These runs were made with an earlier `+` spelling of it (labels `cuda_cpu`, `cuda_auto`), which is equivalent.
- **S4 Gate 0 for the CUDA towers, read 2026-10-07 (nobara), before any CUDA tower code:** the desk map is
  `docs/measurements/multimodal-support-2026-10/s4-gate0-cuda-towers.md`. In short:
  - **Reuse is large.** aikit's `gpu.ViT` already has the f32 GEMMs with bias and bias-plus-residual epilogues, LayerNorm, RMSNorm, GELU tanh and erf, SiLU-mul,
    a NeoX RoPE and scaled, per-segment attention kernels; aikit's `qwencuda` is a correct template for the Qwen3.5 and GLM-OCR block loop. New small CUDA kernels
    are needed only for Gemma 4 (axial RoPE, ClippableLinear clamps, gelu·mul, position add). No audited PTX is touched; the new kernels are a new `.cu`.
  - **Not portable:** Metal's scalar-arena launch trick (CUDA takes scalars as launch arguments), so the base brings its own launch helper with an error latch.
  - **The tiny fixtures do not reach the production GEMM branches** (real GLM-OCR's patch embed has K = 1176, not a multiple of 16; real np is rarely a multiple of 64),
    which is aikit `visioncuda`'s failure class. The real-size gates are therefore mandatory, and per-branch GEMM tests at real shapes are added.
  - **VRAM:** all three towers fit in f32 beside a small decoder at modest sizes (Gemma 4 E2B ~0.7 GB, Qwen3.5-0.8B ~0.35 GB, GLM-OCR ~1.6 GB weights, plus scratch that
    scales with the image: ~2 GB for GLM-OCR at its 4.8 MP ceiling). The planner's drafter reserve (`ExtraResidentBytes`) can carry a tower estimate with no change in `cuda/`.
  - **CUDA-specific planted defects, registered now for G-S2c (they are in addition to Metal's, which port directly):** (a) the bias dropped only in the tiled-fallback GEMM
    branch (K not a multiple of 16), seen only by a real-shape GEMM test or a real tower; (b) padded attention keys not masked (if the matmul-blocked attention is used);
    (c) all patches of a two-image batch in one attention segment; (d) stale scratch across sizes (large, small, large grid on one accelerator); (e) for Gemma 4, the
    position x and y swapped, the value norm weighted instead of unweighted, and the attention scale changed from 1.0. Each must turn its gate red; a defect that stays
    green means the fixture is degenerate along that axis (fix the fixture, not the bar). Controls that are not defects: two identical runs bit-identical, a leak test after
    close, and a factory that errors (never panics) under allocation failure, with serve naming the CPU fallback and `-require-backend` refusing it.
  - **Build order:** serve plumbing made backend-generic; the tower base; the Gemma 4 tower; the Qwen3.5 and GLM-OCR towers; the VRAM reserve; the speed record (queued, not
    started); and only if that says so, a faster attention.
- **S4 steps 0-2 read 2026-10-07 on nobara (`s2-towers`; not on `main`): the tower base and the Gemma 4 tower on CUDA PASS.** Raw: `docs/measurements/multimodal-support-2026-10/s4-gemma4-cuda/`.
  - **Step 0, serve:** `planGridTower` and `chooseGemma4Tower` are backend-generic (Metal and CUDA, each checked against the registry for its own backend, the fallback named with
    that backend's word); the table row that pinned CUDA to the CPU is replaced by CUDA cases, and `chooseGemma4Tower` has a placement test for the first time.
  - **Step 1, the base** (`cuda/tower_base.cu` and `.go`; six small kernels in their own module, NVRTC 12.9.86, FMA-linted; an executor goroutine that owns the device; a launch
    helper with an error latch; GEMM, bias and bias+residual plan selection that handles a K that is not a multiple of 16): every GEMM branch against float64 (the register kernel on
    aligned shapes, the bias kernel at any M, the tiled kernel plus `add_bias` at K = 1176 and K = 20), the norms, the six kernels **bit for bit** against Go's float32 arithmetic,
    and attention at scale 1.0 and 1/sqrt(hd). The tiled-branch bias test was shown red by a mutant (bias dropped on that branch: the output was wrong by 60% of its scale).
    aikit's own attention kernel is used as it is (np <= 12288), so Metal's matmul-blocked attention was not needed for a correct baseline.
  - **Step 2, the Gemma 4 tower** (`cuda/gemma4_vision.go`, registered as multimodal's "cuda" tower), tiny tower with finite clip bounds and random position tables and norms:
    worst soft-token cosine **1.000000000** on all four grids (18, 54 and 135 patches; raster and shuffled). **Planted defects, each alone red** (worst cosine): clamps skipped 0.938,
    position x/y swapped 0.147, value norm weighted 0.829, attention scale 1/sqrt(hd) 0.533, position tables skipped -0.529, RoPE skipped 0.292. Large-small-large on one tower and
    two identical runs: bit-identical; a short patch buffer and an out-of-table position return errors; after `Close` the device ledger is empty.
  - **Real E2B (F2a on CUDA), every soft token against aikit's CPU tower, the four images (2304-2430 patches): worst cosine 1.000000000 on all four.** Tower CUDA 2.15-2.38 s against the
    CPU's 4.19-4.50 s (exploratory; the CUDA figure includes aikit's host tail).
  - **Served (the G-S2d shape; `--backend cuda`, E2B decoding resident, `--kv-sessions 1 -ctx 4096`):** tower on CUDA, tower on the CPU, and a second CPU-tower run give an identical
    32-token reply ("This image shows **Table 2: Quarterly unit sales by region (in thousands)** for the fiscal year FY2025." and the start of the next sentence). The logs name each arm
    (`tower on CPU` / `tower on CUDA`, `decode path: cuda-resident (int4)`, `decoded 32 tokens on the resident path`); the image encode took 2.34 s with the tower on CUDA and 4.45-4.47 s
    on the CPU. The default KV plan was not used, for the reason recorded under G-S4q (the tower needs VRAM the resident decoder would take); step 4 is the reserve for that.
- **S4 step 3 read 2026-10-07 on nobara (`s2-towers`, commit `d7955399`; not on `main`): the Qwen3.5+ and GLM-OCR towers on CUDA PASS G-S2b, G-S2c and G-S2d.** Raw:
  `docs/measurements/multimodal-support-2026-10/s4-grid-cuda/` (`gates-tiny.log`, `real.log`, `served/`).
  - **The build:** `cuda/grid_vision.go`, registered as multimodal's "cuda" Qwen3 and GLM-OCR towers, on the CUDA tower base, with aikit's own scaled attention called once per image frame
    (the frame is the segment) and separate q, k and v (GLM-OCR norms q and k per head before RoPE, which aikit's fused-qkv kernels cannot do), plus one new kernel, `tower_rope_half`
    (NeoX rotate-half on a separate buffer, bit for bit against Go). The base's attention cap of 12288 patches now applies only where the query-tiled kernel does not.
  - **G-S2b, tiny towers (norms randomised through the export's aliasing slices), four grid sets including a two-image batch:** Qwen3.5 and GLM-OCR both worst merged-token cosine
    **1.000000000**.
  - **G-S2c, planted defects, each alone red.** Metal's five reproduce Metal's recorded readings to the digit (Qwen3.5: scale 0.974966, RoPE 0.996732, position grid 0.994887, patch bias
    0.993714; GLM-OCR: scale 0.126404, RoPE 0.698253, q/k norm 0.118984, patch bias 0.531329), which is the strongest cross-check available between the two backends. CUDA's new one,
    **every frame of a batch attending across the boundary**, reads 0.986906 (Qwen3.5) and 0.457776 (GLM-OCR). The other CUDA-specific entries of Gate 0 are covered elsewhere: the bias dropped in
    the tiled GEMM branch by the base's GEMM-branch test (shown red by a mutant; these tiny towers' patch dims are multiples of 16 and never take that branch, as predicted), stale scratch
    across sizes and determinism and the ledger after `Close` by `TestGridVisionCUDA_scratchDeterminismClose`.
  - **G-S2b, real towers (`~/models/qwen3.5-0.8b` at serve's 1024-token cap; `~/models/glm-ocr` at its default 4.8 MP ceiling), nine images, every merged token against aikit's CPU Forward:**
    Qwen3.5 worst cosine **0.999999763, 0.999999999, 0.999999964, 0.999999989**; GLM-OCR **0.999999175, 0.999999999, 0.999996380, 0.999999195, 0.999999049** (the images in order: the Gemma 3 preprocess image, the Qwen2.5-VL one, then
    GLM-OCR's formula and table, and for GLM-OCR also its invoice: Qwen3.5 has four, GLM-OCR five). All over 0.9999.
  - **G-S2d, served (`--backend cuda`, decode `cuda-resident (int4)`, tower on CUDA against `-vision-device cpu`, a second CPU-tower run as the control, `--kv-sessions 1 -ctx 4096`):**
    the CPU-tower control is byte-identical to the first CPU-tower run for both families. Tower on CUDA against the CPU tower: the replies differ, **at near-ties under the registered rule**:
    Qwen3.5 first differs at generated token 3 (CPU-tower ' **' 0.546, CUDA-tower ' a' 0.314; p(other) >= half p(top) = 0.273: True), GLM-OCR at token 1 (' table' 0.561, ' image' 0.439;
    0.280: True). The towers' features differ by about 1e-6 (cosine 0.999999+), so a near-tie flipping is expected, and the rule exists for it. The logs name each arm (`tower (CPU)` /
    `tower (CUDA)`); no `declined` line appears, so the CUDA arms did run the tower on CUDA. Image encode, exploratory: Qwen3.5 5.17 s (CUDA) against 6.05 s (CPU); GLM-OCR 26.0 s (CUDA).
  - **Speed, exploratory and honest:** the CUDA towers are only **1.2-1.5x** the CPU towers (real Qwen3.5 3.2-5.2 s against 4.0-6.2 s; GLM-OCR at 1-1.6 K merged tokens 14-36 s against 20-43 s; the figures
    include aikit's host tail). That is a correct baseline and not a fast tower: the f32 GEMMs and aikit's scalar query-tiled attention are the cost, and Gemma 4's tower (2.2 s against 4.4 s)
    is better only because it has fewer patches. A faster attention and GEMM are Gate 0's step 6, to be decided by the speed record (queued separately), not claimed here.
- **S4 step 4 read 2026-10-07 (`s2-towers`, commit `5970cfa4`; not on `main`): the VRAM reserve and a graceful fallback.** Raw: `docs/measurements/multimodal-support-2026-10/s4-grid-cuda/default-plan/`
  (`s4-reserve-g4`, `s4-reserve-grid`, `s4-reserve-q25-i`).
  - **The reserve:** serve prices a CUDA tower into `Options.ExtraResidentBytes` (the field the resident plan already reads for a drafter) from the checkpoint's `vision_config`: float32
    weights plus peak scratch at the family's largest image, 0.86 GB for Gemma 4 E2B, 0.67 GB for Qwen3.5-0.8B, 2.4 GB for GLM-OCR (scratch priced at about 1.5 MP, not its 4.8 MP ceiling,
    whose 4.15 GB took GLM-OCR's CUDA context from 16384 to 4931 positions), and about 3.4 GB for Qwen2.5-VL (8192 patches). Zero unless the tower will really run on CUDA. It is an
    estimate and a ceiling the KV context gives way to: on the default plan the E2B kept 16384 positions and 4 conversations, GLM-OCR's context became 5046 (4931 after the trim), Qwen2.5-VL's 6160.
  - **What it fixed:** on the default plan the Qwen2.5-VL tower used to be declined at attach (`CUDA_ERROR_OUT_OF_MEMORY`) and run on the CPU; with the reserve it attaches on CUDA.
  - **What it did not fix, found by running it:** a 5504-patch image still missed the budget on the 3B Qwen2.5-VL (2.5 GB of tower weights beside a 3B decoder on an 8 GB card), and **aikit's
    `gpu/qwencuda` panics on that allocation failure instead of returning it, which killed the whole server** (an unrecovered panic on the request goroutine). Two guards now stand in front of it:
    `recoverDeviceTower` turns a device-tower panic into an error that names the cause and the way out, and `deviceFallback` (and, for Qwen2.5-VL, `qwenForwardWithFallback`) closes a device
    tower that ran out of memory so its VRAM comes back, and runs that image and every later one on the CPU tower. On the default plan the same request now returns the CPU tower's identical reply
    (22.3 s encode, the CPU's time) with the log line `the Qwen2.5-VL tower ran out of device memory ...; it runs on the CPU from now on`. For the aikit defect report: `qwencuda` should return
    allocation failures from `ForwardViT`, not panic.
  - **Default-plan served runs, tower on CUDA against `-vision-device cpu`:** Gemma 4 E2B identical (tower encode 2.3 s against 4.5 s); Qwen3.5-0.8B and GLM-OCR differ at near-ties as before
    (token 3, p(other) >= half p(top): True; token 1, same), the towers on CUDA, no fallback needed; Qwen2.5-VL identical via the fallback above.
  - **A test-harness trap found on the way:** `run-gs3c-served.sh` left its server running on the GPU when a request failed (`set -e`), so the next run tested the stray server and held 7 GB;
    it now stops the arm's server on any failure. The three strays were found by name (`pgrep -x`) and stopped.
- **S4 step 5, the speed record, is queued for tonight as `s4-tower-speed`** (`run-s4-tower-speed.sh`: Gemma 4 E2B, Qwen3.5, GLM-OCR and aikit's Qwen2.5-VL tower against the CPU towers, three passes
  under the timing lock, a pre-built test binary at `5970cfa4`, estimate 30 min). One pass was smoke-tested by day (6 min 44 s; exploratory, not a result). Nothing is quoted from it yet.
- **S4 status, 2026-10-07:** steps 0-5 done (step 6, a faster attention and GEMM, waits for the record). Everything is committed on `s2-towers` in `~/wt/goinfer-s2` and **not pushed** (the branch is pushed
  only on the owner's word, and it needs aikit's unreleased `s2-tower-exports`); the Mac's commits on the same branch were merged in. `main` has only the records.
- **The CUDA towers' gates are G-S2b, G-S2c and G-S2d's, unchanged, on CUDA;** the Gemma 4 CUDA tower's are Metal's
  Gemma 4 tower gates (`metal/gemma4_vision_test.go`: every soft token at cosine >= 0.9999 against aikit's CPU Forward,
  the shuffled and clamp controls) plus a served Gemma 4 image turn as in G-S2d. nobara writes its desk map and any
  CUDA-specific planted defects into this doc before the CUDA tower code.

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
- **Gemma 4 E4B and 31B:** download and validate (owner, 2026-10-07). E4B through S1's E-model gates on both GPU
  backends; the 31B through the 26B's (bidirectional image prefill, the S1.0 re-check shape). Disk first on both boxes.
- **EmbeddingGemma 2 on CUDA:** text, image and audio. Optional. The Metal kernels show the shapes; the work is a CUDA
  twin.
- **WebGPU:** moved to its own phase, S12 (owner: invest, 2026-10-07).

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

### S9 — Batched E-model prefill on Metal and CUDA, image turns included

Added 2026-10-07. Once S1 put E2B/E4B decode on the GPU, prefill became the cost: an E2B image turn on the Mac is about
15 s for 32 tokens (G4), most of it the ~286 image and prompt tokens prefilling one at a time on the CPU, and a text
prompt prefills token by token on the GPU too. Both backends decline E-models from every batched prefill path by name
(S1's explicit declines), and only CUDA's bidirectional 26B/31B class runs its image prefill on the GPU.

- **What it needs:** the batched prefill paths learn the E-model shape S1 taught the decode step: the per-row PLE tail
  (each row's `[h || L*P]`, image rows with PAD's token-identity term), KV-shared layers that store nothing and read
  their source's rows, per-layer FFN widths. Then the image turn's prefill runs resident instead of on the CPU plus an
  `UploadKV`.
- **Gates to write before starting:** batched against sequential resident prefill on the tiny E-model, every position
  (bit-identical where the kernels allow, else the S1 near-tie rule); S1's planted defects through the batched path; G4's
  served image request with the prefill resident, reply identical to the CPU's.
- **Speed (night):** image-turn TTFT and text-prompt TTFT on E2B, batched against sequential, on both boxes.
- **Size:** M-L per backend.

### S10 — Towers for the families that have none

Added 2026-10-07: Ministral 3 (Pixtral), LFM2.5-VL, North, and Qwen3-VL's image path (today text only).

- **Pattern:** the Gemma 4 and Qwen ones: the tower in aikit (batched into the cycle's one release), the decoder splice
  and prompt layout in goinfer, then serve. One sub-phase per family, in order of what users ask for; each starts with
  a desk map of its HF processor (image budget, special tokens, position scheme).
- **Gates, per family, before starting it:** the tower's stages against HF on the real checkpoint; full-model logits on an
  image prompt against HF; a served request.
- **Size:** M each.

### S11 — Several images per message

Added 2026-10-07. Today a second image in one message is a 400.

- **What it needs:** prompt layout with several image blocks per family, the resident image-block bookkeeping
  (`residentImageBlock`, prefix reuse) for more than one block, and serve's request parsing.
- **Gates:** two-image prompts against HF on a tiny fixture per family; served two-image requests; prefix reuse refusing a
  changed second image.
- **Size:** S-M.

### S12 — WebGPU for multimodal (owner: invest, 2026-10-07)

WebGPU today: Gemma 3 GPU/GPU, Qwen2.5-VL decode on the GPU, GLM-OCR staged, Gemma 4 and Qwen3.5 on the CPU, and no
release binary (it needs cgo).

- **What it needs:** Gemma 4 resident decode on WebGPU (its Gemma kernels: sandwich norms, the two geometries, K=V, the
  S1.0 fixes it already has), then the E-model shape (S1's three parts), then the towers that S2/S4 put on Metal and CUDA,
  and the release-binary question (a cgo build per platform, or a documented build-it-yourself).
- **Gates:** the S1/S2 gates, run on WebGPU; written per sub-phase before it starts.
- **Size:** L.

### S13 — Against the peers

Added 2026-10-07. S7 grades each cell against a TTFT bar, not against another engine.

- **What:** one night per box, `bench_peer.py`, the S7 cells against llama.cpp (and MLX on the Mac) on the same image
  and prompt, same-session interleaved; reported, not gated, unless a bar is registered first.
- **Size:** S-M (mostly harness: the vision peer path exists for SigLIP only).

### S14 — Speech beyond Gemma 4 (last, after S12)

The owner put this last, after the WebGPU work. It starts with the choice P11 left open
(`docs/measurements/multimodal-finish-2026-10-06/p11-audio-comparison.md`): a Whisper-style front end plus Qwen3-ASR; a
Whisper encoder (which serves Voxtral), toward pure-Go Whisper; or Voxtral Realtime. Gates are written once the family
is chosen.

## Order of work (owner, 2026-10-07)

S2, S3, S4 (towers), S5 (Gemma 4 audio), S9 (batched E-model prefill), S6 (coverage, with E4B and 31B validated), S10
(new towers), S11 (several images), S12 (WebGPU), S7 (measure every cell), S13 (peers), S8 (the README table), S14
(speech). S8's drift check keeps the README true through S14, which updates the table as part of its own work.

## Decisions for the owner

**Decided 2026-10-07:**
- **Video:** deferred; no phase.
- **Gemma 4 E4B and 31B:** download and validate (S6), not a narrowed claim.
- **WebGPU:** invest in it for multimodal (S12).
- **Speech:** last (S14), after the WebGPU work; the family is chosen when it starts.

**Still open:**

- S14's speech family (P11's options; decided when S14 starts).
- Which VL checkpoint `pull` recommends per box class (P9(d)).
- The S7 speed bar.
