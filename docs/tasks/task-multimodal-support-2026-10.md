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
| Gemma 4 E2B, E4B | CPU / CPU | CPU / CPU | GPU / CPU | CPU / CPU |
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
- **G4, real E2B image chat:**
  - The F2b request through the Metal serve binary, resident decode against CPU decode.
  - **PASS:** identical reply, or a first divergence at a near-tie.
- **Speed (night):** E2B decode tokens per second, resident against CPU, same-session interleaved, plus the host's
  PLE milliseconds per token. A record; not a gate.

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
