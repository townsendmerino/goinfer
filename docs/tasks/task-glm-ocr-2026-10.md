# Task: GLM-OCR — documents in, text and schema-bound JSON out (O0–O7) — 2026-10

> **Status: O0, O1, O2, O3, O5 DONE (2026-10-01/02; goinfer commits local, nothing pushed).** Goinfer at f32 on the real checkpoint is
> TOKEN-IDENTICAL to transformers 5.12.0 for 64/64 tokens on three rendered documents (O3); schema-constrained extraction works end to end:
> `goinfer-chat --image invoice.png --schema invoice.schema.json` prints schema-valid JSON, and 15 rendered invoices read 383/397 fields
> (96.5%) correctly at int4 on CUDA (O5). CUDA runs the whole thing resident (pairwise rope kernels, `a306d33e`); Metal and WebGPU decline it and
> run it on the CPU. **aikit v1.52.0 is released (2026-10-02, with the GLM tower) and goinfer's five modules are pinned to it (`36aa0e73`), so root
> builds `GOWORK=off`.** **Open: O4 (the pixel-cap default, an owner decision), O6 (Metal) and O7.** What still blocks a push: the site module's two
> Ollama checks are red until O7's snapshot exists, and `origin/main` has moved (about 50 files, including `README.md`, `banner.go`, `assets.json`) so a merge
> is due. O0's findings are in
> [§O0 results](#o0-results-2026-10-01), and the claims they overturned are corrected in place below, each marked
> *Corrected 2026-10-01 (O0)*. The two that matter most: the pixel budget is half what §1 said (6,144 image tokens,
> not ~12,000), and the tower is nearer aikit's Qwen3.5+ encoder than Qwen2.5-VL's. Everything after O0 is written to
> be handed to a session as it stands, one gate at a time.
>
> **What this is.** `zai-org/GLM-OCR` is a 0.9B document-OCR model (MIT, March 2026, arXiv 2603.10910).
> It turns a page image into markdown, tables, formulas, or JSON under a schema the prompt supplies.
> It is #22 in Ollama's library at 7.7M pulls in about six months
> ([`../measurements/ollama-coverage-2026-09-30.md`](../measurements/ollama-coverage-2026-09-30.md), row 22,
> status N). llama.cpp, vLLM, SGLang, Ollama and `transformers` all run it, so prior art is plentiful.
>
> **Why it is worth a family slot.** Two reasons:
> - It is small enough for any machine goinfer targets.
> - It fits what goinfer already leads on. GLM-OCR's extraction mode asks for strict JSON, and goinfer's
>   grammar constraint guarantees it. "A scanned invoice in, a Go struct out, on a laptop, one binary" is a
>   demonstration few engines can make.
>
> **Why it is a medium task, not a large one.** Its language model is a plain dense transformer. That puts it
> on the generic forward path, which already carries the embed-by-vector seam, m-RoPE positions, image-block
> reuse, and the resident m-RoPE interface `GenerateQwenVL` uses. The new work is one decoder registration
> made mostly from existing parts, one vision encoder in aikit, preprocessing, and a pixel budget.
>
> **Siblings.** [`../multimodal.md`](../multimodal.md) (the VL design record; §P8's Qwen3.5+ work shares the
> image plumbing but not the tower, and neither blocks the other) ·
> [`task-constrained-confidence.md`](task-constrained-confidence.md) (per-field confidence applies to O5's
> extraction output unchanged) · [`task-web-ui-2026-09.md`](task-web-ui-2026-09.md) W11 (image attach,
> already shipped; O5's demo runs through it) · [`task-site-2026-09.md`](task-site-2026-09.md) (O7).

---

## 1. What the model is (read 2026-10-01 from the config and the HF source; O0 re-verifies)

**Lineage.** Every GLM-OCR class subclasses a GLM-4.1V (`glm4v`) class
(`transformers/models/glm_ocr/modular_glm_ocr.py`). The overrides that matter:
- `GlmOcrTextAttention`: q/k/v with `bias=False`;
- `GlmOcrVisionAttention`: adds `q_norm` / `k_norm` (RMSNorm);
- `GlmOcrVisionModel`: removes GLM-4V's learned position `embeddings` and its `post_conv_layernorm`;
- the pretrained model ignores `model.language_model.layers.16.*`, the multi-token-prediction layer
  (`num_nextn_predict_layers: 1`). Inference never uses it.

**Text decoder** (`text_config`, `model_type: glm_ocr_text`):
- 16 layers, hidden 1536, 16 query heads, 8 KV heads, head_dim 128, intermediate 4608, vocab 59,392;
- untied head, no attention bias;
- fused `gate_up_proj` with SiLU;
- four norms per layer, in order: `input_layernorm` → attention → `post_self_attn_layernorm` → residual
  add; `post_attention_layernorm` → MLP → `post_mlp_layernorm` → residual add;
- rotary theta 1e4, `partial_rotary_factor: 1.0`, `mrope_section: [16, 24, 24]`.

GLM-4V rotates **even/odd pairs** (`rotate_half_llm`: `x[..., 0::2]`, `x[..., 1::2]`). It builds the m-RoPE
frequencies by taking contiguous section blocks (`m[i % 3]` over `freq.split(mrope_section)`), then
`repeat_interleave(2)`. So the rotation is pairwise, and the t/h/w layout is contiguous blocks, not Qwen3-VL's
interleaved layout.

(The modular file's config class defaults `hidden_size` to 1024; the shipped `config.json` says 1536. The
checkpoint's file governs.)

> **Confirmed 2026-10-01 (O0)** against the generated `modeling_glm_ocr.py` and the checkpoint's tensors, with these
> specifics the build needs:
> - `head_dim` is 128 and the attention scale is `128^-0.5`; `q_proj` is [2048, 1536], `k_proj`/`v_proj` [1024, 1536],
>   `o_proj` [1536, 2048], none biased. (`hidden_size / num_heads` would be 96; do not derive `head_dim` that way.)
> - `gate_up_proj` is fused [9216, 1536] and splits **gate first, then up** (`chunk(2)`; `up * silu(gate)`).
> - **The four norm tensors are named differently from Gemma's.** GLM: `input_layernorm` (pre-attention),
>   **`post_self_attn_layernorm`** (on the attention output), **`post_attention_layernorm` (the PRE-MLP norm)**,
>   `post_mlp_layernorm` (on the MLP output). Gemma's `post_attention_layernorm` is the opposite: the norm on the
>   attention *output*. All four are [1536] and plain `weight * x` RMSNorm (eps 1e-5), so a copy of Gemma's name map loads
>   without error and computes something else. Pin the mapping with a test (see O1).
> - Rotation is full-width (`partial_rotary_factor` 1.0, 128 of 128 dims), **pairwise**, and the m-RoPE frequency
>   index `d` in [0, 64) takes its position from the contiguous block it falls in: 0–15 temporal, 16–39 height,
>   40–63 width (`chunk[i % 3]` over `split([16, 24, 24])`), then `repeat_interleave(2)` so pair `(2d, 2d+1)` shares
>   frequency `d`. Text tokens carry equal positions, so text rotation is plain pairwise RoPE.
> - Layer 16 is the MTP layer, a full block plus `eh_proj` [1536, 3072], `enorm`, `hnorm`, its own `embed_tokens` and a
>   `shared_head`: 218 M parameters the load must skip. HF's `_keys_to_ignore_on_load_unexpected` is
>   `model\.language_model\.layers\.16.*`.
> - `eos_token_id` is **[59246 `<|endoftext|>`, 59253 `<|user|>`]**; `generation_config` is greedy (`do_sample: false`).

> **Corrected 2026-10-01 (O0): size.** "0.9B" is the model card's name for it. The checkpoint's tensors total 1.325 B
> parameters (2.65 GB bf16): the tower 434 M, the decoder with its embedding 582 M, the untied `lm_head` 91 M, and the
> multi-token-prediction layer 218 M. **Loaded for inference without the MTP layer: 1.107 B, 2.21 GB at bf16.**
> Tensor names are `model.visual.*`, `model.language_model.*` and a top-level `lm_head.weight`.

**Vision tower** (`vision_config`, `model_type: glm_ocr_vision`):
- depth 24, hidden 1024, 16 heads, intermediate 4096, `silu` gated MLP;
- patch 14, temporal patch 2, spatial merge 2, `out_hidden_size` 1536;
- RMSNorm eps 1e-5, attention bias true, full attention in every block.

GLM-4V's forward order with GLM-OCR's removals applied: `patch_embed` (Conv3d) → blocks → `post_layernorm` →
`downsample` (Conv2d, kernel and stride = merge size) → `merger`. The merger runs `proj` →
`post_projection_norm` (LayerNorm) → GELU → `down_proj(silu(gate_proj) * up_proj)`.

> **Corrected 2026-10-01 (O0): what the tower is, against the checkpoint's own tensor shapes** (safetensors header,
> revision `2e85a628`). All of §1's tower facts hold, and these are added:
> - **A 2D rotary, which §1 omitted** (`vision_config.rope_parameters.rope_type: axial`, theta 1e4): rotary dim
>   `head_dim/2` = 32 over (row, col) patch positions, `cat(freqs, freqs)`, and an ordinary **half-split `rotate_half`**
>   (not the text decoder's pairwise one). Patches arrive in merge-block order, (block-row, block-col, merge-row,
>   merge-col), exactly as Qwen's tower expects.
> - **Block order:** `norm1` (RMSNorm) → attention with biased fused `qkv` [3072, 1024], **per-head `q_norm` / `k_norm`
>   (RMSNorm over head_dim 64) applied before the rotary**, biased `proj` → residual; `norm2` → gated MLP → residual.
>   The MLP is **biased and 4096 wide** (`gate_proj`/`up_proj` [4096, 1024], `down_proj` [1024, 4096], all biased).
>   Attention is full within an image, no windows.
> - **`patch_embed.proj` is a biased Conv3d** [1024, 3, 2, 14, 14]; there is no `post_conv_layernorm` and no learned
>   position table. A still image fills both temporal frames with the same pixels, so the Conv3d reduces exactly to a
>   Conv2d over the temporal-summed kernel (llama.cpp does the equivalent: two `conv_2d` added).
> - **`downsample`** is a biased Conv2d [1536, 1024, 2, 2]: one matmul over each merge block's four patches, flattened
>   channel-major `(c, dy, dx)`.
> - **Merger:** `proj` [1536, 1536], LayerNorm(1536) with weight and bias, **exact-erf GELU**, then a `silu`-gated MLP
>   1536 → 4608 → 1536. No biases anywhere in the merger.
> - **Read the generated `modeling_glm_ocr.py`, not `modular_glm_ocr.py`.** In the modular file `GlmOcrVisionMlp` calls
>   its parent's constructor (which builds the layers at `out_hidden_size` = 1536, unbiased) and only then resets
>   `intermediate_size`, so read alone it implies the wrong shapes. The generated file and the checkpoint agree with each
>   other, and with this list.

**Preprocessing** (`preprocessor_config.json`):
- `Glm46VImageProcessor`, patch 14, merge 2, temporal 2;
- CLIP mean and std;
- pixel bounds `shortest_edge: 12544`, `longest_edge: 9633792`.

That upper bound is about 9.6 megapixels: about 49,000 patches in full attention through the tower, and about
12,000 image tokens into the decoder after the 2 × 2 merge. O4 exists because of that number.

> **Corrected 2026-10-01 (O0): those figures are 2× too high.** The processor calls `smart_resize` with
> `num_frames = temporal_patch_size = 2`, and its budget test is `t_bar * h_bar * w_bar > max_pixels`, so the bounds
> apply to **two frames** of a still image. The real ceiling is `longest_edge / 2` = 4,816,896 pixels (**4.82 MP**) =
> **24,576 patches** through the tower = **6,144 image tokens** into the decoder; the floor is 6,272 pixels. Measured by
> running the real `Glm46VImageProcessor` (transformers 5.12.0): 6000×4000 → grid [1,128,192], 24,576 patches, 6,144
> tokens; 4000×3000 → [1,134,180], 6,030 tokens; 1700×2200 → 4,819 tokens; 100×100 → 112×112, 16 tokens. O4 still
> exists, but its ceiling is 4.8 MP, so measure at 1, 2 and 4.8 MP (the ceiling), not 1, 2 and 4.

**Special tokens.** Image 59280, image start/end 59256/59257, video 59281 (unused here).

## 2. What goinfer has (checked at `1a1f3e10`)

| need | in the tree | gap |
|---|---|---|
| Image embeddings into the decoder | `forwardFromEmbed` on the generic path | none expected |
| m-RoPE positions, post-image delta, image-hash reuse | `GenerateQwenVL`, `mropePositions` / `mropeDelta` (`decoder/rope.go`) | GLM's `get_rope_index` must be diffed against them |
| Resident decode after an image | `decoder.ResidentMRoPE`, implemented by CUDA and now by Metal (`metal/backend.go`) | confirm GLM rides it, don't assume |
| Four norms per layer | `NormPlacement` `Sandwich4` (Gemma) | **order matches; tensor NAMES do not** (O0: `post_attention_layernorm` is the pre-MLP norm here) |
| Fused `gate_up_proj` | split at load (`buildPhi3Weights`) | reuse the split |
| Pairwise rotation | `applyRoPEInterleaved` (`decoder/rope.go`) | **pairwise rotation and contiguous m-RoPE sections together** may not exist as one path (**O0: confirmed absent**; `applyMRoPE` is NeoX-only, and `ropeAt` ignores the `interleave` flag for positions inside the image block; the scalar branch after it honors it) |
| Vision tower | aikit `QwenVisionEncoder` (Qwen2.5-VL) is the nearest | new encoder; see O2 (**O0: the nearest is `Qwen3VisionEncoder`**, which already has the biased Conv3d, the 2D rotary and full attention) |
| Image preprocessing | `multimodal/qwen_preprocess.go` | **O0: reproduces GLM's with the pixel bounds halved**; patchify order identical |
| Tokenizer | byte-level BPE with `ignore_merges` (`tokenizer/bytelevel.go`) | not in the original table; GLM-OCR's own 59,246-entry BPE, see O0 results |
| Image attach in the browser UI | W11 | none |
| Schema-bound output | `constrain` and `response_format: json_schema` | none |

## 3. Items

### O0 — read, then write down what you found (no code; may stop the task)

Record the findings in this doc as a dated paragraph, and correct §1 and §2 in place wherever they are wrong.
1. **Text decoder, from source.** Read `modeling_glm4v.py` and `modular_glm_ocr.py` at the version the
   checkpoint names. Confirm the norm order, the pairwise rotation, the contiguous m-RoPE recomposition and
   `repeat_interleave`, and that layer 16 is the MTP layer.
2. **Positions.** Diff GLM-4V's `get_rope_index` against `mropePositions` / `mropeDelta` for one image
   mid-prompt: t/h/w ids at the image, and the scalar position after it.
3. **Tower.** Confirm every fact in §1, including Conv3d bias, what `post_layernorm` is, and the merger order.
4. **Preprocessing.** The resize rule (factor 28, the pixel bounds), how a still image fills temporal
   patch 2, and the exact chat template and task prompts. Take them from the model card and the official SDK;
   do not guess them.
5. **Prior-art sweep (mandatory).**
   - llama.cpp's GLM-OCR path (`ggml-org/GLM-OCR-GGUF`, its clip projector, and any pixel cap it applies by
     default);
   - Ollama's GLM-OCR;
   - the onnxruntime-web build (`brad-agi/glm-ocr-onnx-webgpu`);
   - vLLM.
   
   Note anything they do that the HF source does not say.
6. **The rotary combination.** Say whether any existing family already needs pairwise rotation together with
   contiguous m-RoPE. If none does, O1 adds it as one small, pinned function, not a flag on two others.

**Stop after O0 if** the text decoder turns out not to fit the generic path (an own-forward requirement), or
if the tower needs something aikit's encoder design cannot take. Write it up and leave the call to Francis.

### O0 results (2026-10-01)

**Neither stop condition holds; O1 and O2 are unblocked.** The text decoder is every existing part (GQA without bias,
fused `gate_up_proj` split at load, `Sandwich4`, an untied head, a skipped layer) plus one small rotary function. The
tower has the same structure as aikit's `Qwen3VisionEncoder` with a short list of deltas (O2).

*What was read.* transformers 5.12.0 in `~/.venv-vl` (the generated `modeling_glm_ocr.py`, `modular_glm_ocr.py`, the
`glm4v` classes it subclasses, and the `glm46v` processor); the checkpoint `zai-org/GLM-OCR` at revision `2e85a628`
(config, `preprocessor_config.json`, `tokenizer.json`, the chat template, the card, and the safetensors header by range
request, so every shape below is the checkpoint's own). The checkpoint's config names transformers 5.17.0, so this is an
older reader: the shape cross-check is what makes the facts safe. **No weights were downloaded and no forward ran, so
nothing here is a numeric match to a real forward; O1–O3's gates are.**

1. **Text decoder, from source:** confirmed, with the specifics in §1's confirmation note. The one trap: tensor *names*
   differ from Gemma's even though the norm order is the same.
2. **Positions: identical for a still image.** HF's `get_rope_index` was executed on the real 24-token probe sequence and
   five synthetic ones (8×8, 10×4 and 4×10 grids, a 128×192 grid at 6,156 tokens, and two images in one prompt) and
   compared with a transcription of `mropePositions` / `mropeDelta`: positions and delta equal in all six. The rule is
   `t = base`, `h = base + row`, `w = base + col`, then the scalar resumes at `base + max(h, w) / merge`; the begin and
   end-of-image tokens are text (only `<|image|>`, 59280, is type 1). *Disclosure: the Go side was transcribed to Python,
   not executed.* `GenerateQwenVL` needs no position change; O3's token-identity gate runs the real Go.
3. **Tower:** confirmed, with the additions in §1 (the axial 2D rotary, the per-head q/k norm, the biased Conv3d, and
   the generated-versus-modular trap).
4. **Preprocessing.**
   - **Resize:** factor 28, bicubic (`resample: 3`), bounds applied to two frames, hence the 4.82 MP ceiling (§1
     correction). `multimodal/qwen_preprocess.go`'s `qwenSmartResize` reproduces it **if the loader passes
     `MinPixels = shortest_edge / 2` = 6,272 and `MaxPixels = longest_edge / 2` = 4,816,896.** Passing the file's values
     unhalved would silently double the budget.
   - **Still image:** both temporal frames are the same pixels, `grid_t = 1`. The patchify order is (block-row,
     block-col, merge-row, merge-col; channel, temporal, py, px), identical to the existing Qwen path. CLIP mean and std,
     rescale 1/255.
   - **Chat template, rendered with the real tokenizer:**
     `[gMASK]<sop><|user|>\n<|begin_of_image|><|image|>×N<|end_of_image|>Text Recognition:<|assistant|>\n`, ids
     `59248 59250 59253 10 59256 59280… 59257 3649 7404 49600 58 59254 10`. The image comes first, with no newline
     between `<|end_of_image|>` and the prompt. A `<think></think>\n` is appended after `<|assistant|>\n` **only** when
     `enable_thinking` is false. Whether the model opens a `<think>` block unprompted needs a forward: O3 checks it.
   - **Task prompts:** `Text Recognition:`, `Formula Recognition:`, `Table Recognition:` (the card); Ollama's page also
     lists `Figure Recognition:`, which the card does not.
   - **Extraction prompt:** `请按下列JSON格式输出图中信息:` followed by a JSON object whose values are empty strings. That
     is a **template, not a JSON Schema**; see O5.
5. **Prior-art sweep.** Nothing contradicts the HF source.
   - **llama.cpp** (`ggml-org/GLM-OCR-GGUF`, Q8_0 by default): its `glm4v` vision graph adds two `conv_2d` patch
     embeddings (the temporal frames), uses a 2D vision-type rotary on spatial positions and RMSNorm, notes "GLM-OCR does
     not have learned position embeddings", and merges with a conv2d, then LayerNorm + GELU, then a gated FFN. That is
     HF's graph. Practical reports, not verified by us: it needed `--flash-attn off`; one user ran `-c 12000`; one reported
     repetition and mangled output on mixed text-and-number forms and better results through the official SDK's layout
     stage.
   - **vLLM** (recipe): `--speculative-config.method mtp --speculative-config.num_speculative_tokens 1`, temperature 0,
     transformers ≥ 5.0, 131,072 context; **no measured MTP speedup given.** The peers use the MTP layer §6 parks.
   - **Ollama:** `glm-ocr:latest` 2.2 GB, `q8_0` 1.6 GB, `bf16` 2.2 GB, 128 K context.
   - **onnxruntime-web build:** int8, tower 394 MB + decoder 471 MB + embeddings 348 MB ≈ 1.2 GB, about 20 tok/s with a KV
     cache (0.3 without), carrying the 3D position ids as an explicit model input. That confirms §4's premise.
6. **The rotary combination does not exist.** `applyMRoPE` is NeoX-only, and `ropeAt` ignores its `interleave` flag
   inside the image block. Every `ropeAt` call site except the two `forward_qwen35*.go` ones already passes both
   `arch.ropeInterleave` and `arch.MRopeInterleaved`, so the plumbing is there. O1 adds **one new function**, a pairwise
   m-RoPE that takes the contiguous-section component for each frequency index, called from `ropeAt` when the pairwise
   flag is set and the position is inside the image block. `applyMRoPE` stays untouched so the Qwen paths stay
   bit-identical.

*Findings the six items did not ask for.*
- **Tokenizer.** Byte-level BPE, 59,246 entries plus 36 added tokens, `ignore_merges: true`, a Split regex
  (case-insensitive contractions, `\p{N}{1,3}`) over ByteLevel with `use_regex: false`; the model's vocabulary is padded to
  59,392. `tokenizer/bytelevel.go` handles `ignore_merges`. `<think>`, `<|image|>` and the tool tags are added tokens with
  `special: false`, so the tokenizer must split on non-special added tokens. O1 checks id-identical tokenization on strings
  with digits, newlines and every added token.
- **GPU residency (O6).** *(Corrected 2026-10-01 by O1: the next two sentences were WRONG. Every resident rope kernel is
  half-split NeoX, scalar ones included, so decode past the image cannot ride a GPU resident either; see O1's result.)*
  Decode past the image needs only the resident's scalar rope at `ropePos` (`ResidentMRoPE`,
  `ForwardMRoPE`), which is pairwise on the families that already run it (Command-R is resident on CUDA and Metal). The
  image-block *prefill* kernels (`rope_kv_mrope_batched` on CUDA, and Metal's) are NeoX-only, so GLM's image prefill
  cannot use `PrefillMRoPELast` without a pairwise variant. Expect a CPU prefill of the image rows, or a new kernel.
  O6 measures which.
- **Context.** A full-budget page is 6,144 image tokens before any output, so serve's default context must leave room
  (llama.cpp users run 12,000 or more).

### O1 — the text decoder, `glm_ocr_text` (and `glm_ocr` as the wrapper `model_type`)

- One registration in `decoder/registry.go`. It is built from the existing parts in §2: GQA without bias,
  `gate_up_proj` split at load, `Sandwich4`, the pairwise m-RoPE function from O0.6, an untied head, and layer 16
  skipped at load.
- **The tensor-name map is its own test** (O0): `post_self_attn_layernorm` is the post-attention norm and
  `post_attention_layernorm` is the pre-MLP norm, the reverse of Gemma's `post_attention_layernorm`. The test must go red
  if the two are swapped, since both are [1536] and load either way.
- **Stop ids** are [59246, 59253]. **Tokenizer check:** id-identical to HF on strings with digits, newlines and every
  added token (59246–59281).
- `scripts/pin_glm_ocr_tiny.py`: a random-weights checkpoint with the real text shape and a reduced layer
  count, text only.
- **Gate:** exact argmax and continuation against HF, text only. The rotary function is pinned directly
  against HF's slicing, the way `mropeComponentInterleaved` was.

**O1 RESULT, 2026-10-01: DONE, all gates pass** (local commits `ff79b92a`, `27845987`; not pushed). Re-run and read for
`--- PASS` by the lead.
- **Built:** `glmOcrArchitecture` under `glm_ocr` and `glm_ocr_text` (every field explicit, plus a `validateGlmOcr` that
  refuses what the descriptor cannot do); `buildGlmOcrWeights` (gate-first split, the reversed norm names, layer 16 and
  `model.visual.*` never requested); `applyMRoPEPairwise` called from `ropeAt` only when the pairwise flag is set and the
  position is inside the image block. `applyMRoPE` and every Qwen path are untouched.
- **Rotary pin** against HF's real `GlmOcrTextRotaryEmbedding` + `apply_rotary_pos_emb` (head_dim 128, 32 positions including
  image blocks at bases 5 and 200): max|diff| 1.7e-5; the controls it must reject read 0.4 (temporal-only), 3.7 (strided
  layout) and 6.0 (NeoX). Equal positions are bit-identical to `applyRoPEInterleaved`; the Qwen dispatch is pinned bit-identical.
- **Tiny decoder vs HF, text only** (48-token prompt, f32): argmax exact, cosine 1.000000000, max|diff| 1.9e-5, sequential and
  batched paths, and the 8-token greedy continuation identical. **Through the caller** (`mropePositions` +
  `prefillLogitsQwenVL`, image-style positions): cosine 1.000000000, max|diff| 9.6e-6, continuation identical; scalar positions
  would read cosine 0.138, so the golden is sensitive.
- **Real checkpoint, scratch check (not committed):** the full 2.65 GB text decoder, 26-token prompt, 12-token greedy
  continuation: cosine 1.000000000, max|diff| 1.9e-5, continuation identical. **The family's parity row stays `tiny-golden`
  (experimental), not `validated`, until a committed `realckpt` gate with an `assets.json` entry exists.**
- **Name-mapping test never skips** and goes red on a swap (each tensor distinct; the layer-2 MTP block, a layer-9 tensor and
  `model.visual.*` are NaN-poisoned with wrong shapes). Mutations, each shown red then reverted: norm names swapped, gate/up
  swapped, `ropeInterleave` off, strided layout inside the pairwise function.
- **Tokenizer:** 227 cases id-identical to HF's `AutoTokenizer` (1–7 digit runs, newlines, edge spaces, CJK, all 36 added
  tokens including the 15 with `special: false`, in the chat-template shape). No mismatch. Skips cleanly without
  `~/models/glm-ocr/tokenizer.json`.
- **Housekeeping:** `refresh_parity_hashes.sh` 64 forward goldens passed, 0 failed, 0 skipped; gofmt, vet (untagged and the
  `realckpt`, `goinfer_testhooks`, `cuda goinfer_testhooks` variants) and staticcheck 0.8.0 (proven red on a dead field)
  clean; capability and hardware matrices regenerated; README "39" → "40 model families".
- **O0 corrections: none.** One nuance: HF's `cos`/`sin` are `cat(freqs, freqs)` and the `repeat_interleave(2)` happens
  inside `apply_rotary_pos_emb` on the first half; the result is what §1 said.
- **FINDING (SUPERSEDED 2026-10-02: fixed on CUDA by new pairwise rope kernels, `a306d33e`; CUDA now admits `glm_ocr`, Metal and WebGPU still
  decline): no GPU backend may run `glm_ocr`.** With admission bypassed, resident-vs-CPU on the RTX 2070 SUPER (int8int8,
  48-token prompt) read worst cosine −0.34 on glm-ocr-tiny; a NeoX llama control with the same peaked attention stayed at
  1.000000 and CPU int8int8 against f32 alone read 0.988. The CUDA `rope` and `rope_kv` kernels and Metal's `rope` kernel are
  half-split NeoX, and so are the m-RoPE image-prefill kernels. `FeatPairwiseMRoPE` (derived from `ropeInterleave &&
  MRopeSection`) is declared by no backend, so **`glm_ocr` runs on the CPU everywhere**; pinned by
  `TestGlmOcr_residentDeclined` and `cuda/glm_ocr_resident_test.go`.
- **FINDING, MEASURED 2026-10-01 (lead, after O1 reported it as a hypothesis): the CUDA resident is wrong on real Cohere and
  Cohere2 checkpoints.** Scratch test (not committed), real checkpoints at int4 on the RTX 2070 SUPER with a 512-position
  resident context (the default context does not fit beside int4 weights on 8 GB, so the first attempt declined for fit, which
  is unrelated), resident against CPU at the same quant and against the HF f32 golden:
  - **Command-R7B** (`cohere2`): position 0 matches exactly (cosine 1.000000); on the 6-token golden prompt the last position
    reads 0.445 against the CPU, and the last token reads 0.583 against HF where CPU int4 reads 0.967; on a 48-token prompt the
    worst position reads −0.075. The 8-token greedy continuation matches the golden 2/8 (CPU also 2/8 at int4).
  - **Aya-expanse-8B** (`cohere`, no sliding window, no NoPE): 6-token prompt last token 0.988 against HF (CPU 0.997), but the
    greedy continuation matches the golden **1/8 against the CPU's 8/8**; the 48-token prompt's worst position reads −0.041.
  - **Reading:** exact at position 0 (rotation is the identity there) and diverging as positions grow, on a model with neither
    sliding window nor NoPE, points at the rotation. The mechanism fits: the CUDA `rope` kernel rotates pairs `(d, d+half)` and
    takes no interleave argument, while Cohere's is GPT-J pairwise `(2d, 2d+1)`. **Not isolated further** (no kernel-level
    check), **Metal not run**, and the committed `TestCohereResidentParityCUDA` cannot see it (a flat 0.02-std fixture, bar 0.995).
  - **UPDATE 2026-10-02: FIXED ON CUDA** (pairwise rope kernels, `a306d33e`; real Aya and Command-R7B gate
    `TestCohereRealResidentParityCUDA`, record `docs/measurements/cuda-pairwise-rope-2026-10-01.md`). Cohere, Cohere2 and `glm_ocr` are now
    CUDA-only; Metal and WebGPU decline them until the Mac ports the rotation. What follows is the state when this was measured.
  - **Not fixed here (as measured).** Command-R, Command-R7B and Aya were admitted on CUDA and Metal and shown as resident in the hardware
    matrix. The options (decline them as `glm_ocr` is declined, or write pairwise kernels) are the owner's call.

### O2 — the tower in aikit

- A new encoder, not a flag on an existing one. **Start from `Qwen3VisionEncoder`** (`aikit/vision/qwen3_encoder.go`;
  corrected 2026-10-01, O0), not `QwenVisionEncoder`: it already has the biased Conv3d patch embed, the 2D rotary over
  (row, col) at theta 1e4, full attention with no windows, fused biased `qkv`, and patches kept in merge-block order.
  Deltas from it: **no** learned position table; **RMSNorm** (eps 1e-5, weight only) in place of LayerNorm; **per-head
  q/k RMSNorm** before the rotary; a **`silu`-gated** 4096-wide biased MLP in place of the non-gated tanh-GELU one; a
  `post_layernorm`; the **`downsample` Conv2d** (a matmul over each merge block, flattened `(c, dy, dx)`); and GLM's
  merger (`proj` → LayerNorm → erf-GELU → gated 4608 MLP, unbiased). For a still image the Conv3d can be a Conv2d over the
  temporal-summed kernel.
- A HF pin generator for tower-only merged features.
- **Gate:** cosine ≥ 0.9999 per merged row at f32 against HF, on a small image and a page-sized one.

**O2 RESULT, 2026-10-01: DONE, gate passes** (aikit `ed8053c`, goinfer `495783ca`; local, not pushed, not released). Re-run by the
lead: `TestGlmOcrVisionEncoder_*` all PASS, the real-checkpoint parity (small + page) in 59 s.
- **API for O3:** `vision.LoadGlmOcrVisionEncoder(dir, quant)` then `enc.Forward(pixelValues, gridTHW) ([]float32, error)`.
  Input `[n_patches, 1176]` pre-patchified in merge-block order (the contract of `Qwen3VisionEncoder.Forward` and the layout
  `QwenPreprocess` produces); `gridTHW` one `{t,h,w}` per image in patch units, `h` and `w` multiples of 2; output
  `[Σ t·h·w / 4, 1536]` row-major, equal to HF's `pooler_output`. Stage hooks `Embed`, `ForwardViT`, `ForwardDownsampled`.
  `quant=true` is W8A8; sanity-checked on the tiny fixture only (worst-row cosine 0.9976), **not gated on the real checkpoint**.
- **Real checkpoint, f32, Go vs HF eager attention (transformers 5.12.0), same pixel bytes on both sides** (HF's
  `Glm46VImageProcessor` output, dumped): small image (230×220 → 224×224, grid [1,16,16], 64 merged rows) min per-row cosine
  1.000000000, max|diff| 5.4e-6; page (1100×1350, grid [1,96,78], 7,488 patches, 1,872 merged rows) min per-row cosine
  0.999999987 (row 684), max|diff| 3.4e-5 (relative 7.0e-5). *Both images are procedurally rendered pages, not real scans.*
- **Tiny committed fixture** (head_dim 64, two non-square grids in one call, non-trivial norm weights and biases): relative
  max|diff| ≤ 1.6e-6 at every stage against a 5e-6 bar.
- **Mutations, each shown red then reverted:** q/k norm skipped (worst-row cosine 0.205); downsample flattening patch-major
  (−0.05); rotary before the q/k norm (0.92; visible only because the norm weights are non-uniform); **merger GELU erf → tanh
  reads cosine 0.99999997, which a cosine-only gate would pass, and the relative max|diff| bar catches it ~40× over.**
- **Shared code:** the fused-attention tail of `packedAttentionInto` moved verbatim into `attendPackedInto` (the way qwen3 did);
  the Qwen towers' tests all pass after it.
- **Lint:** gofmt, vet and staticcheck 0.8.0 clean. **golangci-lint could not run** (the installed binary is built for Go
  1.26, aikit needs 1.27); `releasegate` will need a rebuilt one.
- **Departure from aikit's fixture convention:** the 1.6 MB tiny checkpoint is **force-added** (`git add -f`, `*.safetensors` is
  gitignored in aikit) so the tiny gate cannot skip on a clean clone. Owner's call whether to keep it.
- **Cost, NOT gated:** the only number is 30.0 s at 1 MP on nobara at load average ~14, **contaminated and not quotable**.
  The 1 / 2 / 4.8 MP sweep (`TestGlmOcrVisionEncoder_costSweep`, `AIKIT_GLM_OCR_COST=1,2,4.8`, grids [1,70,72], [1,100,102],
  [1,128,192]) is a Mac night job; a paste-ready prompt was given to Francis. The 4.8 MP point is guessed at 10–20 min.
- **Two small facts:** the merger's inner width 4608 is `out_hidden_size × in_channels` (1536 × 3), which only happens to equal
  the text MLP width; and a Conv3d-as-Conv2d over the temporal-summed kernel is exact mathematically but not bit-equal in f32,
  so the reference path is the plain 1176-wide matmul.
- **Release needs (not started):** an additive minor (read the number with `RELEASING.md`'s command); a CHANGELOG section with the
  compare link listing `GlmOcrVisionEncoder`, `GlmOcrEncoderConfig`, `LoadGlmOcrVisionEncoder`; the perfgate exception text (no
  `linalg` or `go.mod` change, one new file plus a verbatim move); `go run -C tools ./releasegate X.Y.Z` (rebuilt
  golangci-lint) and `./vulncheck`; push, root CI green, tag; then `gpupins --fix`; the pin bump in goinfer's five modules in
  one commit; the GitHub Release on goinfer's root tag.
- **Cost, measured and recorded (not gated):** tower wall time at three pixel counts on the Mac CPU. O4 needs
  them.
- Tag an aikit release and bump the pin in all five modules in one commit (`RELEASING.md`, version
  alignment).

### O3 — end to end on the real checkpoint

- Wire the tower and the decoder through `GenerateQwenVL`, or a GLM variant of it if O0.2 says positions
  differ. That covers the image start/end tokens and the serve vision route.
- **Gate:** greedy output token-identical to HF for 64 tokens on three document images (an invoice, a table,
  a page with a formula), at a pixel count both sides use.
- **Gate:** a text-only turn on the same checkpoint is unchanged by O3, and the forward goldens stay green.

**O3 RESULT, 2026-10-02: DONE, gate passes** (local commits `87efcaff`, `466115df`, `25569f32`, `18490f39`, `e08c4831`, `12f00af2`, `244646aa`,
`2996ae23`; logs in `docs/measurements/glm-ocr-o3-2026-10/`). The agent's figures were re-checked by the lead on the invoice: `TestGlmOcrE2E_f32/invoice`
PASS in 70 s, `TOKEN-IDENTICAL: 64/64`, cosine 0.999999995, max|diff| 0.00198, the same numbers the agent reported.
- **The gate.** Goinfer at f32 (no quantisation) on the real checkpoint against transformers 5.12.0 f32 CPU eager, greedy, the checkpoint's own
  processor and chat template. **All three images are procedurally rendered documents, NOT real scans** (`scripts/gen_glm_ocr_doc_images.py`,
  PNGs and HF goldens committed under `testdata/glm_ocr/`, 0.94 MB):

  | image (prompt) | grid, patches, image tokens, prompt ids | f32 identity | last-token logits, goinfer pixels | with HF's pixels |
  |---|---|---|---|---|
  | invoice 1000×1300 (`Text Recognition:`) | [1,92,72], 6,624, 1,656, 1,668 | **64/64** | cos 0.999999995, max 1.98e-3 | cos 1.000000000, 2.4e-5 |
  | table 1200×900 (`Table Recognition:`) | [1,64,86], 5,504, 1,376, 1,388 | **64/64** | cos 1.000000000, 4.6e-4 | cos 1.000000000, 1.9e-5 |
  | formula page 1000×1200 (`Formula Recognition:`) | [1,86,72], 6,192, 1,548, 1,561 | **64/64** | cos 0.999999990, 2.8e-3 | cos 1.000000000, 5.8e-5 |

  No divergence, so no first-divergence step. HF's smallest top-1/top-2 gap over the 64 steps was 1.75 / 4.23 / 0.63 (no near-ties). The ~1e-3 residual in
  the goinfer-pixels column is **preprocessing**: 99.967% / 99.881% / 99.965% of goinfer's `pixel_values` floats are bit-identical to HF's, the largest
  difference is 2.1 eight-bit levels (PIL bicubic against goinfer's, "tolerance-matched, not bit-exact"), and feeding HF's own pixels drops it to ~1e-5.
  Tests: `TestGlmOcrE2E_f32` and `TestGlmOcrE2E_f32_hfPixels` (decoder, tag `realckpt`, heavy; assets `GOINFER_GLM_OCR`, `GOINFER_GLM_OCR_E2E`);
  serve stack `TestServe_glmOcrImage_O3`. Goinfer's serve prompt ids equal HF's, and the HTTP reply equals HF's 64 tokens decoded.
- **What was built.** `glm_ocr` chat template (byte-exact against HF on text turns); the image block and task prompts; `multimodal.LoadGlmOcrPreprocessConfig`
  with the **pixel bounds HALVED** (6,272 / 4,816,896), pinned by unit tests in `multimodal` and `serveapp` that fail if the file's values are used unhalved;
  serve auto-discovery of a `glm_ocr` dir, a lazy f32 CPU tower (int8 only with `-vision-quant int8`, since the int8 tower is not gated on the real checkpoint), and
  the prompt path. **No fork of `GenerateQwenVL`:** nothing differs, including no embed scale. Serve applies no per-image cap beyond the processor's own 6,144
  tokens; the default is O4's decision.
- **Image too large for the context:** a named HTTP 400 `image_too_large_for_context` before the tower runs (1,656-token invoice against a 1,024-token model
  context). With `--ctx 1024` and the 1,668-token invoice serve falls back to the CPU path, returns the same text, and does not truncate the image.
- **CUDA resident against CPU** (real pages, int4 both sides, resident context 4096): prefill logit cosine 0.9983 / 0.9979 / 0.9972, same argmax, and 64 greedy
  tokens identical to the CPU on all three (the resident image prefill was taken). At the ceiling (6,156-token prompt, 6,144 image tokens, grid 128×192,
  `ResidentContext` 8192): prefill cosine 0.9986, 16 tokens identical; **the image rows there are the invoice's real tower rows tiled to 6,144, not a real 4.8 MP
  tower run** (not done: it would take minutes of CPU).
- **Context that fits on the 8 GB card** (`TestGlmOcrCUDAContextFit`): resident at 2,048 through 32,768 positions; 65,536 declines with the reason shown (8.59 GB of KV at
  128 KB per position against 6.97 GB free; about 54k positions by that arithmetic). 8,192 covers a 6,144-token image plus output with about 1 GB of KV.
- **nil resident:** a CPU-only `glm_ocr` load runs the whole `GenerateQwenVL` turn, twice with the same image hash (`TestGlmOcr_generateQwenVL_cpuOnly`).
- **HTTP smoke** (CUDA build the pinned way): banner `[chat: glm_ocr]`, `decode path: cuda-resident (int4)`, `context: 8192`; the CUDA reply equals the library-level resident
  int4 text and the CPU-backend reply; full-length requests finish with `stop` after 321, 419 and 246 tokens.
- **OCR text against ground truth** (we generated the images; default int4 on CUDA, full length): invoice: every number, address and line item matches, **but the
  "Invoice No: INV-2026-0417" header line is dropped** (HF f32 reads it; int4 diverges from f32 at token 4). Table: all 36 cells exact, returned as an HTML table (the
  caption and footer are omitted: `Table Recognition:` returns only the table); int4 matches f32 on all 64 tokens. Formula page: both equations correct as LaTeX, the prose comes out
  as spaced-out `\mathrm{M o r e …}` with a few errors (expected for `Formula Recognition:` on a text-heavy page); int4 diverges from f32 at token 54. **int4 costs real accuracy here;
  it is not the same OCR as f32.**
- **`<think>`:** none appeared in any output (three f32 HF runs, three full-length int4 runs, the CPU smoke). The `glm_ocr` template has no reasoning detection, so serve does no
  splitting; a block, if it opened, would appear verbatim in `content`. No think handling was built.
- **Mutations, each shown red then reverted:** swapped norm names (e2e cosine 0.534, diverges at step 0); scalar positions in the image prefill (cosine 0.917, first 4 tokens
  match, diverges at step 4); unhalved pixel bounds (unit tests red in `multimodal` and `serveapp`; the unhalved bounds give the wrong grid on 4 of the 6 HF-measured sizes); image placed
  after the task prompt (ids test red); a newline concatenated onto the block (the AST guard red). **Gap:** the e2e documents are at most 1.3 MP, so unhalved bounds do not change
  their grids; only the unit tests catch that mutation.
- **Behaviour change to know:** `glm_ocr` text-only turns now go through the new chat template (before, detection found none and the message went in as a raw completion). The decoder forward
  goldens are unchanged and the text-only prompt ids match HF.
- **Evidence:** serveapp `go test -v` 289 PASS, 51 SKIP, 0 FAIL (the new tests pass, not skip); the VL decoder subset 54 PASS; parity refresh 64 goldens passed; gofmt, vet (untagged, `realckpt`,
  `cuda`/`gpu goinfer_testhooks`) and staticcheck 0.8.0 (proven red on a dead field) clean; citation lint exit 0 (23 moved citations re-pointed by `--update`, one AMBIGUOUS re-pointed by hand).
- **Not done:** a real 4.8 MP page through the tower; the full CUDA suite (only new test files changed there); `go run ./cmd/gate census`.
- **Needs the unreleased aikit:** `GOWORK=off go build ./...` fails only in `internal/serveapp` ("undefined: vision.GlmOcrVisionEncoder"), which blocks root `cmd/serve`, `cuda/cmd/serve`,
  `gpu/cmd/serve` and `metal/cmd/serve`, plus `decoder/glm_ocr_e2e_real_test.go` and `cuda/glm_ocr_real_image_test.go`. `chat`, `multimodal` and the untagged decoder build with `GOWORK=off`.
- **`origin/main` has three new commits** touching `internal/serveapp/banner.go` and `testdata/assets.json`; they are not merged here and the `assets.json` append may conflict.

### O4 — the pixel budget (owner decision on the default)

The processor's own ceiling (corrected 2026-10-01, O0: **4.82 MP, 24,576 patches, 6,144 image tokens**; the original
9.6 MP / 12,000 figure was 2× high) costs far more on a laptop than a typical scan needs.
- Measure tower plus prefill time at 1, 2 and 4.8 MP (the ceiling) on the Mac (CPU, plus Metal if O6 has it), and record
  what each does to the O3 outputs.
- Then propose a default cap for `serve` and `chat`, with a flag to raise it. Francis picks the default.

### O5 — structured extraction, the reason to do this

- GLM-OCR's information-extraction prompt with `response_format: json_schema` on `serve`. **The model's prompt carries a
  JSON *template* (`请按下列JSON格式输出图中信息:` plus an object of empty strings), not a JSON Schema** (O0), so build both
  from the one Go struct: its zero value, marshalled, is the prompt's template, and `constrain.GrammarFromStruct` is the
  grammar. A schema-only request has no template to show the model, and a template-only one has no guarantee. In
  `goinfer-chat`, `--schema` exists already (`internal/chatapp/main.go`) and `--image` does not; add it, so
  that `goinfer-chat --image invoice.png --schema invoice.schema.json` is the one-line demo.
- A worked example in `examples/`: an invoice image into a Go struct, through `constrain.GrammarFromStruct`.
- **Reported, not gated:** field-level accuracy on a small hand-labelled set of 10–20 documents. Schema
  validity needs no gate, because the grammar guarantees it. Report it with C1's per-field confidence where
  that applies.
- The same flow works in the `-web` UI through W11's Attach image. Record a short capture of it for the site.

**O5 RESULT, 2026-10-02: DONE** (local commits `af26fe3b`, `1aa61eb3`, `51aecda7`; record and logs in `docs/measurements/glm-ocr-o5-2026-10/`, README first). The lead re-ran the
demo and recomputed the accuracy table from the raw results file; both matched the agent's report.
- **The one-line demo** (CUDA int4, the committed rendered invoice): `goinfer-chat --model ~/models/glm-ocr --image testdata/glm_ocr/invoice.png --schema testdata/glm_ocr/invoice.schema.json`
  prints schema-valid JSON: `invoice_number` INV-2026-0417, 6 line items, total 1140.55 (the CUDA chat binary, 50 s wall: the f32 vision tower on the CPU took 44 s, then 416 tokens
  at about 104 tok/s; exploratory timing). On the CPU, int4, it takes about 73 s (`TestChatImage_demoEndToEnd`).
- **Template convention.** `constrain.TemplateFromSchema` and `TemplateFromStruct` build the blank-valued JSON object from the same source the grammar is compiled from (a test checks the
  card's own example byte for byte): strings `""`, integers and numbers `0`, booleans `false`, arrays one example element, 4-space indent; the prompt is the card's `请按下列JSON格式输出图中信息:` plus a
  newline plus the template. Marshalling the Go zero value is not used because it gives `null` for slices.
- **Serve rule.** For a GLM-OCR image request with `response_format` of type json_schema, the template prompt REPLACES the text part only when it is empty or exactly one of the bare task prompts;
  any other text is sent verbatim, still under the grammar. `json_object` is untouched and an invalid schema is a 400. The vision route already honoured `response_format`; one small shared change:
  the vision chat route now also writes `goinfer_confidence` back (buffered reply and the stream's trailing event) for every vision family (before, images plus that flag got a 400).
  `multimodal.SpliceImageBlock` moved out of serveapp so serve, chat and the example share it. `goinfer-chat --image` implies a one-shot run, defaults `--temp 0` and `--max 2048`, sends no coding
  system prompt, and is GLM-OCR only (a local checkpoint directory).
- **Accuracy, REPORTED NOT GATED, on 15 SYNTHETIC rendered invoices (not real scans), int4 on CUDA, greedy, no text part** (0.99-1.31 MP; documents, labels and a sha256 manifest committed, 0.55 MB). Normalised:
  whitespace collapsed in strings, a currency symbol mapped to its ISO code, money and quantities compared as numbers to the cent; dates compared as printed, nothing else. **15/15 replies parsed and
  finished `stop`.** All fields **383/397 = 96.5%**. Per field: invoice_number, vendor, currency, subtotal, tax, total, line_item_count 15/15; line-item description, quantity, unit_price, amount 58/58;
  bill_to 14/15; date 11/15; due_date 11/15; paid 10/15. Ten documents are perfect. The 14 misses: **`paid` is false on all five stamped documents (0 of 5 stamps read; the ten unstamped are right)**;
  eight date fields were rewritten in ISO (compared as calendar dates 27/30 are right; the three real errors are a day/month swap in both date fields of one document and one due date misread);
  one `bill_to` gained the street address.
- **f32 (CPU) is only a 3-document subset** (documents 2, 8, 10, 79-114 s each, exploratory), chosen BECAUSE int4 had errors there, so it shows whether f32 recovers int4's misses, not f32 accuracy: 83/89 against
  int4's 82/89 on the same three, the only difference being the over-long `bill_to`. The paid-stamp and date-format behaviour are the model's, not int4's. **The full 15-document f32 pass is queued
  (`glm-ocr-o5-f32-15-invoices`, 40 min estimate, `docs/measurements/glm-ocr-o5-2026-10/run-f32-15.sh`, pinned CPU serve binary and eval script staged under `~/goinfer-logs/glm-ocr-o5/`) and has NOT run.**
- **C1's per-field confidence applies to a vision request now.** It reports integer and boolean fields (quantity, paid), not strings or numbers; a Go struct cannot express an enum so none was evaluated. On `paid`, every wrong
  answer (the five missed stamps, confidence 0.69-0.95) sits below every right one (0.995-0.998), AUROC 1.000; **n=15 with 5 wrong, an indication, not a calibration** (`calibrated` stays false). `quantity` is 58/58
  correct so there is nothing to rank. The same ordering held at f32 on the subset.
- **The `-web` UI:** Attach image reaches glm_ocr (`/v1/models` reports vision true; the UI sends an `image_url` part). **Missing: any `response_format`/schema control (`app.js` never sent it)**, so the UI gave
  unconstrained OCR, or a hand-typed extraction prompt with no grammar; it also defaults to temperature 0.7 and `max_tokens` 512, which truncates a long extraction. **FIXED 2026-10-02 (W36):** the Sampling box has a *Response schema* field, sent as `response_format`; a real headless-Chrome run against a live `serve -web` read the committed invoice into the schema (12 checks, 47 s, record `docs/measurements/web-ui-schema-2026-10-02/`). Still true: the 512 default truncates a long extraction (the field's help text says so), and the screen capture for the site could not be recorded here.
- **FINDING: a string-typed numeric column makes the grammar loop on whitespace until `max_tokens`, silently** (the model wants a bare number, the string grammar forbids it, whitespace is the only legal token left;
  `string_typed_quantity_whitespace_runaway.txt`). The example, schema file and docs therefore type amounts as numbers. The root cause was `constrain`'s unbounded optional whitespace. **FIXED 2026-10-02:** structural whitespace is now bounded (64 bytes, and 1 between a `:` and its value), so the same request finishes with six correct line items in 445 tokens instead of looping; see `docs/measurements/constrain-whitespace-bound-2026-10-02.md`. Typing amounts as numbers is still the right schema.
- **Under the extraction prompt int4 reads the INV-2026-0417 header** on CUDA and CPU (plain OCR dropped it at int4 in O3). **Not isolated:** whether the model uses the Chinese instruction or just the template.
- **Red/green proof** for each new gate under a deliberate break: template number convention, alphabetical key order, a full-width colon, always-replace instead of replace-only-if-bare; the serve extraction call removed, the
  `confidenceOK` AST guard, the user-text-respected rule; the same rule in `imageTurns`, the `--batch` conflict check, unhalved bounds; an extra `Invoice` field (schema-file drift test). The real-checkpoint serve gate
  `TestServe_glmOcrExtraction_O5` (195 s on CPU int4) fails without the template (1668 vs 1778 prompt ids) and, with the grammar off, check (e) fails (a schema capping `line_items` at 3 makes the reply not JSON).
- **Evidence:** serveapp 292 PASS / 51 SKIP / 0 FAIL (the new tests pass, not skip), constrain 68 PASS / 2 SKIP, multimodal 20 PASS, chatapp 36 PASS / 1 SKIP (the heavy demo, run separately), examples 6 PASS / 1 SKIP
  (heavy, run separately); gofmt, vet (untagged and `realckpt`), staticcheck 0.8.0 (proven red) and the citation lint (exit 0) clean; `GOWORK=off go build ./...` builds. **Not run:** `go run ./cmd/gate census`, the parity refresh
  (no `decoder/*.go` change), the full CUDA suite. `docs/flags.md` is serve-only and generated, so `--image` is documented in its own help text, README, `docs/server.md`, `docs/multimodal.md` and `docs/use-from-go.md`.
- **Caveats the agent flagged:** no real labelled documents exist; `scripts/readme_counts_check.py` exits 1 on count drift (tasks/ and measurements/ counts), not investigated or fixed; the README paragraph added
  here will probably conflict with `origin/main`'s changes when merged.

### O6 — GPU paths (each on its own measurement; not blocking)

- **Decode after the image:** confirm whether the CUDA and Metal residents serve GLM-OCR through
  `ResidentMRoPE`. If one does not, `GenerateQwenVL`'s CPU fallback is correct and slower; record which. **Corrected 2026-10-01 (O1): O0's expected split was wrong.** Every resident
  rope kernel was half-split NeoX, scalar ones included, so no backend admitted `glm_ocr`, and the whole turn, prefill and decode,
  ran on the CPU. **UPDATE 2026-10-02: CUDA is done** (pairwise rope kernels `a306d33e`; O3 shows the CUDA resident matching the CPU on real pages and at the 6,144-token ceiling).
  **Metal and WebGPU still decline and run it on the CPU; the Mac must port the rotation** (steps in `docs/measurements/cuda-pairwise-rope-2026-10-01.md`). A GPU path needs pairwise variants of `rope`, `rope_kv` and `rope_kv_mrope_batched` on CUDA and the Metal
  equivalents, gated first by a resident-vs-CPU parity test on glm-ocr-tiny (a flat-weight fixture cannot see this bug; see the
  Cohere finding under O1). Measure the CPU turn first (6,144 image rows through a 0.58 B decoder, then CPU decode); it may be
  cheap enough to leave.
- **The tower:** CPU in v1, unless aikit's GPU vision path covers it cheaply. A GPU tower is a follow-on with
  its own band.

### O7 — docs and site

- Regenerate `capability-matrix.json` after O3. Add GLM-OCR to `multimodal.md`'s served-families paragraph
  and to the README's vision list.
- The site's Ollama-coverage row 22 changes from N to S. Per that snapshot's own rule, this needs a new dated
  snapshot, not an edit.
- A Models-page entry with O5's example and the O4 default stated.
- **Known red before any push (O1, 2026-10-01):** with `glm_ocr` in the matrix, the site module's
  `TestCheckOllama_theRealDataHolds` and `TestBuild_realRepo` (`GOWORK=off`, `site/internal/site`) fail, because Ollama
  row 22 `needs` the family. They demand the new dated snapshot described above after re-reading Ollama's page, and a faked
  one would defeat them. So the O1 commits should not be pushed until that snapshot exists (or the owner decides otherwise).

## 4. Would it make a good in-browser demo?

A good demo, yes. An in-browser engine, no; not for this. Two facts settle it:
- **The browser version already exists.** `brad-agi/glm-ocr-onnx-webgpu` runs GLM-OCR entirely client-side
  on onnxruntime-web with WebGPU, about 1.2 GB of int8 files, at about 20 tokens/s by its own card. A goinfer
  version would repeat it.
- **goinfer cannot run on the GPU inside a browser today.** Its WebGPU backend is wgpu-native through cgo,
  which does not exist under `GOOS=js`. Go's wasm target runs single-threaded and without SIMD, so the CPU
  path in a tab would take minutes for a page-sized tower. A `navigator.gpu` backend is a new backend, and
  `roadmap.md` already parks browser/WASM as "a demo, not a binding strategy". This task does not change
  that.

The demo that fits is O5 through the `-web` UI: drop a scan into a browser tab and get validated JSON back,
with the engine running on the visitor's own machine from one downloaded binary. That is the claim goinfer
can make and the browser build cannot: the JSON matches your schema every time, and the whole thing runs
from one binary with no runtime install.

## 5. Order

O0 → O1 ∥ O2 → O3 → O4 → O5 → O7. O6 runs on its own measurements whenever O3 is in.

## 6. Not in scope, stated

- **The layout stage.** The official SDK's pipeline runs PP-DocLayout-V3, a separate detector, before
  recognition. goinfer runs the recognition model on whatever image it is given; a detector is a different
  network class.
- **GGUF.** That needs `mmproj` support (`multimodal.md` P8b). The safetensors path comes first, and at 0.9B
  it costs nothing.
- **Video and multi-page batching.** One image per turn, as the other VL families.
- **The MTP layer as a speculative drafter.** It ships in the checkpoint. goinfer's MTP self-draft adapter was
  removed on 2026-09-25 with that track stopped, so this stays parked unless the track reopens.
- **An in-browser build** (§4).

## Sources

`decoder/registry.go` (phi3's fused split, `qwen2_5_vl`'s m-RoPE registration) · `decoder/rope.go`
(`applyRoPEInterleaved`, `mropePositions`, `mropeDelta`) · `decoder/generate_vl.go` (`GenerateQwenVL`) ·
`decoder/arch.go` (`NormPlacement`) · `metal/backend.go` (`ResidentMRoPE`) ·
[zai-org/GLM-OCR](https://huggingface.co/zai-org/GLM-OCR) (card, `config.json`, `preprocessor_config.json`) ·
[transformers `modular_glm_ocr.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/glm_ocr/modular_glm_ocr.py) ·
[transformers `modular_glm4v.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/glm4v/modular_glm4v.py) ·
[Using OCR models with llama.cpp (2026-04-10)](https://huggingface.co/blog/ggml-org/using-ocr-models-with-llama-cpp) ·
[brad-agi/glm-ocr-onnx-webgpu](https://huggingface.co/brad-agi/glm-ocr-onnx-webgpu)

## Sources read for O0

*Added 2026-10-01.* Read for O0: transformers 5.12.0, `models/glm_ocr/{modeling,modular}_glm_ocr.py`,
`models/glm4v/{modeling_glm4v,processing_glm4v,image_processing_pil_glm4v}.py`, `models/glm46v/` (the processor in use),
`vision_utils.py`; the checkpoint's `config.json`, `preprocessor_config.json`, `processor_config.json`,
`generation_config.json`, `chat_template.jinja`, `tokenizer.json` and safetensors header (revision `2e85a628`);
`multimodal/qwen_preprocess.go`, `decoder/rope.go`, `decoder/arch.go`, `aikit/vision/qwen3_encoder.go`; and, for the
sweep, the llama.cpp mtmd `glm4v.cpp` and discussion 19721, the vLLM and Ollama GLM-OCR pages, and the ONNX WebGPU card.
The probes (processor grid sizes, the rendered template, the position diff) were throwaway scripts in the session
scratchpad and are not kept; O1–O3 will pin the facts that matter as tests.

<!-- doc-reviewed: 2026-10-01 -->

