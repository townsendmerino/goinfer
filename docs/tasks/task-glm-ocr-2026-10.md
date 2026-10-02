# Task: GLM-OCR — documents in, text and schema-bound JSON out (O0–O7) — 2026-10

> **Status: O0 DONE 2026-10-01 — it does not stop the task; O1 and O2 are unblocked.** O0's findings are in
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
- **GPU residency (O6).** Decode past the image needs only the resident's scalar rope at `ropePos` (`ResidentMRoPE`,
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

### O6 — GPU paths (each on its own measurement; not blocking)

- **Decode after the image:** confirm whether the CUDA and Metal residents serve GLM-OCR through
  `ResidentMRoPE`. If one does not, `GenerateQwenVL`'s CPU fallback is correct and slower; record which. **O0 expects a
  split:** decode past the image needs only the scalar pairwise rope, which exists; the image-block *prefill* kernels
  are NeoX-only, so that prefill runs on the CPU (6,144 rows through a 0.58 B decoder) unless a pairwise variant of the
  kernel is written. Measure the CPU prefill first; it may be cheap enough to leave.
- **The tower:** CPU in v1, unless aikit's GPU vision path covers it cheaply. A GPU tower is a follow-on with
  its own band.

### O7 — docs and site

- Regenerate `capability-matrix.json` after O3. Add GLM-OCR to `multimodal.md`'s served-families paragraph
  and to the README's vision list.
- The site's Ollama-coverage row 22 changes from N to S. Per that snapshot's own rule, this needs a new dated
  snapshot, not an edit.
- A Models-page entry with O5's example and the O4 default stated.

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

