# Task: GLM-OCR — documents in, text and schema-bound JSON out (O0–O7) — 2026-10

> **Status: SCOPED 2026-10-01, unstarted.** O0 is reading only and can stop the task. Everything after
> it is written to be handed to a session as it stands, one gate at a time.
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

**Vision tower** (`vision_config`, `model_type: glm_ocr_vision`):
- depth 24, hidden 1024, 16 heads, intermediate 4096, `silu` gated MLP;
- patch 14, temporal patch 2, spatial merge 2, `out_hidden_size` 1536;
- RMSNorm eps 1e-5, attention bias true, full attention in every block.

GLM-4V's forward order with GLM-OCR's removals applied: `patch_embed` (Conv3d) → blocks → `post_layernorm` →
`downsample` (Conv2d, kernel and stride = merge size) → `merger`. The merger runs `proj` →
`post_projection_norm` (LayerNorm) → GELU → `down_proj(silu(gate_proj) * up_proj)`.

**Preprocessing** (`preprocessor_config.json`):
- `Glm46VImageProcessor`, patch 14, merge 2, temporal 2;
- CLIP mean and std;
- pixel bounds `shortest_edge: 12544`, `longest_edge: 9633792`.

That upper bound is about 9.6 megapixels: about 49,000 patches in full attention through the tower, and about
12,000 image tokens into the decoder after the 2 × 2 merge. O4 exists because of that number.

**Special tokens.** Image 59280, image start/end 59256/59257, video 59281 (unused here).

## 2. What goinfer has (checked at `1a1f3e10`)

| need | in the tree | gap |
|---|---|---|
| Image embeddings into the decoder | `forwardFromEmbed` on the generic path | none expected |
| m-RoPE positions, post-image delta, image-hash reuse | `GenerateQwenVL`, `mropePositions` / `mropeDelta` (`decoder/rope.go`) | GLM's `get_rope_index` must be diffed against them |
| Resident decode after an image | `decoder.ResidentMRoPE`, implemented by CUDA and now by Metal (`metal/backend.go`) | confirm GLM rides it, don't assume |
| Four norms per layer | `NormPlacement` `Sandwich4` (Gemma) | confirm GLM's order matches Gemma's |
| Fused `gate_up_proj` | split at load (`buildPhi3Weights`) | reuse the split |
| Pairwise rotation | `applyRoPEInterleaved` (`decoder/rope.go`) | **pairwise rotation and contiguous m-RoPE sections together** may not exist as one path |
| Vision tower | aikit `QwenVisionEncoder` (Qwen2.5-VL) is the nearest | new encoder; see O2 |
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

### O1 — the text decoder, `glm_ocr_text` (and `glm_ocr` as the wrapper `model_type`)

- One registration in `decoder/registry.go`. It is built from the existing parts in §2: GQA without bias,
  `gate_up_proj` split at load, `Sandwich4`, the rotary combination from O0.6, an untied head, and layer 16
  skipped at load.
- `scripts/pin_glm_ocr_tiny.py`: a random-weights checkpoint with the real text shape and a reduced layer
  count, text only.
- **Gate:** exact argmax and continuation against HF, text only. The rotary function is pinned directly
  against HF's slicing, the way `mropeComponentInterleaved` was.

### O2 — the tower in aikit

- A new encoder beside `QwenVisionEncoder`, not a flag on it. Relative to Qwen2.5-VL's tower it removes
  window attention, and it adds q/k RMSNorm, the `downsample` Conv2d, and GLM-4V's merger.
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

The processor's own ceiling (about 9.6 MP, about 12,000 image tokens) costs far more on a laptop than a
typical scan needs.
- Measure tower plus prefill time at 1, 2 and 4 MP on the Mac (CPU, plus Metal if O6 has it), and record what
  each does to the O3 outputs.
- Then propose a default cap for `serve` and `chat`, with a flag to raise it. Francis picks the default.

### O5 — structured extraction, the reason to do this

- GLM-OCR's information-extraction prompt with `response_format: json_schema` on `serve`. In
  `goinfer-chat`, `--schema` exists already (`internal/chatapp/main.go`) and `--image` does not; add it, so
  that `goinfer-chat --image invoice.png --schema invoice.schema.json` is the one-line demo.
- A worked example in `examples/`: an invoice image into a Go struct, through `constrain.GrammarFromStruct`.
- **Reported, not gated:** field-level accuracy on a small hand-labelled set of 10–20 documents. Schema
  validity needs no gate, because the grammar guarantees it. Report it with C1's per-field confidence where
  that applies.
- The same flow works in the `-web` UI through W11's Attach image. Record a short capture of it for the site.

### O6 — GPU paths (each on its own measurement; not blocking)

- **Decode after the image:** confirm whether the CUDA and Metal residents serve GLM-OCR through
  `ResidentMRoPE`. If one does not, `GenerateQwenVL`'s CPU fallback is correct and slower; record which.
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

<!-- doc-reviewed: 2026-10-01 -->
