# aikit task: Qwen2.5-VL vision encoder (second ViT family, dynamic resolution)

## Context

goinfer is adding **Qwen2.5-VL** as its second vision-language family (the first,
Gemma 3 / SigLIP, ships). The **decoder side is already done and gated** in goinfer:
the Qwen2.5-VL text decoder loads (it's Qwen2 + m-RoPE) and m-RoPE (`applyMRoPE` +
the `get_rope_index` port) is implemented and parity-gated vs HF. What's missing is
the **vision tower**, which lives in aikit (`aikit/vision`). This task adds it.

Unlike SigLIP (fixed 896×896 → 256 tokens, learned absolute pos, LayerNorm), the
Qwen2.5-VL ViT is **dynamic-resolution**: variable patch grids, 2D rotary position
embedding, RMSNorm, windowed+full attention, a gated MLP, and a spatial-merge patch
merger. **Dynamic resolution is in scope** (the model's native mode), so the encoder
must take `pixel_values` + `grid_thw` rather than a fixed square image.

Mirror the existing `aikit/vision/encoder.go` (SigLIP) conventions — `EncoderConfig`,
`LoadEncoder`, `Forward`, the `qmat` W8A8 wrapper, the resident-GPU seam — but add
the Qwen family **additively** (do not change SigLIP behavior or its gates).

## Reference

HF `transformers.models.qwen2_5_vl.modeling_qwen2_5_vl`:
`Qwen2_5_VisionTransformerPretrainedModel` (the `.visual` submodule). Implement to
match it in fp32. A pinned tiny fixture + the parity methodology are below.

## Architecture spec (the deltas vs SigLIP)

Config (HF `vision_config`), with the tiny test values in parens:
- `depth` (2) — num blocks; `hidden_size` (32); `intermediate_size` (64);
  `num_heads` (2) → head_dim = hidden/heads (16); `in_chans` (3);
  `patch_size` (14); `spatial_merge_size` (2); `temporal_patch_size` (2);
  `out_hidden_size` (64) — the decoder hidden the merger projects to;
  `window_size` (112, in pixels); `fullatt_block_indexes` ([1]) — block indices
  that use FULL attention (the rest use windowed); `hidden_act` (silu).

**Input** (NOT a CHW image — preprocessing is done upstream, see goinfer P5.3):
- `pixel_values`: `[n_patches, in_chans*temporal_patch_size*patch_size*patch_size]`
  (tiny: patch_dim = 3*2*14*14 = 1176). Patches are pre-flattened.
- `grid_thw`: `[][3]int` per image = (t, h, w) in **patch units** (h,w multiples of
  spatial_merge_size). `n_patches = Σ t*h*w`.

**Forward** (one or more images concatenated; per HF):
1. **Patch embed** — linear: `pixel_values @ patch_embed.proj` → `[n_patches, hidden]`.
   `patch_embed.proj.weight` is stored as a Conv3d `[hidden, in_chans, temporal,
   patch, patch]`; flatten to `[hidden, patch_dim]` and matmul (no bias).
2. **2D rotary position embedding** (`rot_pos_emb`): per patch, its (h_idx, w_idx)
   grid coords. head_dim/2 frequencies are split — head_dim/4 for height, head_dim/4
   for width — concatenated per patch, then the rotation covers the full head_dim
   (NeoX rotate_half). Build per the HF `rot_pos_emb` / `VisionRotaryEmbedding`
   (note the `spatial_merge_size` interleave when generating hpos/wpos ids).
3. **window_index + cu_seqlens**: patches are grouped into windows of
   `(window_size // patch_size // spatial_merge_size)` merged-units per side; reorder
   patches by window, build `cu_seqlens_window` (window boundaries) and
   `cu_seqlens_full` (per-image boundaries). Windowed blocks attend within a window;
   full-attention blocks (`fullatt_block_indexes`) attend across the whole image.
   Reorder back after the blocks (the merger output must be in original patch order).
4. **Blocks** (×depth), pre-norm residual:
   - `norm1` = **RMSNorm** (weight only, no bias; eps 1e-6).
   - **Attention**: `attn.qkv` is **fused** `[3*hidden, hidden]` (+bias) → split q/k/v
     per head; apply the 2D rotary to q,k; attend (softmax, scale `head_dim**-0.5`)
     within the block's cu_seqlens; `attn.proj` `[hidden, hidden]` (+bias).
   - residual add.
   - `norm2` = RMSNorm.
   - **Gated MLP**: `silu(gate_proj·x) * (up_proj·x)` → `down_proj`. gate/up
     `[inter, hidden]` (+bias), down `[hidden, inter]` (+bias).
   - residual add.
5. **Merger** (`visual.merger`): `ln_q` = **RMSNorm** over hidden; reshape each group
   of `spatial_merge_size²` patches into one `hidden*merge²` vector; `mlp.0`
   `[hidden*merge², hidden*merge²]` (+bias) → **GELU** → `mlp.2` `[out_hidden,
   hidden*merge²]` (+bias). Output `[n_patches/merge², out_hidden]` — the image
   embeddings that replace the decoder's `<image>` placeholders.

## Tensor names (prefix `visual.`; inside a real VL checkpoint it's `model.visual.`)

```
visual.patch_embed.proj.weight                 [hidden, in_chans, temporal, patch, patch]
visual.blocks.{i}.norm1.weight                 [hidden]                 # RMSNorm
visual.blocks.{i}.attn.qkv.weight / .bias      [3*hidden, hidden] / [3*hidden]
visual.blocks.{i}.attn.proj.weight / .bias     [hidden, hidden] / [hidden]
visual.blocks.{i}.norm2.weight                 [hidden]                 # RMSNorm
visual.blocks.{i}.mlp.gate_proj.weight/.bias   [inter, hidden] / [inter]
visual.blocks.{i}.mlp.up_proj.weight/.bias     [inter, hidden] / [inter]
visual.blocks.{i}.mlp.down_proj.weight/.bias   [hidden, inter] / [hidden]
visual.merger.ln_q.weight                      [hidden]                 # RMSNorm
visual.merger.mlp.0.weight / .bias             [hidden*merge², hidden*merge²]
visual.merger.mlp.2.weight / .bias             [out_hidden, hidden*merge²]
```

## API contract (what goinfer will call)

Add a Qwen encoder type with a Forward that takes pre-patchified input + grids:

```go
// QwenVisionEncoder is a loaded Qwen2.5-VL vision tower (dynamic resolution).
type QwenVisionEncoder struct { ... }

// LoadQwenVisionEncoder reads config.json (vision_config) + safetensors.
func LoadQwenVisionEncoder(dir string, quant bool) (*QwenVisionEncoder, error)

// Forward runs the ViT + merger on pre-patchified pixel_values [n_patches, patch_dim]
// with per-image grids (t,h,w in patch units), returning the merged image embeddings
// [n_merged_tokens, out_hidden_size] in original patch order — the embeddings that
// replace the decoder's <image> placeholders.
func (e *QwenVisionEncoder) Forward(pixelValues []float32, gridTHW [][3]int) ([]float32, error)
```

Keep the SigLIP `Encoder` untouched. A small family interface (or just two concrete
types goinfer dispatches on) is fine — match aikit's preference.

## Parity gate (the bar to merge)

Pin a tiny Qwen2.5-VL vision fixture with the HF rig (same approach as goinfer's
`scripts/pin_qwen25vl_image.py`, which already exists and works on transformers 5.12):

- Build a tiny `Qwen2_5_VLForConditionalGeneration`, run
  `model.model.get_image_features(pixel_values, grid_thw)`:
  - `.last_hidden_state` `[n_patches, hidden]` — the **ViT pre-merge** output.
  - `.pooler_output[0]` `[n_merged, out_hidden]` — the **merged** output.
- **Gate**: aikit `Forward` cosine ≥ 0.9999 (fp32) vs `.pooler_output[0]`, AND the
  pre-merge hidden ≥ 0.9999 vs `.last_hidden_state` (stage isolation catches whether
  a mismatch is in the blocks or the merger). Mirror SigLIP's `encoder_test.go`.

goinfer already has these goldens pinned for ITS tiny checkpoint
(`testdata/qwen25vl_tiny_image_golden.json`: `vit_hidden` [n_patches,hidden],
`image_features` [n_merged,out_hidden], plus the `pixel_values` + `grid_thw` inputs),
so a cross-repo cross-check is available if useful — but aikit should pin its own
fixture for a self-contained gate.

## Scope / non-goals

- **In scope**: dynamic resolution (variable grids, windowed+full attention, 2D
  rotary, the merger), fp32 CPU `Forward`, the loader, the parity gate.
- **Follow-on (don't block v1)**: W8A8 quant of the Qwen ViT (wire `qmat` like
  SigLIP once fp32 parity holds), the resident-GPU path (`ResidentEncoder`-style),
  multi-image batching beyond what `cu_seqlens` already gives, video (temporal>1
  beyond the patch dim).
- Preprocessing (image → pixel_values + grid_thw via smart-resize) lives in **goinfer
  P5.3**, not aikit — `Forward` takes pre-patchified input, as above.

## Deliverable

`LoadQwenVisionEncoder` + `Forward` matching HF in fp32, a pinned tiny fixture, a
parity test (cosine ≥ 0.9999 on both stages), and a one-line note of the aikit
version to tag so goinfer can bump its dependency (goinfer currently pins aikit
v1.7.3 with no replace directive).
