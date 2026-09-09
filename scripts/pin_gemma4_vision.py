#!/usr/bin/env python
"""Pin a tiny-random Gemma 4 vision tower forward as a goinfer parity golden —
the P7 Phase A fixture for aikit/vision's Gemma4Encoder.

Builds a SMALL random Gemma4VisionModel + Gemma4MultimodalEmbedder (the same
classes the real "gemma4" family — E2B/E4B/26B-A4B/31B — uses; NOT the separate
"gemma4_unified" 12B family, which has no vision tower at all), runs it on a
fixed synthetic pre-patchified pixel_values + position_ids tensor at CPU
float32, and dumps the pooled+projected soft-token embeddings. The Go
Gemma4Encoder loads the SAME saved checkpoint + the SAME patches/position ids
and must reproduce the output (cosine ~= 1.0).

Deliberately sets use_clipped_linears=True (matching the real E2B checkpoint) so
the tiny golden exercises Gemma4ClippableLinear's real input/output clamp, not
just the harmless disabled case.

    ~/g4venv/bin/python scripts/pin_gemma4_vision.py
    -> testdata/gemma4_vision_golden.json   (config + patches + position_ids + soft tokens)
    -> testdata/gemma4-vision-tiny/          (HF safetensors checkpoint of the same weights)
"""
import json
import os
import types

import torch
from transformers.models.gemma4.configuration_gemma4 import Gemma4VisionConfig
from transformers.models.gemma4.modeling_gemma4 import Gemma4VisionModel, Gemma4MultimodalEmbedder

OUT = os.path.join(os.path.dirname(__file__), "..", "testdata", "gemma4_vision_golden.json")

# Tiny config exercising every component: patch_size=4, pooling_kernel_size=3 ->
# 12px effective patch; a 6x6=36-patch grid pools to 2x2=4 soft tokens.
# head_dim=8 (32/4, divisible by 4 for axial rope's two 4-wide rotate-half halves).
PATCH_SIZE = 4
POOLING_KERNEL_SIZE = 3
GRID = 6  # patches per side, must be a multiple of POOLING_KERNEL_SIZE
TEXT_HIDDEN = 24

CFG = dict(
    hidden_size=32,
    intermediate_size=64,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=4,
    head_dim=8,
    patch_size=PATCH_SIZE,
    pooling_kernel_size=POOLING_KERNEL_SIZE,
    position_embedding_size=16,
    rms_norm_eps=1e-6,
    use_clipped_linears=True,
    rope_parameters={"rope_theta": 100.0, "rope_type": "default"},
    standardize=False,
)


def main():
    torch.manual_seed(0)
    config = Gemma4VisionConfig(**CFG)
    model = Gemma4VisionModel(config)
    model.eval()
    model.to(torch.float32)

    text_config = types.SimpleNamespace(hidden_size=TEXT_HIDDEN)
    embedder = Gemma4MultimodalEmbedder(config, text_config)
    embedder.eval()
    embedder.to(torch.float32)

    # use_clipped_linears buffers default to +/-inf (identity clamp) unless a
    # checkpoint sets real bounds. Give them real, exercised finite bounds so
    # the golden actually tests the clamp path (matching the real E2B checkpoint,
    # which does NOT carry +/-inf here) — narrow enough that random activations
    # clip on both sides for at least a few tensors.
    gen = torch.Generator().manual_seed(2)
    for name, buf in model.named_buffers():
        if name.endswith("_min"):
            buf.copy_(torch.tensor(-2.0 + 0.1 * torch.randn(1, generator=gen).item()))
        elif name.endswith("_max"):
            buf.copy_(torch.tensor(2.0 + 0.1 * torch.randn(1, generator=gen).item()))

    num_patches = GRID * GRID
    patch_dim = 3 * PATCH_SIZE * PATCH_SIZE

    gen = torch.Generator().manual_seed(1)
    # [0,1]-range synthetic patches (the model does its own 2*(x-0.5) rescale).
    pixel_values = torch.rand(1, num_patches, patch_dim, generator=gen, dtype=torch.float32)

    xs, ys = torch.meshgrid(torch.arange(GRID), torch.arange(GRID), indexing="xy")
    # row-major (row,col) patch order: patch index = row*GRID + col.
    position_ids = torch.stack([xs.reshape(-1), ys.reshape(-1)], dim=-1).unsqueeze(0).long()

    with torch.no_grad():
        vis_out = model(pixel_values=pixel_values, pixel_position_ids=position_ids)
        pooled = vis_out.last_hidden_state  # [1, num_pooled, hidden]
        projected = embedder(pooled)  # [1, num_pooled, TEXT_HIDDEN]

    golden = {
        "note": "tiny-random Gemma4VisionModel+Gemma4MultimodalEmbedder (gemma4 plain family, NOT gemma4_unified); CPU fp32; use_clipped_linears=True with real finite bounds",
        "config": CFG,
        "text_hidden_size": TEXT_HIDDEN,
        "num_patches": num_patches,
        "patches_shape": list(pixel_values.shape),
        "patches": pixel_values.flatten().tolist(),
        "position_ids_shape": list(position_ids.shape),
        "position_ids": position_ids.flatten().tolist(),
        "pooled_shape": list(pooled.shape),
        "pooled": pooled.flatten().float().tolist(),
        "projected_shape": list(projected.shape),
        "projected": projected.flatten().float().tolist(),
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(golden, f)
    print(f"wrote {OUT}")
    print(f"  num_patches={num_patches}  pooled_shape={golden['pooled_shape']}  projected_shape={golden['projected_shape']}")

    ckpt = os.path.join(os.path.dirname(OUT), "gemma4-vision-tiny")
    model.save_pretrained(ckpt, safe_serialization=True)
    # embedder isn't itself a PreTrainedModel — save its state dict alongside,
    # under the SAME tensor names Gemma4Model.embed_vision would carry
    # ("embed_vision.embedding_projection.weight"), merged into the same file
    # goinfer's loader reads (embedding_pre_projection_norm has no weight —
    # with_scale=False — so there's nothing to save for it).
    from safetensors.torch import save_file, load_file
    st_path = os.path.join(ckpt, "model.safetensors")
    tensors = load_file(st_path)
    tensors["embed_vision.embedding_projection.weight"] = embedder.embedding_projection.weight.detach().clone()
    # Re-key the vision tower's own tensors under "vision_tower." — goinfer's
    # loader expects "vision_tower.patch_embedder..."/"vision_tower.encoder...",
    # matching the real checkpoint's "model.vision_tower...."/"model.embed_vision...."
    # nesting (this tiny fixture omits the "model." prefix — LoadGemma4Encoder's
    # tensorPrefix probe handles both).
    reprefixed = {}
    for k, v in tensors.items():
        if k.startswith("embed_vision."):
            reprefixed[k] = v
        else:
            reprefixed["vision_tower." + k] = v
    save_file(reprefixed, st_path)
    # config.json needs a nested text_config.hidden_size for LoadGemma4Encoder's
    # projector-width resolution.
    cfg_path = os.path.join(ckpt, "config.json")
    with open(cfg_path) as f:
        saved_cfg = json.load(f)
    saved_cfg["text_config"] = {"hidden_size": TEXT_HIDDEN}
    with open(cfg_path, "w") as f:
        json.dump(saved_cfg, f)
    print(f"saved checkpoint -> {ckpt}  (model_type={config.model_type!r})")


if __name__ == "__main__":
    main()
