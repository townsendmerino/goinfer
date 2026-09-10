#!/usr/bin/env python
"""Pin the IMAGE->logits path of the tiny Gemma 4 VL checkpoint (scripts/
pin_gemma4_vl_tiny.py) — the P7 end-to-end serving gate (tower + embed-by-vector
splice + sequential causal prefill).

Loads the SAME tiny checkpoint pin_gemma4_vl_tiny.py saved
(testdata/gemma4-vl-tiny), builds an input with an image-placeholder run +
patches/position_ids, and dumps two stage-isolated goldens:
  - image_features: the tower's pooled+projected output (the
    [n_image_tokens, text_hidden] block that replaces the placeholders), and
  - last_logits: the end-to-end logits after masked_scatter + the causal mask
    (use_bidirectional_attention is None on this fixture, matching E2B/E4B).
The Go side gates the decoder-side wiring on last_logits, given image_features
directly (isolating it from the vision tower itself, same shape as Gemma 3's
own pin_gemma3_vl_image.py).

Tiny -> sub-second; no download.

    ~/.venv-vl/bin/python scripts/pin_gemma4_vl_image.py
    -> testdata/gemma4_vl_tiny_image_golden.json
"""
import json
import os

import torch
from transformers.models.gemma4.modeling_gemma4 import Gemma4ForConditionalGeneration

HERE = os.path.dirname(__file__)
CKPT = os.path.join(HERE, "..", "testdata", "gemma4-vl-tiny")
OUT = os.path.join(HERE, "..", "testdata", "gemma4_vl_tiny_image_golden.json")
IMAGE_TOKEN_ID = 250  # within the tiny vocab (256); config's own default (258880) would OOB it


def main():
    if not os.path.isdir(CKPT):
        raise SystemExit(f"{CKPT} missing — run scripts/pin_gemma4_vl_tiny.py first")
    torch.manual_seed(0)
    model = Gemma4ForConditionalGeneration.from_pretrained(CKPT, dtype=torch.float32)
    model.eval()
    model.config.image_token_id = IMAGE_TOKEN_ID
    model.config.text_config.image_token_id = IMAGE_TOKEN_ID

    vcfg = model.config.vision_config
    patch_size, pool_k = vcfg.patch_size, vcfg.pooling_kernel_size
    grid = 6  # patches per side (multiple of pool_k) -> pools to 2x2=4 soft tokens
    n_image_tokens = (grid // pool_k) * (grid // pool_k)

    prefix = [2, 7, 42]
    suffix = [13, 88, 5]
    input_ids = prefix + [IMAGE_TOKEN_ID] * n_image_tokens + suffix
    img_start = len(prefix)

    gen = torch.Generator().manual_seed(1)
    num_patches = grid * grid
    patch_dim = 3 * patch_size * patch_size
    # [0,1]-range synthetic patches (the model does its own 2*(x-0.5) rescale) —
    # same convention as pin_gemma4_vision.py's isolated-tower golden.
    pixel_values = torch.rand(1, num_patches, patch_dim, generator=gen, dtype=torch.float32)
    xs, ys = torch.meshgrid(torch.arange(grid), torch.arange(grid), indexing="xy")
    position_ids = torch.stack([xs.reshape(-1), ys.reshape(-1)], dim=-1).unsqueeze(0).long()

    with torch.no_grad():
        # Stage 1: the tower's own pooled+projected output (isolation gate).
        image_features = model.model.get_image_features(pixel_values, position_ids, return_dict=True).pooler_output
        # Stage 2: end-to-end logits (masked_scatter splice + causal mask).
        out = model(
            input_ids=torch.tensor([input_ids], dtype=torch.long),
            pixel_values=pixel_values,
            image_position_ids=position_ids,
            use_cache=False,
        )
        last_logits = out.logits[0, -1].float().tolist()

    img_feat = image_features.reshape(-1).float().tolist()
    golden = {
        "note": "tiny Gemma4 VL image->logits; CPU fp32 (P7 end-to-end serving gate). "
                "use_bidirectional_attention is None on this fixture (E2B/E4B-class causal "
                "image-block attention), matching GenerateGemma4VL's v1 scope.",
        "input_ids": input_ids,
        "image_token_id": IMAGE_TOKEN_ID,
        "image_token_start": img_start,
        "n_image_tokens": n_image_tokens,
        "pixel_values_shape": list(pixel_values.shape),
        "position_ids_shape": list(position_ids.shape),
        "image_features_shape": list(image_features.shape),
        "image_features": img_feat,
        "argmax": int(torch.tensor(last_logits).argmax()),
        "last_logits": last_logits,
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(golden, f)
    print(f"wrote {OUT}")
    print(f"  n_image_tokens={n_image_tokens}  image_features_shape={golden['image_features_shape']}  argmax={golden['argmax']}")


if __name__ == "__main__":
    main()
