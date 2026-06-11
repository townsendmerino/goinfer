#!/usr/bin/env python
"""Pin the IMAGE→logits path of the tiny Gemma 3 VL checkpoint — the P3 end-to-end
gate (projector + embedding interleaving + bidirectional image-block mask).

Loads the SAME tiny checkpoint pin_gemma3_vl_tiny.py saved (testdata/gemma3-vl-tiny),
builds an input with an <image> placeholder run + pixel_values, and dumps two
stage-isolated goldens:
  - image_features: the projector output (vision encoder → pool → projector), the
    [mm_tokens_per_image, text_hidden] block that replaces the placeholders, and
  - last_logits: the end-to-end logits after interleaving + the bidirectional mask.
The Go side gates the projector on image_features and the full path on last_logits.

Tiny → sub-second; no download.

    ~/g4venv/bin/python scripts/pin_gemma3_vl_image.py
    -> testdata/gemma3_vl_tiny_image_golden.json
"""
import json
import os

import torch
from transformers import Gemma3ForConditionalGeneration

CKPT = os.path.join(os.path.dirname(__file__), "..", "testdata", "gemma3-vl-tiny")
OUT = os.path.join(os.path.dirname(__file__), "..", "testdata", "gemma3_vl_tiny_image_golden.json")
IMAGE_TOKEN = 260
MM_TOKENS = 4  # mm_tokens_per_image for the tiny config


def main():
    if not os.path.isdir(CKPT):
        raise SystemExit(f"{CKPT} missing — run scripts/pin_gemma3_vl_tiny.py first")
    torch.manual_seed(0)
    model = Gemma3ForConditionalGeneration.from_pretrained(CKPT, torch_dtype=torch.float32)
    model.eval()

    # input: [bos, t, t, <image>*MM_TOKENS, t, t]; the image tokens get replaced.
    prefix = [2, 7, 42]
    suffix = [13, 88]
    input_ids = prefix + [IMAGE_TOKEN] * MM_TOKENS + suffix
    img_start = len(prefix)

    gen = torch.Generator().manual_seed(1)
    cfg = model.config.vision_config
    pixel_values = torch.randn(1, cfg.num_channels, cfg.image_size, cfg.image_size,
                               generator=gen, dtype=torch.float32)

    with torch.no_grad():
        # Stage 1: projector output (vision tower → mm_soft_emb_norm → pool →
        # mm_input_projection). model.get_image_features returns the raw vision
        # wrapper in this transformers version, so run the two modules directly.
        vision_out = model.model.vision_tower(pixel_values=pixel_values).last_hidden_state
        image_features = model.model.multi_modal_projector(vision_out)  # [1, MM_TOKENS, text_hidden]
        # Stage 2: end-to-end logits (interleave + bidirectional image-block mask).
        out = model(input_ids=torch.tensor([input_ids], dtype=torch.long),
                    pixel_values=pixel_values, use_cache=False)
        last_logits = out.logits[0, -1].float().tolist()

    img_feat = image_features.reshape(-1).float().tolist()
    golden = {
        "note": "tiny Gemma3 VL image→logits; CPU fp32 (P3 end-to-end gate)",
        "input_ids": input_ids,
        "image_token_index": IMAGE_TOKEN,
        "image_token_start": img_start,
        "mm_tokens_per_image": MM_TOKENS,
        "pixel_values_shape": list(pixel_values.shape),
        "pixel_values": pixel_values.flatten().tolist(),
        "vision_last_hidden_state_shape": list(vision_out.shape),
        "vision_last_hidden_state": vision_out.reshape(-1).float().tolist(),  # projector INPUT (isolation)
        "image_features_shape": list(image_features.shape),
        "image_features": img_feat,           # projector output (stage gate)
        "argmax": int(torch.tensor(last_logits).argmax()),
        "last_logits": last_logits,           # end-to-end (final gate)
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(golden, f)
    print(f"wrote {OUT}")
    print(f"  image_features_shape={golden['image_features_shape']}  argmax={golden['argmax']}")


if __name__ == "__main__":
    main()
