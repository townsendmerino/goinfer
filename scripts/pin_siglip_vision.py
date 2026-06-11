#!/usr/bin/env python
"""Pin a tiny-random SigLIP vision encoder (the Gemma 3 vision tower) forward as a
goinfer parity golden — the P2 fixture for the vision-encoder descriptor.

Builds a SMALL random SiglipVisionModel (the same class Gemma 3's vision_tower is),
runs it on a fixed synthetic pixel_values tensor at CPU float32, and dumps
last_hidden_state. The Go vision encoder loads the SAME saved checkpoint + the SAME
pixel_values and must reproduce last_hidden_state (cosine ≈ 1.0). Tiny config →
sub-second; no model download.

Exercises the real structure goinfer must reproduce: Conv2d patch embedding,
learned position embeddings, N pre-LN transformer blocks (bidirectional MHA — NOT
causal — + gelu-tanh MLP), and the final post-layernorm.

    ~/g4venv/bin/python scripts/pin_siglip_vision.py
    -> testdata/siglip_vision_golden.json   (config + pixel_values + last_hidden_state)
    -> testdata/siglip-tiny/                 (HF safetensors checkpoint of the same weights)
"""
import json
import os

import torch
from transformers import SiglipVisionConfig, SiglipVisionModel

OUT = os.path.join(os.path.dirname(__file__), "..", "testdata", "siglip_vision_golden.json")

# Tiny config that still exercises every SigLIP component. image_size/patch_size
# give a 4×4 = 16-patch grid; 2 layers; gelu_pytorch_tanh (Gemma 3's SigLIP act).
CFG = dict(
    hidden_size=32,
    intermediate_size=64,
    num_hidden_layers=2,
    num_attention_heads=2,
    num_channels=3,
    image_size=32,
    patch_size=8,
    hidden_act="gelu_pytorch_tanh",
    layer_norm_eps=1e-6,
    attention_dropout=0.0,
)


def main():
    torch.manual_seed(0)
    config = SiglipVisionConfig(**CFG)
    model = SiglipVisionModel(config)
    model.eval()
    model.to(torch.float32)

    # Fixed synthetic image batch [1, 3, image_size, image_size], deterministic.
    gen = torch.Generator().manual_seed(1)
    pixel_values = torch.randn(1, CFG["num_channels"], CFG["image_size"], CFG["image_size"],
                               generator=gen, dtype=torch.float32)

    with torch.no_grad():
        out = model(pixel_values=pixel_values)
        last_hidden = out.last_hidden_state  # [1, num_patches, hidden]

    grid = CFG["image_size"] // CFG["patch_size"]
    golden = {
        "note": "tiny-random SiglipVisionModel (Gemma 3 vision tower); CPU fp32",
        "config": CFG,
        "num_patches": grid * grid,
        "pixel_values_shape": list(pixel_values.shape),
        "pixel_values": pixel_values.flatten().tolist(),
        "last_hidden_state_shape": list(last_hidden.shape),
        "last_hidden_state": last_hidden.flatten().float().tolist(),
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(golden, f)
    print(f"wrote {OUT}")
    print(f"  num_patches={golden['num_patches']}  last_hidden_state_shape={golden['last_hidden_state_shape']}")

    # Save the SAME random model as a real HF checkpoint so the Go vision encoder
    # loads it through the actual safetensors path and reproduces last_hidden_state.
    ckpt = os.path.join(os.path.dirname(OUT), "siglip-tiny")
    model.save_pretrained(ckpt, safe_serialization=True)
    print(f"saved checkpoint -> {ckpt}  (model_type={config.model_type!r})")


if __name__ == "__main__":
    main()
