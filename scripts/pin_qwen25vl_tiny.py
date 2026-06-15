#!/usr/bin/env python
"""Build a tiny-random Qwen2.5-VL checkpoint (mirrors pin_gemma3_vl_tiny.py) — the
P5 fixture for the second vision family's loader + the "text path stays inert"
invariant.

Saves a small Qwen2_5_VLForConditionalGeneration (tiny ViT vision_config + tiny
Qwen2.5-VL text_config with m-RoPE + the patch merger) as a real HF checkpoint, and
pins a TEXT-ONLY forward golden (no pixel_values). goinfer loads the text decoder
from this VL checkpoint and must reproduce the golden — proving a Qwen2.5-VL
checkpoint's text path (Qwen2 + m-RoPE) equals the causal path before any image
seam fires. The image->logits end-to-end golden is pin_qwen25vl_image.py.

Unlike Gemma 3 (fixed 896x896 / 256 tokens, standard RoPE), Qwen2.5-VL adds
m-RoPE (3D positions) + dynamic resolution (variable patch grids) — so this tiny
config exercises mrope_section, spatial_merge_size, and a windowed ViT.

Tiny -> sub-second; no download.

    ~/.venv-vl/bin/python scripts/pin_qwen25vl_tiny.py
    -> testdata/qwen25vl_tiny_text_golden.json
    -> testdata/qwen25vl-tiny/   (HF safetensors checkpoint: vision + merger + text)
"""
import json
import os

import torch
from transformers import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration
from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import (
    Qwen2_5_VLTextConfig, Qwen2_5_VLVisionConfig)

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "..", "testdata", "qwen25vl_tiny_text_golden.json")
CKPT = os.path.join(HERE, "..", "testdata", "qwen25vl-tiny")

# head_dim 16 -> mrope_section must sum to head_dim/2 = 8 (the real 3B: head_dim
# 128, mrope_section [16,24,24] summing to 64). out_hidden_size == text hidden so
# the merger output drops straight into the residual stream.
TEXT = dict(
    vocab_size=300, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
    num_attention_heads=4, num_key_value_heads=2, head_dim=16,
    max_position_embeddings=128, rms_norm_eps=1e-6,
    rope_theta=10000.0, rope_scaling={"type": "mrope", "mrope_section": [4, 2, 2]},
)
VISION = dict(
    depth=2, hidden_size=32, intermediate_size=64, num_heads=2, in_chans=3,
    patch_size=14, spatial_merge_size=2, temporal_patch_size=2, out_hidden_size=64,
    window_size=112, fullatt_block_indexes=[1],
)
IMAGE_TOKEN = 299
VISION_START = 298
PROMPT = [2, 7, 42, 100, 5, 200, 13, 88]  # text-only; no image / vision-start tokens
N_NEW = 6


def main():
    torch.manual_seed(0)
    cfg = Qwen2_5_VLConfig(
        text_config=Qwen2_5_VLTextConfig(**TEXT),
        vision_config=Qwen2_5_VLVisionConfig(**VISION),
        image_token_id=IMAGE_TOKEN, vision_start_token_id=VISION_START,
    )
    model = Qwen2_5_VLForConditionalGeneration(cfg).eval().to(torch.float32)

    with torch.no_grad():
        ids = torch.tensor([PROMPT], dtype=torch.long)
        out = model(input_ids=ids, use_cache=False)  # text-only: no pixel_values
        last_logits = out.logits[0, -1].float().tolist()
        cur, cont = list(PROMPT), []
        for _ in range(N_NEW):
            o = model(input_ids=torch.tensor([cur], dtype=torch.long), use_cache=False)
            cont.append(int(o.logits[0, -1].argmax()))
            cur.append(cont[-1])

    golden = {
        "note": "tiny-random Qwen2_5_VLForConditionalGeneration, TEXT-ONLY forward; CPU fp32",
        "text_config": TEXT, "vision_config": VISION,
        "image_token_id": IMAGE_TOKEN, "vision_start_token_id": VISION_START,
        "prompt_ids": PROMPT,
        "argmax": int(torch.tensor(last_logits).argmax()),
        "last_logits": last_logits, "n_new": N_NEW, "continuation_ids": cont,
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(golden, f)
    print(f"wrote {OUT}\n  argmax={golden['argmax']}  continuation={cont}")

    model.save_pretrained(CKPT, safe_serialization=True)
    print(f"saved checkpoint -> {CKPT}  (model_type={cfg.model_type!r})")


if __name__ == "__main__":
    main()
