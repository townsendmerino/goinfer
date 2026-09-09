#!/usr/bin/env python
"""Build a tiny-random Qwen3-VL checkpoint (mirrors pin_qwen25vl_tiny.py) — the
P8 Phase 0 fixture for the TEXT decoder only. No vision tower, no DeepStack:
this pins the "goinfer can load a Qwen3-VL checkpoint and generate correct text"
invariant, the prerequisite before the image seam (vision tower + DeepStack +
interleaved m-RoPE over genuinely divergent positions) lands in a later phase.

Saves a small Qwen3VLForConditionalGeneration (tiny vision_config, unused by this
golden's forward + a tiny Qwen3-VL text_config with interleaved-layout m-RoPE) as
a real HF checkpoint, and pins a TEXT-ONLY forward golden (no pixel_values).
goinfer loads the text decoder from this VL checkpoint and must reproduce the
golden — proving qwen3_vlArchitecture (Qwen3's dense attention shape: per-head
q/k RMSNorm, GQA, no q/k/v bias) is wired correctly for a Qwen3-VL-nested config.

Every text position here has equal (t,h,w) components (no image token in the
prompt), so m-RoPE degenerates to scalar RoPE regardless of whether the
component lookup is chunked (Qwen2.5-VL) or interleaved (Qwen3-VL) — this golden
does NOT exercise mropeComponentInterleaved's divergent-position behavior; that
is covered separately by decoder/qwen3vl_test.go's direct, HF-verified table
test (no image path exists yet to exercise it end-to-end).

Tiny -> sub-second; no download.

    ~/.venv-vl/bin/python scripts/pin_qwen3vl_tiny.py
    -> testdata/qwen3vl_tiny_text_golden.json
    -> testdata/qwen3vl-tiny/   (HF safetensors checkpoint: vision + merger + text)
"""
import json
import os

import torch
from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration
from transformers.models.qwen3_vl.configuration_qwen3_vl import (
    Qwen3VLTextConfig, Qwen3VLVisionConfig)

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "..", "testdata", "qwen3vl_tiny_text_golden.json")
CKPT = os.path.join(HERE, "..", "testdata", "qwen3vl-tiny")

# head_dim 16 -> mrope_section must sum to head_dim/2 = 8 (the real defaults:
# head_dim 128, mrope_section [24,20,20] summing to 64). [4,2,2] is deliberately
# NOT symmetric across H/W so a future divergent-position test can distinguish
# the interleaved layout from the chunked one at this same section shape.
TEXT = dict(
    vocab_size=300, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
    num_attention_heads=4, num_key_value_heads=2, head_dim=16,
    max_position_embeddings=128, rms_norm_eps=1e-6,
    rope_parameters={"rope_theta": 10000.0, "mrope_section": [4, 2, 2]},
)
# Unused by this golden (text-only forward, no pixel_values) — present only so
# the saved checkpoint is a valid, loadable Qwen3VLForConditionalGeneration with
# model_type "qwen3_vl", matching how a real released checkpoint is shaped.
VISION = dict(
    depth=2, hidden_size=32, intermediate_size=64, num_heads=2, in_channels=3,
    patch_size=14, spatial_merge_size=2, temporal_patch_size=2, out_hidden_size=64,
    num_position_embeddings=16, deepstack_visual_indexes=[0, 1],
)
IMAGE_TOKEN = 299
VISION_START = 298
PROMPT = [2, 7, 42, 100, 5, 200, 13, 88]  # text-only; no image / vision-start tokens
N_NEW = 6


def main():
    torch.manual_seed(0)
    cfg = Qwen3VLConfig(
        text_config=Qwen3VLTextConfig(**TEXT),
        vision_config=Qwen3VLVisionConfig(**VISION),
        image_token_id=IMAGE_TOKEN, vision_start_token_id=VISION_START,
    )
    model = Qwen3VLForConditionalGeneration(cfg).eval().to(torch.float32)

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
        "note": "tiny-random Qwen3VLForConditionalGeneration, TEXT-ONLY forward; CPU fp32",
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
