#!/usr/bin/env python
"""Build a tiny-random Gemma 3 vision-language checkpoint (mirrors the qwen35-tiny
approach) — the P2/P3 fixture for the VL loader + the "text path stays inert"
invariant.

Saves a small Gemma3ForConditionalGeneration (tiny SigLIP vision_config + tiny
Gemma3 text_config + the multimodal projector) as a real HF checkpoint, and pins a
TEXT-ONLY forward golden (no pixel_values). goinfer loads the text decoder from
this VL checkpoint (ignoring vision_tower / multi_modal_projector, like the
qwen3.6-VL text side) and must reproduce the golden bit-for-bit — proving a VL
checkpoint's text path equals today's causal path before any image seam fires.
(The image→logits end-to-end golden is added in P3 once the Go vision path lands.)

Tiny → sub-second; no download.

    ~/g4venv/bin/python scripts/pin_gemma3_vl_tiny.py
    -> testdata/gemma3_vl_tiny_text_golden.json
    -> testdata/gemma3-vl-tiny/   (HF safetensors checkpoint: vision + projector + text)
"""
import json
import os

import torch
from transformers import (Gemma3Config, Gemma3ForConditionalGeneration,
                          Gemma3TextConfig, SiglipVisionConfig)

OUT = os.path.join(os.path.dirname(__file__), "..", "testdata", "gemma3_vl_tiny_text_golden.json")

TEXT = dict(
    vocab_size=262,           # > image_token_index below
    hidden_size=64,
    intermediate_size=128,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=16,
    sliding_window=16,
    rms_norm_eps=1e-6,
    max_position_embeddings=64,
)
VISION = dict(
    hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=2,
    num_channels=3, image_size=32, patch_size=8, hidden_act="gelu_pytorch_tanh",
)
PROMPT = [2, 7, 42, 100, 5, 200, 13, 88]  # text-only, no image-soft-token
N_NEW = 6


def main():
    torch.manual_seed(0)
    cfg = Gemma3Config(
        text_config=Gemma3TextConfig(**TEXT),
        vision_config=SiglipVisionConfig(**VISION),
        mm_tokens_per_image=4,
        image_token_index=260,
    )
    model = Gemma3ForConditionalGeneration(cfg)
    model.eval().to(torch.float32)
    # HF inits mm_input_projection_weight to zeros, which makes the projector golden
    # degenerate (all-zero image_features); give it real values so parity is meaningful.
    with torch.no_grad():
        model.model.multi_modal_projector.mm_input_projection_weight.normal_(0.0, 0.02)

    with torch.no_grad():
        ids = torch.tensor([PROMPT], dtype=torch.long)
        out = model(input_ids=ids, use_cache=False)        # text-only: no pixel_values
        last_logits = out.logits[0, -1].float().tolist()
        cur, cont = list(PROMPT), []
        for _ in range(N_NEW):
            o = model(input_ids=torch.tensor([cur], dtype=torch.long), use_cache=False)
            cont.append(int(o.logits[0, -1].argmax()))
            cur.append(cont[-1])

    golden = {
        "note": "tiny-random Gemma3ForConditionalGeneration, TEXT-ONLY forward; CPU fp32",
        "text_config": TEXT,
        "vision_config": VISION,
        "image_token_index": 260,
        "mm_tokens_per_image": 4,
        "prompt_ids": PROMPT,
        "argmax": int(torch.tensor(last_logits).argmax()),
        "last_logits": last_logits,
        "n_new": N_NEW,
        "continuation_ids": cont,
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(golden, f)
    print(f"wrote {OUT}\n  argmax={golden['argmax']}  continuation={cont}")

    ckpt = os.path.join(os.path.dirname(OUT), "gemma3-vl-tiny")
    model.save_pretrained(ckpt, safe_serialization=True)
    print(f"saved checkpoint -> {ckpt}  (model_type={cfg.model_type!r})")


if __name__ == "__main__":
    main()
