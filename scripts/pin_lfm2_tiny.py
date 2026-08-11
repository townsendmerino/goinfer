#!/usr/bin/env python
"""Build a tiny-random LFM2 (Lfm2ForCausalLM) checkpoint + text golden — the T1 full-model
LFM2 parity fixture (mirrors pin_granite_tiny). Exercises every LFM2 seam at once: a MIX of
short-conv + full-attention layers (layer_types), per-head Q/K RMSNorm, GQA, SwiGLU MLP with
the ff dim taken verbatim from intermediate_size (block_auto_adjust_ff_dim=False, matching
LFM2.5-2.6B), tied embeddings, and rope_theta set to the LFM2.5 value so a loader that drops
any of them fails parity.

    ~/.venv-vl/bin/python scripts/pin_lfm2_tiny.py
    -> testdata/lfm2_tiny_text_golden.json
    -> testdata/lfm2-tiny/   (HF safetensors checkpoint)

NOTE: needs transformers with the `lfm2` architecture (>= 4.55 for LFM2; the LFM2.5-2.6B
checkpoint declares transformers_version 5.2.0). If Lfm2Config rejects a kwarg, reconcile the
param names against the installed transformers `Lfm2Config` — this fixture mirrors the real
LFM2.5-2.6B config.json field names (layer_types, conv_L_cache, conv_bias, block_*, norm_eps,
rope_parameters). This is FREEZE-SAFE prep (new files only); the family itself is v1.0-gated
(see docs/scoping-lfm2.md §G).
"""
import json
import os

import torch
from transformers import Lfm2Config, Lfm2ForCausalLM

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "..", "testdata", "lfm2_tiny_text_golden.json")
CKPT = os.path.join(HERE, "..", "testdata", "lfm2-tiny")

# A tiny mix of short-conv and full-attention layers (LFM2.5-2.6B interleaves 22 conv / 8 attn;
# the fixture keeps the SAME two block kinds and the same layer_types mechanism at 4 layers).
CFG = dict(
    vocab_size=256, hidden_size=64, intermediate_size=128, num_hidden_layers=4,
    num_attention_heads=4, num_key_value_heads=2,  # GQA 2:1; head_dim = 64/4 = 16
    max_position_embeddings=128, norm_eps=1e-5,
    layer_types=["conv", "full_attention", "conv", "conv"],
    conv_L_cache=3, conv_bias=False,
    # ff dim taken verbatim from intermediate_size (toggle OFF) — matches the real checkpoint.
    block_auto_adjust_ff_dim=False, block_multiple_of=256, block_ffn_dim_multiplier=1.0,
    # LFM2.5 rope base (1e7) rather than a default 1e4, so a loader that hardcodes theta fails.
    rope_parameters={"rope_type": "default", "rope_theta": 10000000.0},
    tie_word_embeddings=True,
)
PROMPT = [2, 7, 42, 100, 5, 200, 13, 88]
N_NEW = 6


def main():
    torch.manual_seed(0)
    model = Lfm2ForCausalLM(Lfm2Config(**CFG)).eval().to(torch.float32)
    with torch.no_grad():
        ids = torch.tensor([PROMPT], dtype=torch.long)
        last_logits = model(input_ids=ids, use_cache=False).logits[0, -1].float().tolist()
        cur, cont = list(PROMPT), []
        for _ in range(N_NEW):
            o = model(input_ids=torch.tensor([cur], dtype=torch.long), use_cache=False)
            cont.append(int(o.logits[0, -1].argmax()))
            cur.append(cont[-1])

    golden = {
        "note": "tiny-random Lfm2 (conv+attn hybrid), text forward; CPU fp32",
        "config": CFG, "prompt_ids": PROMPT,
        "argmax": int(torch.tensor(last_logits).argmax()),
        "last_logits": last_logits, "n_new": N_NEW, "continuation_ids": cont,
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(golden, f)
    print(f"wrote {OUT}\n  argmax={golden['argmax']}  continuation={cont}")
    model.save_pretrained(CKPT, safe_serialization=True)
    print(f"saved checkpoint -> {CKPT}")


if __name__ == "__main__":
    main()
