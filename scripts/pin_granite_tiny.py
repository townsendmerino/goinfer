#!/usr/bin/env python
"""Build a tiny-random Granite-4.0-H (GraniteMoeHybrid) checkpoint + text golden —
the full-model Granite parity fixture (mirrors pin_glm_tiny). Exercises every Granite
seam at once: a MIX of mamba + attention layers (layer_types), MoE on every layer
(GraniteMoe fused experts + ungated shared_mlp), and the four Granite multipliers set
to NON-trivial values so a loader/forward that ignores them fails parity.

    ~/.venv-vl/bin/python scripts/pin_granite_tiny.py
    -> testdata/granite_tiny_text_golden.json
    -> testdata/granite-tiny/   (HF safetensors checkpoint)
"""
import json
import os

import torch
from transformers import GraniteMoeHybridConfig, GraniteMoeHybridForCausalLM

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "..", "testdata", "granite_tiny_text_golden.json")
CKPT = os.path.join(HERE, "..", "testdata", "granite-tiny")

CFG = dict(
    vocab_size=256, hidden_size=64, intermediate_size=128, num_hidden_layers=4,
    num_attention_heads=4, num_key_value_heads=2,
    mamba_n_heads=8, mamba_d_head=16, mamba_d_state=16, mamba_d_conv=4,
    mamba_n_groups=1, mamba_expand=2,
    num_local_experts=8, num_experts_per_tok=2, shared_intermediate_size=128,
    max_position_embeddings=128, rms_norm_eps=1e-6,
    layer_types=["mamba", "attention", "mamba", "mamba"],
    # Non-trivial Granite scalars — parity catches a forward that drops any of them.
    embedding_multiplier=12.0, attention_multiplier=0.5,
    residual_multiplier=0.22, logits_scaling=6.0,
    rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
    tie_word_embeddings=False,
)
PROMPT = [2, 7, 42, 100, 5, 200, 13, 88]
N_NEW = 6


def main():
    torch.manual_seed(0)
    model = GraniteMoeHybridForCausalLM(GraniteMoeHybridConfig(**CFG)).eval().to(torch.float32)
    with torch.no_grad():
        ids = torch.tensor([PROMPT], dtype=torch.long)
        last_logits = model(input_ids=ids, use_cache=False).logits[0, -1].float().tolist()
        cur, cont = list(PROMPT), []
        for _ in range(N_NEW):
            o = model(input_ids=torch.tensor([cur], dtype=torch.long), use_cache=False)
            cont.append(int(o.logits[0, -1].argmax()))
            cur.append(cont[-1])

    golden = {
        "note": "tiny-random GraniteMoeHybrid, text forward; CPU fp32",
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
