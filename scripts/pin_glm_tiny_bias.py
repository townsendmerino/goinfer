#!/usr/bin/env python
"""GLM tiny variant pinning the REAL GLM-4.5-Air flag combination the default tiny
doesn't exercise: attention_bias=True (q/k/v bias) + use_qk_norm=False. The default
pin_glm_tiny.py is the reverse (no bias, QK-norm). Together they cover both axes
against an HF reference, so a forward bug on the bias / no-QK-norm path is caught on
a tiny model (where a full oracle fits) rather than only suspected on the 106B.

    ~/.venv-vl/bin/python scripts/pin_glm_tiny_bias.py
    -> testdata/glm_tiny_bias_text_golden.json
    -> testdata/glm-tiny-bias/   (HF safetensors checkpoint)
"""
import json
import os

import torch
from transformers import Glm4MoeConfig, Glm4MoeForCausalLM

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "..", "testdata", "glm_tiny_bias_text_golden.json")
CKPT = os.path.join(HERE, "..", "testdata", "glm-tiny-bias")

CFG = dict(
    vocab_size=256, hidden_size=64, intermediate_size=128, num_hidden_layers=3,
    num_attention_heads=4, num_key_value_heads=2, head_dim=16,
    partial_rotary_factor=0.5, rope_theta=10000.0, max_position_embeddings=128,
    rms_norm_eps=1e-6, use_qk_norm=False, attention_bias=True,  # the real GLM-4.5-Air axes
    n_routed_experts=8, num_experts_per_tok=2, n_shared_experts=1,
    moe_intermediate_size=32, first_k_dense_replace=1, norm_topk_prob=True,
    routed_scaling_factor=1.0, num_nextn_predict_layers=1,
    tie_word_embeddings=False,
)
PROMPT = [2, 7, 42, 100, 5, 200, 13, 88]
N_NEW = 6


def main():
    torch.manual_seed(0)
    model = Glm4MoeForCausalLM(Glm4MoeConfig(**CFG)).eval().to(torch.float32)
    with torch.no_grad():
        ids = torch.tensor([PROMPT], dtype=torch.long)
        out = model(input_ids=ids, use_cache=False)
        last_logits = out.logits[0, -1].float().tolist()
        cur, cont = list(PROMPT), []
        for _ in range(N_NEW):
            o = model(input_ids=torch.tensor([cur], dtype=torch.long), use_cache=False)
            cont.append(int(o.logits[0, -1].argmax()))
            cur.append(cont[-1])

    golden = {
        "note": "tiny-random Glm4MoeForCausalLM (attention_bias=True, use_qk_norm=False); CPU fp32",
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
