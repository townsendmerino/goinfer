#!/usr/bin/env python
"""Build a tiny-random GLM-4.5 (Glm4Moe) checkpoint + pin a text golden — the GLM
parity fixture (mirrors pin_qwen35_*/pin_qwen25vl_tiny). GLM-4.6 (355B) / 4.5-Air
(106B) won't fit a full bf16 oracle on the box (see the families plan), so the tiny
synthetic is the cheap descriptor+loader gate; weightDiff + a layer-slice oracle on
the real model come later.

Exercises every GLM-specific seam at once: a **dense** layer 0 (first_k_dense_replace)
+ **MoE** layers (routed experts FUSED+STACKED as `mlp.experts.gate_up_proj`, a shared
always-on expert), **partial RoPE** (partial_rotary_factor 0.5), **QK-norm**, and an
**MTP** (nextn) head that goinfer drops (loads only num_hidden_layers).

    ~/.venv-vl/bin/python scripts/pin_glm_tiny.py
    -> testdata/glm_tiny_text_golden.json
    -> testdata/glm-tiny/   (HF safetensors checkpoint)
"""
import json
import os

import torch
from transformers import Glm4MoeConfig, Glm4MoeForCausalLM

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "..", "testdata", "glm_tiny_text_golden.json")
CKPT = os.path.join(HERE, "..", "testdata", "glm-tiny")

CFG = dict(
    vocab_size=256, hidden_size=64, intermediate_size=128, num_hidden_layers=3,
    num_attention_heads=4, num_key_value_heads=2, head_dim=16,
    partial_rotary_factor=0.5, rope_theta=10000.0, max_position_embeddings=128,
    rms_norm_eps=1e-6, use_qk_norm=True, attention_bias=False,
    # DeepSeek-style MoE: dense layer 0, MoE 1+, shared expert, stacked routed experts
    n_routed_experts=8, num_experts_per_tok=2, n_shared_experts=1,
    moe_intermediate_size=32, first_k_dense_replace=1, norm_topk_prob=True,
    routed_scaling_factor=1.0, num_nextn_predict_layers=1,  # MTP head — dropped at load
    tie_word_embeddings=False,
)
PROMPT = [2, 7, 42, 100, 5, 200, 13, 88]
N_NEW = 6


def main():
    torch.manual_seed(0)
    model = Glm4MoeForCausalLM(Glm4MoeConfig(**CFG)).eval().to(torch.float32)
    # Random-init routers/experts give meaningful (non-degenerate) routing.
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
        "note": "tiny-random Glm4MoeForCausalLM, text forward; CPU fp32",
        "config": CFG, "prompt_ids": PROMPT,
        "argmax": int(torch.tensor(last_logits).argmax()),
        "last_logits": last_logits, "n_new": N_NEW, "continuation_ids": cont,
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(golden, f)
    print(f"wrote {OUT}\n  argmax={golden['argmax']}  continuation={cont}")
    model.save_pretrained(CKPT, safe_serialization=True)
    print(f"saved checkpoint -> {CKPT}  (model_type={Glm4MoeConfig(**CFG).model_type!r})")


if __name__ == "__main__":
    main()
