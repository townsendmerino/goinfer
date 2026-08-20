#!/usr/bin/env python
"""Tiny-random Qwen3.8 (model_type qwen3_5) checkpoint + text golden — the dense sibling of
pin_qwen35_forward.py.

Pinned oracle: transformers 5.12.0 (the first release carrying models/qwen3_5). The parity
targets are the torch fallbacks (torch_recurrent_gated_delta_rule / torch_chunk_...), which is
what runs when the `fla` kernels are absent — as they are here.

The fixture keeps the released model's SHAPE CHARACTER, not its size:
  * head_dim (32) is NOT hidden/num_heads — the real model is head_dim 256 at hidden 5120 with
    24 heads, so nH·hd != hidden. A fixture that let them coincide would hide exactly the
    class of bug this family is prone to.
  * 3:1 layer_types (linear, linear, linear, full) so BOTH mixer kinds run.
  * GVA: linear_num_value_heads a multiple (>1x) of linear_num_key_heads.
  * partial_rotary_factor 0.25, mrope_section + mrope_interleaved present — the text path must
    reduce to plain partial RoPE despite them.

    ~/.venv-vl/bin/python scripts/pin_qwen3_5_forward.py
    -> decoder/testdata/qwen3_5_tiny_text_golden.json + decoder/testdata/qwen3_5-tiny/
"""
import json
import os
import sys

import torch
from transformers import (Qwen3_5ForCausalLM, Qwen3_5MoeForCausalLM,
                          Qwen3_5MoeTextConfig, Qwen3_5TextConfig)

HERE = os.path.dirname(os.path.abspath(__file__))
TD = os.path.join(HERE, "..", "decoder", "testdata")
OUT = os.path.join(TD, "qwen3_5_tiny_text_golden.json")
CKPT = os.path.join(TD, "qwen3_5-tiny")

# --moe writes a SECOND fixture, the MoE sibling — Qwen3_5MoeForCausalLM, its own model_type
# (goinfer's dense adapter rejects a num_experts config outright, and correctly so). It exists for one reason: the WebGPU resident
# bridge composes a DeltaNet mixer with a sparse-MoE FFN in the same layer, and nothing gated
# that pairing. The mixer alone is gated by the dense fixture, and mixer+MoE is gated for
# Mamba-2 (Granite) — but "two proven halves" is the argument that has been wrong here before
# (the A' zero-copy post-mortem: isolation proves the primitive, never the composition).
MOE_OUT = os.path.join(TD, "qwen3_5_moe_tiny_text_golden.json")
MOE_CKPT = os.path.join(TD, "qwen3_5_moe-tiny")
MOE_CFG = dict(
    num_experts=4, num_experts_per_tok=2, moe_intermediate_size=64,
    shared_expert_intermediate_size=64, norm_topk_prob=True, decoder_sparse_step=1,
    mlp_only_layers=[],
)

CFG = dict(
    vocab_size=256, hidden_size=64, intermediate_size=128, num_hidden_layers=4,
    num_attention_heads=4, num_key_value_heads=2, head_dim=32,   # nH*hd = 128 != hidden 64
    layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"],
    linear_conv_kernel_dim=4, linear_key_head_dim=16, linear_value_head_dim=16,
    linear_num_key_heads=2, linear_num_value_heads=4,            # GVA rep = 2
    rms_norm_eps=1e-6, max_position_embeddings=512, tie_word_embeddings=False,
    hidden_act="silu", attention_bias=False, attn_output_gate=True,
    rope_parameters={"rope_type": "default", "rope_theta": 10000000.0,
                     "partial_rotary_factor": 0.25,
                     "mrope_section": [3, 3, 2], "mrope_interleaved": True},
)
PROMPT = [2, 7, 42, 100, 5, 200, 13, 88]
N_NEW = 6


def main():
    moe = "--moe" in sys.argv
    cfg = dict(CFG, **MOE_CFG) if moe else CFG
    out, ckpt = (MOE_OUT, MOE_CKPT) if moe else (OUT, CKPT)
    torch.manual_seed(0)
    if moe:
        model = Qwen3_5MoeForCausalLM(Qwen3_5MoeTextConfig(**cfg))
    else:
        model = Qwen3_5ForCausalLM(Qwen3_5TextConfig(**cfg))
    model = model.eval().to(torch.float32)
    with torch.no_grad():
        ids = torch.tensor([PROMPT], dtype=torch.long)
        last = model(input_ids=ids, use_cache=False).logits[0, -1].float().tolist()
        cur, cont = list(PROMPT), []
        for _ in range(N_NEW):
            o = model(input_ids=torch.tensor([cur], dtype=torch.long), use_cache=False)
            cont.append(int(o.logits[0, -1].argmax()))
            cur.append(cont[-1])

    json.dump({
        "note": "tiny-random Qwen3_5ForCausalLM (dense DeltaNet/softmax hybrid), CPU fp32, "
                "transformers 5.12.0 torch fallback kernels",
        "config": {k: v for k, v in cfg.items()},
        "prompt_ids": PROMPT, "n_new": N_NEW,
        "argmax": int(torch.tensor(last).argmax()), "last_logits": last,
        "continuation_ids": cont,
    }, open(out, "w"))
    model.save_pretrained(ckpt, safe_serialization=True)
    print(f"argmax={int(torch.tensor(last).argmax())} cont={cont}")
    print(f" -> {out}\n -> {ckpt}")


if __name__ == "__main__":
    main()
