#!/usr/bin/env python
"""Tiny-random InternLM2 checkpoint + text golden.

THE FIXTURE EXISTS FOR ONE LINE OF CODE: the GROUPED de-interleave of attention.wqkv. HF's
InternLM2Attention reshapes wqkv to (..., num_kv_heads, 2 + num_kv_groups, head_dim) and takes
query = [..., :num_kv_groups, :], so the rows run  q q .. q k v | q q .. q k v | …  per KV
head — not phi3's [Q ‖ K ‖ V]. Read as a plain concat it yields correct shapes and finite
values with K rows standing in for query heads.

So the golden is built by CONSTRUCTING that layout from a reference llama's separate q/k/v and
running the reference forward on the llama — the two must agree only if goinfer's gather
inverts the interleave exactly. num_key_value_heads is deliberately > 1 and
num_attention_heads/num_key_value_heads > 1, or the interleave degenerates and the test would
pass against a plain concat.

    ~/.venv-vl/bin/python scripts/pin_internlm2_tiny.py
    -> decoder/testdata/internlm2_tiny_text_golden.json + decoder/testdata/internlm2-tiny/
"""
import json
import os

import torch
from safetensors.torch import save_file
from transformers import LlamaConfig, LlamaForCausalLM

HERE = os.path.dirname(os.path.abspath(__file__))
TD = os.path.join(HERE, "..", "decoder", "testdata")
OUT = os.path.join(TD, "internlm2_tiny_text_golden.json")
CKPT = os.path.join(TD, "internlm2-tiny")

H, I, L, NH, NKV, HD, V = 64, 128, 3, 8, 2, 8, 256   # groups = 4 → gs = 6
PROMPT = [2, 7, 42, 100, 5, 200, 13, 88]
N_NEW = 6


def main():
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=V, hidden_size=H, intermediate_size=I, num_hidden_layers=L,
                      num_attention_heads=NH, num_key_value_heads=NKV, head_dim=HD,
                      max_position_embeddings=128, rms_norm_eps=1e-5, rope_theta=1000000.0,
                      attention_bias=False, mlp_bias=False, tie_word_embeddings=False,
                      hidden_act="silu", rope_scaling={"rope_type": "dynamic", "factor": 2.0})
    ref = LlamaForCausalLM(cfg).eval().to(torch.float32)

    with torch.no_grad():
        ids = torch.tensor([PROMPT], dtype=torch.long)
        last = ref(input_ids=ids, use_cache=False).logits[0, -1].float().tolist()
        cur, cont = list(PROMPT), []
        for _ in range(N_NEW):
            o = ref(input_ids=torch.tensor([cur], dtype=torch.long), use_cache=False)
            cont.append(int(o.logits[0, -1].argmax()))
            cur.append(cont[-1])

    groups = NH // NKV
    tens = {
        "model.tok_embeddings.weight": ref.model.embed_tokens.weight.data.clone(),
        "model.norm.weight": ref.model.norm.weight.data.clone(),
        "output.weight": ref.lm_head.weight.data.clone(),
    }
    for i, lyr in enumerate(ref.model.layers):
        p = f"model.layers.{i}."
        q = lyr.self_attn.q_proj.weight.data      # [NH*HD, H]
        k = lyr.self_attn.k_proj.weight.data      # [NKV*HD, H]
        v = lyr.self_attn.v_proj.weight.data
        # Build the GROUPED fused tensor the way InternLM2 stores it.
        rows = []
        for g in range(NKV):
            rows.append(q[g * groups * HD:(g + 1) * groups * HD])
            rows.append(k[g * HD:(g + 1) * HD])
            rows.append(v[g * HD:(g + 1) * HD])
        tens[p + "attention.wqkv.weight"] = torch.cat(rows, dim=0).contiguous()
        tens[p + "attention.wo.weight"] = lyr.self_attn.o_proj.weight.data.clone()
        tens[p + "attention_norm.weight"] = lyr.input_layernorm.weight.data.clone()
        tens[p + "ffn_norm.weight"] = lyr.post_attention_layernorm.weight.data.clone()
        tens[p + "feed_forward.w1.weight"] = lyr.mlp.gate_proj.weight.data.clone()
        tens[p + "feed_forward.w3.weight"] = lyr.mlp.up_proj.weight.data.clone()
        tens[p + "feed_forward.w2.weight"] = lyr.mlp.down_proj.weight.data.clone()

    os.makedirs(CKPT, exist_ok=True)
    save_file({k: v.contiguous() for k, v in tens.items()},
              os.path.join(CKPT, "model.safetensors"), metadata={"format": "pt"})
    json.dump({
        "model_type": "internlm2", "architectures": ["InternLM2ForCausalLM"],
        "vocab_size": V, "hidden_size": H, "intermediate_size": I, "num_hidden_layers": L,
        "num_attention_heads": NH, "num_key_value_heads": NKV, "head_dim": HD,
        "max_position_embeddings": 128, "rms_norm_eps": 1e-5, "rope_theta": 1000000.0,
        "bias": False, "hidden_act": "silu", "tie_word_embeddings": False,
        "rope_scaling": {"type": "dynamic", "factor": 2.0},
    }, open(os.path.join(CKPT, "config.json"), "w"), indent=1)

    json.dump({
        "note": "tiny InternLM2: a reference llama's q/k/v re-packed into the GROUPED fused "
                "wqkv layout; the golden is the llama's own forward, so it agrees only if the "
                "de-interleave inverts exactly.",
        "prompt_ids": PROMPT, "n_new": N_NEW, "groups": groups,
        "argmax": int(torch.tensor(last).argmax()), "last_logits": last, "continuation_ids": cont,
    }, open(OUT, "w"))
    print(f"argmax={int(torch.tensor(last).argmax())} cont={cont} groups={groups} gs={2+groups}")
    print(f" -> {OUT}\n -> {CKPT}")


if __name__ == "__main__":
    main()
