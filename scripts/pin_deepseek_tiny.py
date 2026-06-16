#!/usr/bin/env python
"""Tiny-random DeepSeek-V3 (DeepseekV3ForCausalLM) checkpoint + text golden — the
MLA parity fixture. Exercises the full strategic V3 path: Multi-head Latent Attention
(q-LoRA bottleneck + compressed KV latent + decoupled RoPE on the rope-carrying dims),
DeepSeekMoE (sigmoid scoring + e_score_correction_bias group-LIMITED routing + ungated
shared expert), and the first_k_dense_replace dense prefix. No YaRN (default RoPE) so the
attention scale is the plain qk_head_dim^-0.5.

    ~/.venv-vl/bin/python scripts/pin_deepseek_tiny.py
    -> testdata/deepseek_tiny_text_golden.json + testdata/deepseek-tiny/
"""
import json, os, torch
from transformers import DeepseekV3Config, DeepseekV3ForCausalLM
HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "..", "testdata", "deepseek_tiny_text_golden.json")
CKPT = os.path.join(HERE, "..", "testdata", "deepseek-tiny")
CFG = dict(
    vocab_size=128, hidden_size=64, intermediate_size=128, moe_intermediate_size=32,
    num_hidden_layers=4, first_k_dense_replace=1,
    num_attention_heads=4, num_key_value_heads=4,
    q_lora_rank=24, kv_lora_rank=16,
    qk_nope_head_dim=16, qk_rope_head_dim=8, v_head_dim=16,
    n_routed_experts=8, n_shared_experts=1, num_experts_per_tok=2,
    n_group=2, topk_group=1, routed_scaling_factor=2.5, norm_topk_prob=True,
    rms_norm_eps=1e-6, max_position_embeddings=64, tie_word_embeddings=False,
    rope_parameters={"rope_theta": 10000.0, "rope_type": "default"},
)
PROMPT = [2, 7, 42, 100, 5, 88, 13, 19]
N_NEW = 6


def main():
    torch.manual_seed(0)
    c = DeepseekV3Config(**CFG)
    print("=== config of interest ===")
    cd = c.to_dict()
    for k in ['rope_interleave', 'attention_bias', 'hidden_act', 'scoring_func',
              'topk_method', 'rope_parameters', 'q_lora_rank', 'kv_lora_rank',
              'qk_nope_head_dim', 'qk_rope_head_dim', 'v_head_dim', 'qk_head_dim']:
        print(f"  {k} = {cd.get(k, '<absent>')}")
    m = DeepseekV3ForCausalLM(c).eval().to(torch.float32)
    # Random e_score_correction_bias so the group-limited selection actually bites
    # (zeros would make bias a no-op and never exercise the bias path).
    with torch.no_grad():
        for layer in m.model.layers:
            if hasattr(layer.mlp, "gate"):
                layer.mlp.gate.e_score_correction_bias.copy_(
                    torch.randn_like(layer.mlp.gate.e_score_correction_bias) * 0.5)
        ids = torch.tensor([PROMPT])
        last = m(input_ids=ids, use_cache=False).logits[0, -1].float().tolist()
        cur, cont = list(PROMPT), []
        for _ in range(N_NEW):
            o = m(input_ids=torch.tensor([cur]), use_cache=False)
            cont.append(int(o.logits[0, -1].argmax())); cur.append(cont[-1])
    g = dict(note="tiny DeepSeekV3 (MLA) text fwd fp32", config=CFG, prompt_ids=PROMPT,
             argmax=int(torch.tensor(last).argmax()), last_logits=last,
             n_new=N_NEW, continuation_ids=cont)
    os.makedirs(os.path.dirname(OUT), exist_ok=True); json.dump(g, open(OUT, "w"))
    print(f"argmax={g['argmax']} cont={cont}")
    m.save_pretrained(CKPT, safe_serialization=True); print("saved", CKPT)


if __name__ == "__main__":
    main()
