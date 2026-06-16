#!/usr/bin/env python
"""Tiny-random Kimi-K2-shaped checkpoint + text golden — the Kimi K2 parity fixture.
Kimi K2's architectures=DeepseekV3ForCausalLM: it IS DeepSeek-V3 (MLA + DeepSeekMoE), the
deltas vs V3 are config SCALARS (more routed experts, fewer heads), no new mechanism. So
this instantiates DeepseekV3ForCausalLM with a K2-flavored config — q-LoRA bottleneck,
SIGMOID noaux_tc routing with a real e_score_correction_bias, more experts + higher top-k,
n_group=1 (NO group limiting, like K2), routed_scaling_factor 2.827 — then patches the
saved config's model_type to "kimi_k2" so goinfer loads it through the kimi_k2 → deepseek
alias. Default RoPE (theta 50000); K2's YaRN is already validated by the real V2-Lite /
Moonlight deepseek gates.

    ~/.venv-vl/bin/python scripts/pin_kimi_tiny.py
    -> testdata/kimi_tiny_text_golden.json + testdata/kimi-tiny/ (model_type=kimi_k2)
"""
import json, os, torch
from transformers import DeepseekV3Config, DeepseekV3ForCausalLM
HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "..", "testdata", "kimi_tiny_text_golden.json")
CKPT = os.path.join(HERE, "..", "testdata", "kimi-tiny")
CFG = dict(
    vocab_size=128, hidden_size=256, intermediate_size=128, moe_intermediate_size=32,
    num_hidden_layers=4, first_k_dense_replace=1,
    num_attention_heads=64, num_key_value_heads=64,  # K2's actual head count (MHA, kv == heads); MLA decouples q from hidden via q_lora, so hidden need not divide heads
    q_lora_rank=24, kv_lora_rank=16,
    qk_nope_head_dim=16, qk_rope_head_dim=8, v_head_dim=16,
    n_routed_experts=16, n_shared_experts=1, num_experts_per_tok=4,  # K2-shaped: more experts, higher top-k
    n_group=1, topk_group=1, routed_scaling_factor=2.827, norm_topk_prob=True,
    rms_norm_eps=1e-6, max_position_embeddings=64, tie_word_embeddings=False,
    rope_parameters={"rope_theta": 50000.0, "rope_type": "default"},
)
PROMPT = [2, 7, 42, 100, 5, 88, 13, 19]
N_NEW = 6


def main():
    torch.manual_seed(0)
    c = DeepseekV3Config(**CFG)
    m = DeepseekV3ForCausalLM(c).eval().to(torch.float32)
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
    g = dict(note="tiny Kimi-K2-shaped (deepseek_v3 MLA) text fwd fp32", config=CFG,
             prompt_ids=PROMPT, argmax=int(torch.tensor(last).argmax()), last_logits=last,
             n_new=N_NEW, continuation_ids=cont)
    os.makedirs(os.path.dirname(OUT), exist_ok=True); json.dump(g, open(OUT, "w"))
    print(f"argmax={g['argmax']} cont={cont}")
    m.save_pretrained(CKPT, safe_serialization=True)
    # Patch model_type → kimi_k2 so goinfer resolves the kimi_k2 alias (HF wrote deepseek_v3).
    cfgp = os.path.join(CKPT, "config.json")
    cj = json.load(open(cfgp)); cj["model_type"] = "kimi_k2"; json.dump(cj, open(cfgp, "w"))
    print("saved", CKPT, "(model_type=kimi_k2)")


if __name__ == "__main__":
    main()
