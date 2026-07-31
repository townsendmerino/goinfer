#!/usr/bin/env python
"""Pin a tiny-random Gemma 4 26B-A4B-style MoE with K=V GLOBAL LAYERS
(attention_k_eq_v=True, num_global_key_value_heads=1) as a goinfer parity golden —
the Phase-4 variant of scripts/pin_gemma4_moe_forward.py.

The real Gemma 4 unified checkpoints (gemma4_unified_text: E2B/E4B/12B dense +
26B-A4B MoE) set attention_k_eq_v: the GLOBAL (full-attention) layers carry NO
v_proj — V is v_norm(k_proj output) — and use their own num_global_key_value_heads
(1) instead of the local num_key_value_heads. goinfer's loader marks those layers
VFromK; that path is untested by the plain MoE golden (v_proj on every layer), so
this golden exercises it end-to-end. Everything else matches the plain MoE golden:
2 layers (one sliding, one global), 8 experts top-2, gelu-tanh, softcap 30,
proportional global RoPE (partial_rotary_factor 0.25 nested in rope_parameters).

Degeneracy guard (same as the plain MoE pinner): the norm / router.scale /
per_expert_scale / layer_scalar identity params are strengthened with a seeded
separate RNG so the golden pins those paths; the HF forward stays the oracle.

    ~/.venv-vl/bin/python scripts/pin_gemma4_moe_kv_forward.py
    -> testdata/gemma4_moe_kv_forward_golden.json  (+ testdata/gemma4-moe-kv-tiny/ checkpoint)
"""
import json
import os

import torch
from transformers import Gemma4TextConfig
from transformers.models.gemma4.modeling_gemma4 import Gemma4ForCausalLM

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "..", "testdata", "gemma4_moe_kv_forward_golden.json")
CKPT = os.path.join(HERE, "..", "testdata", "gemma4-moe-kv-tiny")

CFG = dict(
    vocab_size=256, hidden_size=64, num_hidden_layers=2,
    num_attention_heads=4, num_key_value_heads=2, head_dim=16,
    intermediate_size=48, rms_norm_eps=1e-6, tie_word_embeddings=True,
    max_position_embeddings=128, sliding_window=4,
    layer_types=["sliding_attention", "full_attention"],
    hidden_activation="gelu_pytorch_tanh", final_logit_softcapping=30.0,
    hidden_size_per_layer_input=0,  # PLE-free, like the real 12B/26B
    rope_local_base_freq=10000.0, rope_theta=1000000.0,
    # K=V global layers: the global (full-attention) layer has NO v_proj (V is
    # v_norm(k)) and its own KV-head count. This is what the real 26B-A4B does.
    attention_k_eq_v=True, num_global_key_value_heads=1,
    enable_moe_block=True, num_experts=8, top_k_experts=2, moe_intermediate_size=16,
)
PROMPT = [1, 7, 42, 100, 5, 200, 13, 88]
N_NEW = 6


def strengthen(model):
    g = torch.Generator().manual_seed(1234)
    n_norm = n_router = n_pes = n_ls = 0
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith(("layernorm.weight", "q_norm.weight", "k_norm.weight")) or name == "model.norm.weight":
                p.normal_(1.0, 0.1, generator=g)
                n_norm += 1
            elif name.endswith("router.scale"):
                p.normal_(1.0, 0.1, generator=g)
                n_router += 1
            elif name.endswith("per_expert_scale"):
                p.uniform_(0.5, 1.5, generator=g)
                n_pes += 1
        for name, b in model.named_buffers():
            if name.endswith("layer_scalar"):
                b.normal_(1.0, 0.1, generator=g)
                n_ls += 1
    return dict(norm=n_norm, router_scale=n_router, per_expert_scale=n_pes, layer_scalar=n_ls)


def main():
    torch.manual_seed(0)
    config = Gemma4TextConfig(**CFG)
    model = Gemma4ForCausalLM(config).eval().to(torch.float32)
    counts = strengthen(model)

    # Sanity: the global layer must have no v_proj (V=v_norm(k)); the sliding layer keeps it.
    has_v = {i: (model.model.layers[i].self_attn.v_proj is not None) for i in range(config.num_hidden_layers)}
    assert has_v[1] is False, f"expected K=V (no v_proj) on the global layer, got {has_v}"

    with torch.no_grad():
        ids = torch.tensor([PROMPT], dtype=torch.long)
        last_logits = model(ids, use_cache=False).logits[0, -1].float().tolist()
        cur, cont = list(PROMPT), []
        for _ in range(N_NEW):
            nxt = int(model(torch.tensor([cur], dtype=torch.long), use_cache=False).logits[0, -1].argmax())
            cont.append(nxt)
            cur.append(nxt)

    cfg_out = {k: getattr(config, k) for k in CFG}
    cfg_out.update(
        model_type=config.model_type,
        global_head_dim=getattr(config, "global_head_dim", config.head_dim),
        num_global_key_value_heads=getattr(config, "num_global_key_value_heads", config.num_key_value_heads),
        attention_k_eq_v=getattr(config, "attention_k_eq_v", False),
    )
    golden = {
        "note": "tiny-random gemma4 enable_moe_block WITH K=V global layers "
                "(attention_k_eq_v, num_global_key_value_heads=1 → no v_proj on the global layer); "
                "CPU fp32. Scaling params strengthened (seeded separate RNG); HF forward is the oracle.",
        "strengthened": counts,
        "global_has_v_proj": has_v,
        "config": cfg_out,
        "prompt_ids": PROMPT,
        "argmax": int(torch.tensor(last_logits).argmax()),
        "last_logits": last_logits,
        "n_new": N_NEW,
        "continuation_ids": cont,
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(golden, f, indent=2)
    print(f"wrote {OUT}")
    print(f"  strengthened={counts}  has_v_proj={has_v}  argmax={golden['argmax']}  continuation={cont}")

    model.save_pretrained(CKPT, safe_serialization=True)
    print(f"saved checkpoint -> {CKPT}  (attention_k_eq_v={config.attention_k_eq_v})")


if __name__ == "__main__":
    main()
