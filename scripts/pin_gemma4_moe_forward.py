#!/usr/bin/env python
"""Pin a tiny-random Gemma 4 26B-A4B-style MoE (gemma4, enable_moe_block=true)
forward + greedy decode as a goinfer parity golden — the independent HF oracle for
the Phase-2 forward (docs/task-gemma4-moe.md).

Builds a SMALL random text model (Gemma4ForCausalLM on a Gemma4TextConfig) with the
parallel dense-MLP + 8-expert MoE FFN sub-block, one sliding + one full layer,
gelu-tanh GeGLU everywhere, final-logit softcap — CPU fp32, which goinfer must
reproduce. Dumps the resolved arch config so goinfer's loader reads identical values.

Degeneracy guard (same lesson as testdata/*_mamba/deltanet_golden): HF's default
init leaves EVERY gemma4-MoE scaling param at identity — norm weights = 1,
router.scale = 1, per_expert_scale = 1, layer_scalar = 1 — so a bug in how goinfer
*applies* them (× 1) would not move the golden. We override them with seeded,
NON-TRIVIAL values via a SEPARATE torch.Generator (the global RNG that drew the
linear weights + the input is untouched), so the golden genuinely pins the router
pre-norm/scale, the per-expert scale, layer_scalar, and the parallel-branch norms.
The HF forward stays the oracle.

    ~/.venv-vl/bin/python scripts/pin_gemma4_moe_forward.py
    -> testdata/gemma4_moe_forward_golden.json  (+ testdata/gemma4-moe-tiny/ checkpoint)
"""
import json
import os

import torch
from transformers import Gemma4TextConfig
from transformers.models.gemma4.modeling_gemma4 import Gemma4ForCausalLM

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "..", "testdata", "gemma4_moe_forward_golden.json")
CKPT = os.path.join(HERE, "..", "testdata", "gemma4-moe-tiny")

CFG = dict(
    vocab_size=256, hidden_size=64, num_hidden_layers=2,
    num_attention_heads=4, num_key_value_heads=2, head_dim=16,
    intermediate_size=48, rms_norm_eps=1e-6, tie_word_embeddings=True,
    max_position_embeddings=128, sliding_window=4,
    layer_types=["sliding_attention", "full_attention"],
    hidden_activation="gelu_pytorch_tanh", final_logit_softcapping=30.0,
    hidden_size_per_layer_input=0,  # PLE-free, like the 12B/26B
    rope_local_base_freq=10000.0, rope_theta=1000000.0,
    # uniform attention (no global-wide head / K=V) — that path is covered by the
    # dense gemma4 goldens; this golden isolates the NEW FFN sub-block.
    # the parallel dense + MoE FFN:
    enable_moe_block=True, num_experts=8, top_k_experts=2, moe_intermediate_size=16,
)
PROMPT = [1, 7, 42, 100, 5, 200, 13, 88]  # len 8 > sliding_window 4 (sliding layer clips)
N_NEW = 6


def strengthen(model):
    """Override identity/no-op scaling params with seeded non-trivial values.
    Separate generator ⇒ the linear weights + input stay bit-identical."""
    g = torch.Generator().manual_seed(1234)
    n_norm = n_router = n_pes = n_ls = 0
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith(("layernorm.weight", "q_norm.weight", "k_norm.weight")) or name == "model.norm.weight":
                p.normal_(1.0, 0.1, generator=g)  # RMSNorm weights (were 1)
                n_norm += 1
            elif name.endswith("router.scale"):
                p.normal_(1.0, 0.1, generator=g)  # learned [hidden] router pre-proj scale (was 1)
                n_router += 1
            elif name.endswith("per_expert_scale"):
                p.uniform_(0.5, 1.5, generator=g)  # learned [E] per-expert multiplier (was 1)
                n_pes += 1
        for name, b in model.named_buffers():
            if name.endswith("layer_scalar"):
                b.normal_(1.0, 0.1, generator=g)  # per-layer output scalar (was 1)
                n_ls += 1
    return dict(norm=n_norm, router_scale=n_router, per_expert_scale=n_pes, layer_scalar=n_ls)


def main():
    torch.manual_seed(0)
    config = Gemma4TextConfig(**CFG)
    model = Gemma4ForCausalLM(config).eval().to(torch.float32)
    counts = strengthen(model)

    with torch.no_grad():
        ids = torch.tensor([PROMPT], dtype=torch.long)
        last_logits = model(ids, use_cache=False).logits[0, -1].float().tolist()
        cur, cont = list(PROMPT), []
        for _ in range(N_NEW):
            nxt = int(model(torch.tensor([cur], dtype=torch.long), use_cache=False).logits[0, -1].argmax())
            cont.append(nxt)
            cur.append(nxt)

    # resolved arch config goinfer's loader/descriptor needs (superset of CFG).
    cfg_out = {k: getattr(config, k) for k in CFG}
    cfg_out.update(
        model_type=config.model_type,
        global_head_dim=getattr(config, "global_head_dim", config.head_dim),
        num_global_key_value_heads=getattr(config, "num_global_key_value_heads", config.num_key_value_heads),
        attention_k_eq_v=getattr(config, "attention_k_eq_v", False),
        partial_rotary_factor=getattr(config, "partial_rotary_factor", 0.0),
    )
    golden = {
        "note": "tiny-random gemma4 enable_moe_block; CPU fp32. Degenerate scaling params "
                "(norms / router.scale / per_expert_scale / layer_scalar) strengthened with a "
                "seeded separate RNG so the golden pins those paths; HF forward is the oracle.",
        "strengthened": counts,
        "config": cfg_out,
        "prompt_ids": PROMPT,
        "argmax": int(torch.tensor(last_logits).argmax()),
        "last_logits": last_logits,   # full vocab (256) for cosine
        "n_new": N_NEW,
        "continuation_ids": cont,
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(golden, f, indent=2)
    print(f"wrote {OUT}")
    print(f"  strengthened={counts}  argmax={golden['argmax']}  continuation={cont}")

    model.save_pretrained(CKPT, safe_serialization=True)
    print(f"saved checkpoint -> {CKPT}  (model_type={config.model_type!r}, enable_moe_block={config.enable_moe_block})")


if __name__ == "__main__":
    main()
