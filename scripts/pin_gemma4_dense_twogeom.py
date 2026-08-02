#!/usr/bin/env python
"""Pin a tiny-random DENSE Gemma 4 with TWO attention geometries + K=V on the global
layer — the Split-A fixture for GPU-resident bring-up (docs/task-gemma4-moe.md, 9a-P2).

The existing gemma4-moe goldens are MoE (enable_moe_block); the dense E2B golden is
uniform-geometry. Neither isolates the geometry seam's TWO-LIVE-VARIANT path together
with attention_k_eq_v. This one does, with NO MoE:

  - 2 layers: one sliding (local, head_dim=16) + one full (global, global_head_dim=512),
  - attention_k_eq_v=True so the GLOBAL layer derives V from K (V = v_norm(k_proj), a
    scale-less per-head RMSNorm, no RoPE) instead of a separate v_proj,
  - partial (proportional) rotary on the global layer, gelu-tanh GeGLU dense MLP,
  - final-logit softcap 30, √hidden embed scale, the 4-norm sandwich.

CPU fp32 — the resident runner must reproduce it under the 3% near-tie rule. The 16/512
split is deliberately harsher than the real 256/512 so no buffer size or dispatch dim can
coincide between the two geometries (a Split-A single-variable strength).

Same degeneracy guard as pin_gemma4_moe_forward: HF init leaves norms/scalars at identity,
so a bug applying them (x1) would not move the golden. strengthen() overrides them with a
seeded SEPARATE torch.Generator, so the linear weights + input stay bit-identical and the
golden genuinely pins the norm / layer_scalar paths. The HF forward is the oracle.

    ~/.venv-vl/bin/python scripts/pin_gemma4_dense_twogeom.py
    -> testdata/gemma4_dense_twogeom_golden.json  (+ testdata/gemma4-dense-twogeom-tiny/)
"""
import json
import os

import torch
from transformers import Gemma4TextConfig
from transformers.models.gemma4.modeling_gemma4 import Gemma4ForCausalLM

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "..", "testdata", "gemma4_dense_twogeom_golden.json")
CKPT = os.path.join(HERE, "..", "testdata", "gemma4-dense-twogeom-tiny")

CFG = dict(
    vocab_size=256, hidden_size=64, num_hidden_layers=2,
    num_attention_heads=4, num_key_value_heads=2, head_dim=16,
    intermediate_size=48, rms_norm_eps=1e-6, tie_word_embeddings=True,
    max_position_embeddings=128, sliding_window=4,
    layer_types=["sliding_attention", "full_attention"],
    hidden_activation="gelu_pytorch_tanh", final_logit_softcapping=30.0,
    hidden_size_per_layer_input=0,  # PLE-free, like the 12B/26B
    rope_local_base_freq=10000.0, rope_theta=1000000.0,
    # DENSE (no MoE sub-block) — Split A isolates geometry + K=V, not the FFN delta.
    enable_moe_block=False,
    # TWO geometries + K=V on the global/full layer.
    global_head_dim=512, num_global_key_value_heads=2, attention_k_eq_v=True,
)
PROMPT = [1, 7, 42, 100, 5, 200, 13, 88]  # len 8 > sliding_window 4 (sliding layer clips)
N_NEW = 6


def strengthen(model):
    """Override identity/no-op scaling params with seeded non-trivial values.
    Separate generator => the linear weights + input stay bit-identical."""
    g = torch.Generator().manual_seed(1234)
    n_norm = n_ls = 0
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith(("layernorm.weight", "q_norm.weight", "k_norm.weight")) or name == "model.norm.weight":
                p.normal_(1.0, 0.1, generator=g)  # RMSNorm weights (were 1)
                n_norm += 1
        for name, b in model.named_buffers():
            if name.endswith("layer_scalar"):
                b.normal_(1.0, 0.1, generator=g)  # per-layer output scalar (was 1)
                n_ls += 1
    return dict(norm=n_norm, layer_scalar=n_ls)


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
        "note": "tiny-random DENSE gemma4, two attention geometries (local head_dim=16 / "
                "global head_dim=512) + attention_k_eq_v on the global layer; CPU fp32. "
                "Degenerate scaling params (norms / layer_scalar) strengthened with a seeded "
                "separate RNG so the golden pins those paths; HF forward is the oracle.",
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
    print(f"saved checkpoint -> {CKPT}  (model_type={config.model_type!r}, "
          f"enable_moe_block={config.enable_moe_block}, attention_k_eq_v={cfg_out['attention_k_eq_v']}, "
          f"global_head_dim={cfg_out['global_head_dim']})")


if __name__ == "__main__":
    main()
