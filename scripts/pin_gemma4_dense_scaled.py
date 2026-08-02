#!/usr/bin/env python
"""Pin a SCALED dense Gemma 4 — realistic dimensions, still no download, fits an 8 GB card
easily — to exercise the CUDA resident bridge at a scale the tiny Split-A fixture can't.

The Split-A fixture (pin_gemma4_dense_twogeom) is deliberately harsh-and-tiny: hidden=256,
2 layers, local head_dim=16. That isolated the geometry seam, but it left two gaps this
fixture closes:

  1. CONDITIONING. At hidden=256 the int4-vs-f32 noise floor is ~0.79 (chaotic), so the
     resident int4-vs-int4 gate is dominated by fixture quantization chaos, not kernel
     signal. At hidden=1024 the floor should be far higher, making the gate a real signal.
  2. REAL GEOMETRY. 256-local head_dim was NEVER exercised — the tiny fixture's local
     layer is hd=16 (a documented coverage gap in docs/gemma4-resident-scope.md). This
     uses the real 256 local / 512 global split.

Shape: hidden 1024, 12 layers in the real 5:1 sliding/full interleave ([s×5, f]×2), K=V on
the two global/full layers, head_dim 256 local / 512 global, GQA 4:2, gelu-tanh GeGLU dense
MLP, final-logit softcap 30, √hidden embed scale, the 4-norm sandwich. PLE-free
(hidden_size_per_layer_input=0), DENSE (no enable_moe_block). CPU fp32 — the resident runner
must reproduce it under the 3% near-tie rule.

Same degeneracy guard as the other pinners: HF init leaves norms/scalars at identity, so a
bug applying them (x1) would not move the golden. strengthen() overrides them via a seeded
SEPARATE torch.Generator so the linear weights + input stay bit-identical. HF forward is the
oracle. Weights are random ⇒ no coherent text is expected (the coherence gate applies only to
a real checkpoint).

    ~/.venv-vl/bin/python scripts/pin_gemma4_dense_scaled.py
    -> testdata/gemma4_dense_scaled_golden.json  (+ testdata/gemma4-dense-scaled/)
"""
import json
import os

import torch
from transformers import Gemma4TextConfig
from transformers.models.gemma4.modeling_gemma4 import Gemma4ForCausalLM

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "..", "testdata", "gemma4_dense_scaled_golden.json")
CKPT = os.path.join(HERE, "..", "testdata", "gemma4-dense-scaled")

# 5:1 sliding/full interleave over 12 layers → two global (full) layers at indices 5 and 11.
LAYER_TYPES = (["sliding_attention"] * 5 + ["full_attention"]) * 2

CFG = dict(
    vocab_size=256, hidden_size=1024, num_hidden_layers=12,
    num_attention_heads=4, num_key_value_heads=2, head_dim=256,  # local q_dim=1024, kv_dim=512
    intermediate_size=2048, rms_norm_eps=1e-6, tie_word_embeddings=True,
    max_position_embeddings=256, sliding_window=8,               # prompt len 16 > 8 ⇒ sliding layers clip
    layer_types=LAYER_TYPES,
    hidden_activation="gelu_pytorch_tanh", final_logit_softcapping=30.0,
    hidden_size_per_layer_input=0,  # PLE-free, like the 12B/26B dense
    rope_local_base_freq=10000.0, rope_theta=1000000.0,
    enable_moe_block=False,  # DENSE
    # TWO geometries + K=V on the global/full layers, REAL head dims (256 local / 512 global).
    global_head_dim=512, num_global_key_value_heads=2, attention_k_eq_v=True,
)
PROMPT = [1, 7, 42, 100, 5, 200, 13, 88, 3, 71, 128, 9, 250, 17, 60, 200]  # len 16 > sliding_window 8
N_NEW = 6


def strengthen(model):
    """Override identity/no-op scaling params with seeded non-trivial values.
    Separate generator => the linear weights + input stay bit-identical."""
    g = torch.Generator().manual_seed(1234)
    n_norm = n_ls = 0
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith(("layernorm.weight", "q_norm.weight", "k_norm.weight")) or name == "model.norm.weight":
                p.normal_(1.0, 0.1, generator=g)
                n_norm += 1
        for name, b in model.named_buffers():
            if name.endswith("layer_scalar"):
                b.normal_(1.0, 0.1, generator=g)
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

    cfg_out = {k: getattr(config, k) for k in CFG}
    cfg_out.update(
        model_type=config.model_type,
        global_head_dim=getattr(config, "global_head_dim", config.head_dim),
        num_global_key_value_heads=getattr(config, "num_global_key_value_heads", config.num_key_value_heads),
        attention_k_eq_v=getattr(config, "attention_k_eq_v", False),
        partial_rotary_factor=getattr(config, "partial_rotary_factor", 0.0),
    )
    golden = {
        "note": "SCALED dense gemma4: hidden 1024, 12 layers (5:1 sliding/full interleave), "
                "REAL head dims (local 256 / global 512) + attention_k_eq_v on the two global "
                "layers; CPU fp32. Random weights (no coherent text expected). Degenerate "
                "scaling params strengthened via a seeded separate RNG; HF forward is the oracle.",
        "strengthened": counts,
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
    print(f"  strengthened={counts}  argmax={golden['argmax']}  continuation={cont}")

    model.save_pretrained(CKPT, safe_serialization=True)
    print(f"saved checkpoint -> {CKPT}  (hidden={config.hidden_size}, layers={config.num_hidden_layers}, "
          f"head_dim={config.head_dim}/{cfg_out['global_head_dim']}, k_eq_v={cfg_out['attention_k_eq_v']})")


if __name__ == "__main__":
    main()
