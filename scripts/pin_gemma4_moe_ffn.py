#!/usr/bin/env python
"""Pin the Gemma 4 26B-A4B parallel dense+MoE FFN SUB-BLOCK in isolation as an
op-level parity golden — the deltanet-style oracle for goinfer's Phase-2 forward
(docs/task-gemma4-moe.md §A1), independent of the loader / attention / head stack.

Builds one tiny Gemma4TextDecoderLayer (enable_moe_block=true), zeroes its attention
(a stub self_attn ⇒ post-attention hidden == input), and runs the REAL HF layer
forward on a random hidden state. The output is therefore exactly the FFN sub-block
applied to the input, plus layer_scalar:

    x1 = post_feedforward_layernorm_1( mlp( pre_feedforward_layernorm(h) ) )
    _, w, idx = router(h)                                      # own weightless norm + scale
    x2 = post_feedforward_layernorm_2( experts( pre_feedforward_layernorm_2(h), idx, w ) )
    out = h + post_feedforward_layernorm(x1 + x2)
    out *= layer_scalar

Degeneracy guard (mamba/deltanet lesson): HF default init leaves norms / router.scale
/ per_expert_scale / layer_scalar at identity; strengthen() overrides them (seeded,
separate RNG) so the golden pins those paths. The HF layer forward stays the oracle.

    ~/.venv-vl/bin/python scripts/pin_gemma4_moe_ffn.py
    -> testdata/gemma4_moe_ffn_golden.json
"""
import json
import os

import torch
from transformers import Gemma4TextConfig
from transformers.models.gemma4.modeling_gemma4 import Gemma4ForCausalLM

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "..", "testdata", "gemma4_moe_ffn_golden.json")

HIDDEN, DENSE_INTER, MOE_INTER, N_EXPERTS, TOP_K = 64, 48, 16, 8, 2
SEQ, EPS = 5, 1e-6


def flat(t):
    return t.detach().float().reshape(-1).tolist()


def strengthen(layer):
    g = torch.Generator().manual_seed(1234)
    with torch.no_grad():
        for name, p in layer.named_parameters():
            if "layernorm" in name and name.endswith(".weight"):  # catches layernorm, _1, _2
                p.normal_(1.0, 0.1, generator=g)
            elif name.endswith("router.scale"):
                p.normal_(1.0, 0.1, generator=g)
            elif name.endswith("per_expert_scale"):
                p.uniform_(0.5, 1.5, generator=g)
        for name, b in layer.named_buffers():
            if name.endswith("layer_scalar"):
                b.normal_(1.0, 0.1, generator=g)


def main():
    cfg = Gemma4TextConfig(
        vocab_size=256, hidden_size=HIDDEN, num_hidden_layers=1,
        num_attention_heads=4, num_key_value_heads=2, head_dim=16,
        intermediate_size=DENSE_INTER, rms_norm_eps=EPS,
        hidden_activation="gelu_pytorch_tanh", hidden_size_per_layer_input=0,
        enable_moe_block=True, num_experts=N_EXPERTS, top_k_experts=TOP_K,
        moe_intermediate_size=MOE_INTER, layer_types=["full_attention"],
    )
    torch.manual_seed(0)
    # Build via the full model so the experts-implementation dispatch is set (a
    # standalone Gemma4TextDecoderLayer leaves config._experts_implementation None → NaN).
    model = Gemma4ForCausalLM(cfg).eval().to(torch.float32)
    layer = model.model.layers[0]
    strengthen(layer)

    # Zero the attention so the post-attention hidden equals the input: the layer
    # output is then purely the FFN sub-block (+ layer_scalar). post_attention_layernorm
    # of a zero vector is zero (RMSNorm(0)=0), so residual+0 = residual = input.
    class _ZeroAttn(torch.nn.Module):
        def forward(self, hidden_states, **kw):
            return torch.zeros_like(hidden_states), None

    layer.self_attn = _ZeroAttn()

    torch.manual_seed(1)
    h = torch.randn(1, SEQ, HIDDEN, dtype=torch.float32)
    with torch.no_grad():
        out = layer(h, position_embeddings=None, attention_mask=None)
    out = out[0] if isinstance(out, tuple) else out

    r = layer  # aliases for weight extraction
    golden = {
        "note": "gemma4 enable_moe_block FFN sub-block in isolation (attention zeroed); "
                "HF Gemma4TextDecoderLayer forward is the oracle. Scaling params strengthened.",
        "dims": {"hidden": HIDDEN, "dense_inter": DENSE_INTER, "moe_inter": MOE_INTER,
                 "num_experts": N_EXPERTS, "top_k": TOP_K, "seq": SEQ, "rms_eps": EPS},
        "weights": {
            "pre_ffn_norm": flat(r.pre_feedforward_layernorm.weight),
            "mlp_gate": flat(r.mlp.gate_proj.weight),     # [dense_inter, hidden]
            "mlp_up": flat(r.mlp.up_proj.weight),         # [dense_inter, hidden]
            "mlp_down": flat(r.mlp.down_proj.weight),     # [hidden, dense_inter]
            "post_ffn_norm_1": flat(r.post_feedforward_layernorm_1.weight),
            "router_scale": flat(r.router.scale),         # [hidden]
            "router_proj": flat(r.router.proj.weight),    # [num_experts, hidden]
            "per_expert_scale": flat(r.router.per_expert_scale),  # [num_experts]
            "pre_ffn_norm_2": flat(r.pre_feedforward_layernorm_2.weight),
            "experts_gate_up": flat(r.experts.gate_up_proj),  # [E, 2*moe_inter, hidden]
            "experts_down": flat(r.experts.down_proj),        # [E, hidden, moe_inter]
            "post_ffn_norm_2": flat(r.post_feedforward_layernorm_2.weight),
            "post_ffn_norm": flat(r.post_feedforward_layernorm.weight),  # joint
            "layer_scalar": flat(r.layer_scalar),         # [1]
        },
        "input": flat(h),    # [seq, hidden] (the post-attention residual)
        "output": flat(out),  # [seq, hidden]
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(golden, f)
    print(f"wrote {OUT}  (seq={SEQ}, out[0][:4]={golden['output'][:4]})")


if __name__ == "__main__":
    main()
