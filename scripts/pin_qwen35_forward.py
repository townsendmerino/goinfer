#!/usr/bin/env python
"""Pin a tiny-random Qwen3.5/3.6-MoE (qwen3_5_moe) forward + greedy decode as a
goinfer parity golden.

Builds a SMALL random text model exercising both layer kinds (the 3:1
linear_attention / full_attention mix), the Gated DeltaNet recurrence (with GVA
v/k=2), per-head QK-norm attention, and the 4-expert + shared-expert MoE — runs
on CPU float32 with the torch fallback kernels (no fla / causal_conv1d), which is
exactly what goinfer must reproduce. Dumps the resolved arch-relevant config so
goinfer's loader reads identical values.

    ~/g4venv/bin/python scripts/pin_qwen35_forward.py
    -> testdata/qwen35_forward_golden.json
"""
import json
import os

import torch
from transformers import Qwen3_5MoeForCausalLM, Qwen3_5MoeTextConfig
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import torch_recurrent_gated_delta_rule

OUT = os.path.join(os.path.dirname(__file__), "..", "testdata", "qwen35_forward_golden.json")

# Tiny config: 4 layers as [linear, linear, linear, full] (the 3:1 pattern),
# GVA with linear_num_value_heads / linear_num_key_heads = 2, a 4-expert MoE.
CFG = dict(
    vocab_size=256,
    hidden_size=64,
    num_hidden_layers=4,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=16,
    hidden_act="silu",
    rms_norm_eps=1e-6,
    tie_word_embeddings=False,
    attention_bias=False,
    layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"],
    # Gated DeltaNet
    linear_conv_kernel_dim=4,
    linear_key_head_dim=16,
    linear_value_head_dim=16,
    linear_num_key_heads=2,
    linear_num_value_heads=4,
    # MoE (every layer)
    num_experts=4,
    num_experts_per_tok=2,
    moe_intermediate_size=32,
    shared_expert_intermediate_size=32,
    max_position_embeddings=64,
    rope_parameters={"rope_type": "default", "rope_theta": 1000000.0},
)


def main():
    torch.manual_seed(0)
    config = Qwen3_5MoeTextConfig(**CFG)
    model = Qwen3_5MoeForCausalLM(config)
    model.eval()
    model.to(torch.float32)

    # Force the exact per-step recurrence on every linear layer (drop the chunk
    # kernel's cu_seqlens kwarg) so the golden matches goinfer's sequential
    # DeltaNet bit-for-bit, not chunk-vs-recurrent within fp tolerance.
    def _recurrent(*args, **kw):
        kw.pop("cu_seqlens", None)
        return torch_recurrent_gated_delta_rule(*args, **kw)

    for layer in model.model.layers:
        if hasattr(layer, "linear_attn"):
            layer.linear_attn.chunk_gated_delta_rule = _recurrent

    prompt = [1, 7, 42, 100, 5, 200, 13, 88]
    n_new = 6

    with torch.no_grad():
        ids = torch.tensor([prompt], dtype=torch.long)
        out = model(ids, use_cache=False)
        last_logits = out.logits[0, -1].float().tolist()

        # Greedy decode (EOS not suppressed; tiny random model rarely hits it).
        cur = list(prompt)
        cont = []
        for _ in range(n_new):
            o = model(torch.tensor([cur], dtype=torch.long), use_cache=False)
            nxt = int(o.logits[0, -1].argmax())
            cont.append(nxt)
            cur.append(nxt)

    golden = {
        "note": "tiny-random qwen3_5_moe; CPU fp32, torch fallback kernels",
        "config": {k: getattr(config, k) for k in CFG},
        "prompt_ids": prompt,
        "argmax": int(torch.tensor(last_logits).argmax()),
        "last_logits": last_logits,  # full vocab (256) for cosine
        "n_new": n_new,
        "continuation_ids": cont,
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(golden, f, indent=2)
    print(f"wrote {OUT}")
    print(f"  argmax={golden['argmax']}  continuation={cont}")

    # Save the SAME random model as a real HF checkpoint so goinfer can load it
    # through its actual safetensors path and reproduce the golden.
    ckpt = os.path.join(os.path.dirname(OUT), "qwen35-tiny")
    model.save_pretrained(ckpt, safe_serialization=True)
    print(f"saved checkpoint -> {ckpt}  (config.model_type={config.model_type!r})")


if __name__ == "__main__":
    main()
