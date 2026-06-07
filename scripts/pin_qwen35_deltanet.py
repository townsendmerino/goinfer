#!/usr/bin/env python
"""Pin ONE Qwen3.5/3.6 Gated DeltaNet layer (conv + gated delta recurrence +
gated RMSNorm + out_proj) as an op-level parity golden for goinfer's primitive,
independent of the full-model weight loader.

The multi-token forward normally runs the chunked kernel; we monkeypatch it to
the exact per-step `torch_recurrent_gated_delta_rule` so goinfer's sequential
recurrence matches bit-for-bit (the two are mathematically equivalent; this
removes chunk-vs-recurrent fp drift from the gate). CPU fp32, torch fallback.

    ~/g4venv/bin/python scripts/pin_qwen35_deltanet.py
    -> testdata/qwen35_deltanet_golden.json
"""
import json
import os

import torch
from transformers import Qwen3_5MoeTextConfig
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeGatedDeltaNet,
    torch_recurrent_gated_delta_rule,
)

OUT = os.path.join(os.path.dirname(__file__), "..", "testdata", "qwen35_deltanet_golden.json")

HIDDEN, NK, NV, HK, HV, K = 32, 2, 4, 8, 8, 4
SEQ = 6


def flat(t):
    return t.detach().float().reshape(-1).tolist()


def main():
    cfg = Qwen3_5MoeTextConfig(
        hidden_size=HIDDEN, hidden_act="silu", rms_norm_eps=1e-6,
        linear_conv_kernel_dim=K,
        linear_key_head_dim=HK, linear_value_head_dim=HV,
        linear_num_key_heads=NK, linear_num_value_heads=NV,
        vocab_size=64, num_hidden_layers=1, num_attention_heads=4,
        num_key_value_heads=2, head_dim=8, layer_types=["linear_attention"],
    )
    torch.manual_seed(0)
    layer = Qwen3_5MoeGatedDeltaNet(cfg, layer_idx=0).eval().to(torch.float32)
    # Force the exact per-step recurrence for the multi-token forward (the chunk
    # call passes a cu_seqlens kwarg the recurrent fn doesn't take — drop it).
    def _recurrent(*args, **kw):
        kw.pop("cu_seqlens", None)
        return torch_recurrent_gated_delta_rule(*args, **kw)

    layer.chunk_gated_delta_rule = _recurrent

    torch.manual_seed(1)
    h = torch.randn(1, SEQ, HIDDEN, dtype=torch.float32)
    with torch.no_grad():
        out = layer(h)

    key_dim, value_dim = HK * NK, HV * NV
    golden = {
        "dims": {
            "hidden": HIDDEN, "num_k_heads": NK, "num_v_heads": NV,
            "head_k_dim": HK, "head_v_dim": HV, "conv_kernel": K,
            "key_dim": key_dim, "value_dim": value_dim,
            "conv_dim": 2 * key_dim + value_dim, "seq": SEQ, "rms_eps": 1e-6,
        },
        "weights": {
            "in_proj_qkv": flat(layer.in_proj_qkv.weight),  # [conv_dim, hidden]
            "in_proj_z": flat(layer.in_proj_z.weight),      # [value_dim, hidden]
            "in_proj_b": flat(layer.in_proj_b.weight),      # [num_v, hidden]
            "in_proj_a": flat(layer.in_proj_a.weight),      # [num_v, hidden]
            "conv1d_weight": flat(layer.conv1d.weight),     # [conv_dim, 1, K]; bias=False
            "dt_bias": flat(layer.dt_bias),                 # [num_v]
            "A_log": flat(layer.A_log),                     # [num_v]
            "norm_weight": flat(layer.norm.weight),         # [head_v_dim]
            "out_proj": flat(layer.out_proj.weight),        # [hidden, value_dim]
        },
        "input": flat(h),    # [seq, hidden]
        "output": flat(out),  # [seq, hidden]
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(golden, f)
    print(f"wrote {OUT}  (seq={SEQ}, hidden={HIDDEN}, out[0][:4]={golden['output'][:4]})")


if __name__ == "__main__":
    main()
