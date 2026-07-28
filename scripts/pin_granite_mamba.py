#!/usr/bin/env python
"""Pin an isolated Granite-4.0 Mamba-2 mixer golden: one GraniteMoeHybridMambaLayer's
weights + a random input sequence + its output. Validates goinfer's mamba2 sequential
scan in ISOLATION (no full loader), the way deltanet's parity was first proven — the
"one genuinely new thing" for the Granite family.

    ~/.venv-vl/bin/python scripts/pin_granite_mamba.py
    -> testdata/granite_mamba_golden.json
"""
import json
import os

import torch
from transformers import GraniteMoeHybridConfig, GraniteMoeHybridForCausalLM

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "..", "testdata", "granite_mamba_golden.json")

CFG = dict(
    vocab_size=256, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
    num_attention_heads=4, num_key_value_heads=2,
    mamba_n_heads=8, mamba_d_head=16, mamba_d_state=16, mamba_d_conv=4,
    mamba_n_groups=2, mamba_expand=2,
    num_local_experts=8, num_experts_per_tok=2, shared_intermediate_size=128,
    max_position_embeddings=128, layer_types=["mamba", "mamba"],
)
SEQ = 12


def main():
    torch.manual_seed(0)
    cfg = GraniteMoeHybridConfig(**CFG)
    model = GraniteMoeHybridForCausalLM(cfg).eval().to(torch.float32)
    mixer = model.model.layers[0].mamba  # the Mamba-2 mixer module

    # Strengthen the golden: HF's default init leaves D=1, dt_bias=1, norm.weight=1 and
    # conv1d.bias=0 — all identity/no-op, so a bug in how goinfer *applies* these (× 1 or
    # + 0) would not move the reference output and the parity test would pass anyway.
    # Override them with seeded, non-trivial values using a SEPARATE generator, so the
    # global RNG stream (and thus the input x and every OTHER weight) stays bit-identical
    # to the prior fixture — only these four scaling params + the reference-recomputed
    # output change. The HF mixer forward below remains the independent oracle.
    g = torch.Generator().manual_seed(1234)
    mixer.D.data.uniform_(0.5, 2.0, generator=g)          # skip-connection scale (was 1)
    mixer.dt_bias.data.normal_(0.0, 0.2, generator=g)     # Δt bias, pre-softplus (was 1)
    mixer.norm.weight.data.normal_(1.0, 0.1, generator=g)  # gated RMSNorm weight (was 1)
    mixer.conv1d.bias.data.normal_(0.0, 0.1, generator=g)  # depthwise-conv bias (was 0)

    x = torch.randn(1, SEQ, cfg.hidden_size, dtype=torch.float32)
    with torch.no_grad():
        y = mixer(x)  # [1, seq, hidden]

    sd = {k: v for k, v in mixer.state_dict().items()}

    def t(name):
        return sd[name].float().flatten().tolist()

    golden = {
        "dims": dict(
            hidden=cfg.hidden_size, n_heads=cfg.mamba_n_heads, head_dim=cfg.mamba_d_head,
            d_state=cfg.mamba_d_state, n_groups=cfg.mamba_n_groups, d_conv=cfg.mamba_d_conv,
            d_inner=cfg.mamba_n_heads * cfg.mamba_d_head, seq=SEQ, rms_eps=cfg.rms_norm_eps,
        ),
        "weights": {
            "in_proj": t("in_proj.weight"),         # [2*d_inner + 2*ng*N + nheads, hidden]
            "conv1d_w": t("conv1d.weight"),          # [conv_dim, 1, K]
            "conv1d_b": t("conv1d.bias"),            # [conv_dim]
            "A_log": t("A_log"),                     # [nheads]
            "D": t("D"),                             # [nheads]
            "dt_bias": t("dt_bias"),                 # [nheads]
            "norm": t("norm.weight"),                # [d_inner]
            "out_proj": t("out_proj.weight"),        # [hidden, d_inner]
        },
        "input": x.flatten().tolist(),               # [seq*hidden]
        "output": y.flatten().tolist(),              # [seq*hidden]
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(golden, f)
    print(f"wrote {OUT}  (seq={SEQ}, d_inner={golden['dims']['d_inner']}, conv_dim={len(golden['weights']['conv1d_b'])})")


if __name__ == "__main__":
    main()
