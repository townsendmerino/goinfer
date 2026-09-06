#!/usr/bin/env python
"""F4 (docs/task-families-2026-09.md): KDA (Kimi Delta Attention) recurrence rehearsal.

Ling-3.0-tiny (inclusionAI, model_type "bailing_hybrid", BailingMoeV3ForCausalLM) interleaves
KDA and MLA layers 3:1 (layer_group_size=4). Phase 0 found MLA + the MoE router are pure
composition of goinfer's existing deepseekArchitecture primitives (same q/kv-LoRA MLA shape,
same noaux_tc sigmoid+bias+group-limited-topk+routed_scaling router, ungated shared expert) — but
KDA is genuinely new: verified against fla-org/flash-linear-attention's real source (not the HF
modeling file's paraphrase, which just calls into the opaque Triton kernel), its forget gate is
PER-CHANNEL (shape [..., H, K]) where goinfer's existing Gated DeltaNet (qwen3_5_moe,
decoder/deltanet.go's gatedDeltaNetStep) uses a single PER-HEAD SCALAR decay for the whole
[head_k_dim, head_v_dim] state block. Everything else — the beta write-gate, the delta-rule
update, the q/k L2-norm-in-kernel, the output accumulation — is structurally identical to Gated
DeltaNet's own recurrence.

This script is the oracle for that ONE new piece (the per-channel-gated delta rule + its
"safe_gate" lower-bound activation), not a full model bring-up: no real checkpoint, no MLA, no
MoE, no conv/o_norm wrapper (those are already-shipped primitives this rehearsal doesn't need to
re-prove). The two reference functions below are copied VERBATIM (MIT license, matching this
repo's own) from fla-org/flash-linear-attention's fla/ops/kda/{naive,gate}.py — fetched directly
from https://github.com/fla-org/flash-linear-attention, not reconstructed from a description —
because installing the `fla` package pulls in Triton, which needs a CUDA toolchain this Mac
doesn't have; the two functions themselves are plain PyTorch/einops with no such dependency.

    python3 scripts/pin_kda_rehearsal.py
    -> testdata/kda_rehearsal_golden.json
"""
import json
import os

import torch
import torch.nn.functional as F
from einops import rearrange

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "..", "testdata", "kda_rehearsal_golden.json")

# ---------------------------------------------------------------------------
# Verbatim from fla-org/flash-linear-attention, fla/ops/kda/gate.py (MIT license).
# ---------------------------------------------------------------------------


def naive_kda_lowerbound_gate(g, A_log=None, dt_bias=None, lower_bound=-5.0, output_dtype=torch.float32):
    H, _ = g.shape[-2:]
    g = g.float()
    if dt_bias is not None:
        g = g + dt_bias.view(H, -1)
    if A_log is not None:
        g = A_log.view(H, 1).float().exp() * g
    g = lower_bound * F.sigmoid(g)
    return g.to(output_dtype)


# ---------------------------------------------------------------------------
# Verbatim from fla-org/flash-linear-attention, fla/ops/kda/naive.py (MIT license),
# sequential/recurrent form only (naive_recurrent_kda) -- the chunked form
# (naive_chunk_kda) is a parallel-scan reformulation of the SAME recurrence and is
# out of scope for this rehearsal, same as goinfer's own gatedDeltaNetStep only
# implements the sequential form today.
# ---------------------------------------------------------------------------


def naive_recurrent_kda(q, k, v, g, beta, scale=None, initial_state=None, output_final_state=False):
    dtype = v.dtype
    B, T, H, K, HV, V = *q.shape, v.shape[2], v.shape[-1]
    G = HV // H
    if scale is None:
        scale = K ** -0.5

    q, k, v, g, beta = map(lambda x: x.to(torch.float), [q, k, v, g, beta])
    q = q.repeat_interleave(G, dim=2) * scale
    k = k.repeat_interleave(G, dim=2)

    S = k.new_zeros(B, HV, K, V).to(q)
    if initial_state is not None:
        S += initial_state
    o = torch.zeros_like(v)
    for i in range(0, T):
        q_i, k_i, v_i, g_i, b_i = q[:, i], k[:, i], v[:, i], g[:, i], beta[:, i]
        S = S * g_i[..., None].exp()
        S = S + torch.einsum('b h k, b h v -> b h k v', b_i[..., None] * k_i, v_i - (k_i[..., None] * S).sum(-2))
        o[:, i] = torch.einsum('b h k, b h k v -> b h v', q_i, S)
    if not output_final_state:
        S = None
    return o.to(dtype), S


def l2norm(x, eps=1e-6):
    return x * torch.rsqrt(x.pow(2).sum(-1, keepdim=True) + eps)


def main():
    torch.manual_seed(0)
    B, T, H, K, V = 1, 6, 2, 4, 4  # tiny: 1 batch, 6 timesteps, 2 heads, head_k_dim=4, head_v_dim=4

    q_raw = torch.randn(B, T, H, K)
    k_raw = torch.randn(B, T, H, K)
    v = torch.randn(B, T, H, V)
    raw_gate_in = torch.randn(B, T, H, K)  # the f_proj(hidden_states) output, pre-activation
    beta_logits = torch.randn(B, T, H)
    A_log = torch.log(torch.empty(H).uniform_(1, 16))
    dt_bias = torch.randn(H, K)
    lower_bound = -5.0

    # q/k L2-norm "in kernel" (use_qk_l2norm_in_kernel=True in the real call) + the qk_head_dim^-0.5
    # scale naive_recurrent_kda itself applies via its own `scale` default -- kept OUT of q_raw here
    # (matching HF's own division of labor: l2norm happens before the op call, not inside it).
    q = l2norm(q_raw)
    k = l2norm(k_raw)
    beta = beta_logits.sigmoid()
    g = naive_kda_lowerbound_gate(raw_gate_in, A_log=A_log, dt_bias=dt_bias, lower_bound=lower_bound)

    o, _ = naive_recurrent_kda(q, k, v, g, beta, output_final_state=False)

    golden = dict(
        note="F4 KDA recurrence rehearsal oracle (naive_kda_lowerbound_gate + naive_recurrent_kda, "
             "verbatim from fla-org/flash-linear-attention, MIT). Not a full model -- see "
             "docs/task-families-2026-09.md F4.",
        shape=dict(B=B, T=T, H=H, K=K, V=V),
        lower_bound=lower_bound,
        q=q_raw.tolist(), k=k_raw.tolist(), v=v.tolist(),
        raw_gate_in=raw_gate_in.tolist(), beta_logits=beta_logits.tolist(),
        A_log=A_log.tolist(), dt_bias=dt_bias.tolist(),
        gate=g.tolist(),
        output=o.tolist(),
    )
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(golden, f)
    print(f"B={B} T={T} H={H} K={K} V={V}")
    print("output[0][-1] =", o[0, -1].tolist())
    print("wrote", OUT)


if __name__ == "__main__":
    main()
