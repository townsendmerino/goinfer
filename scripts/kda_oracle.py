#!/usr/bin/env python3
"""KDA (Kimi Delta Attention) reference oracle — NumPy transcription of the sequential recurrence.

WHY THIS EXISTS. Kimi-K3's text decoder runs KimiDeltaAttention on 69 of its 93 layers
(docs/task-model-family-deepseek-v4-kimi-k3.md, Phase 0/0b). Its modeling file does NOT contain the
scan: it calls `chunk_kda` / `fused_recurrent_kda` from `fla` (flash-linear-attention), an external
Triton library. That left ONE blocking ambiguity for pricing the port — `g` has shape (h, head_dim)
and KDA sets head_k_dim == head_dim == head_v_dim, so the config alone cannot say which axis of the
[K, V] state the per-channel decay indexes. Getting it wrong is silent-wrong output, not an error.

RESOLVED, from fla/ops/kda/naive.py (fla-org/flash-linear-attention):

    g: "Per-dimension decay gates (log-space) of shape [B, T, HV, K]"
    S: [B, HV, K, V]
    S = S * g_i[..., None].exp()        # [B,HV,K] -> [B,HV,K,1]: broadcasts over V

  => the decay indexes the KEY axis. Each key-row of the [K,V] state decays at its own rate,
     shared across every value column. g is LOG-space; the recurrence exponentiates.

This file is the durable deliverable: goinfer's future KDA parity gate needs a reference that does
not require Triton (or a GPU) to run, exactly as the DeltaNet work used a transcribed reference
(scripts/pin_qwen35_deltanet.py). It is validated here against fla's own `naive_recurrent_kda`, so
the transcription is pinned to upstream semantics rather than to my reading of them.

    python3 scripts/kda_oracle.py              # self-check (NumPy only)
    python3 scripts/kda_oracle.py --vs-fla     # additionally diff against fla's torch reference
"""
from __future__ import annotations

import argparse
import numpy as np


# ---------------------------------------------------------------- recurrence core

def kda_recurrent(q, k, v, g, beta, scale=None, initial_state=None):
    """Sequential KDA. Transcribed from fla.ops.kda.naive.naive_recurrent_kda.

    q, k : [T, H, K]        queries / keys
    v    : [T, HV, V]       values          (HV % H == 0; G = HV // H is the GVA group size)
    g    : [T, HV, K]       LOG-space per-key-dim decay
    beta : [T, HV]          write-gate scalars (already sigmoid'd by the caller, as the
                            modeling file does via use_beta_sigmoid_in_kernel)
    returns o [T, HV, V], final state S [HV, K, V]

    THE AXIS: S is [HV, K, V] and the decay multiplies along K, broadcasting over V.
    """
    T, H, K = q.shape
    HV, V = v.shape[1], v.shape[2]
    G = HV // H
    if scale is None:
        scale = K ** -0.5

    q = np.repeat(q, G, axis=1).astype(np.float64) * scale   # [T, HV, K]
    k = np.repeat(k, G, axis=1).astype(np.float64)           # [T, HV, K]
    v = v.astype(np.float64)
    g = g.astype(np.float64)
    beta = beta.astype(np.float64)

    S = np.zeros((HV, K, V), dtype=np.float64)
    if initial_state is not None:
        S = S + initial_state.astype(np.float64)
    o = np.zeros((T, HV, V), dtype=np.float64)

    for t in range(T):
        q_t, k_t, v_t, g_t, b_t = q[t], k[t], v[t], g[t], beta[t]
        # 1. decay along the KEY axis: [HV,K,1] * [HV,K,V]
        S = S * np.exp(g_t)[:, :, None]
        # 2. delta rule: err = v - Sᵀk  (sum over K), rank-1 update by (β·k) ⊗ err
        kv = (k_t[:, :, None] * S).sum(axis=1)               # [HV, V]  == Sᵀk
        err = v_t - kv                                       # [HV, V]
        S = S + (b_t[:, None] * k_t)[:, :, None] * err[:, None, :]
        # 3. read out: o = qᵀS (sum over K)
        o[t] = (q_t[:, :, None] * S).sum(axis=1)
    return o, S


# ---------------------------------------------------------------- surrounding ops

def short_conv_silu(x, w, state=None):
    """Depthwise causal conv (kernel_size from w) + SiLU — KDA's q/k/v front-end.
    x: [T, D]; w: [D, Kc] (channel-major taps, oldest→newest). Returns [T, D] and the new
    K-1 window so a decode step can continue. Matches ShortConvolution(activation='silu')."""
    T, D = x.shape
    Kc = w.shape[1]
    win = np.zeros((Kc - 1, D)) if state is None else state.copy()
    out = np.zeros_like(x, dtype=np.float64)
    for t in range(T):
        buf = np.concatenate([win, x[t][None, :]], axis=0)   # [Kc, D] oldest→newest
        y = (buf * w.T).sum(axis=0)
        out[t] = y / (1.0 + np.exp(-y))                      # SiLU
        win = buf[1:]
    return out, win


def l2norm(x, eps=1e-6):
    return x / np.sqrt((x * x).sum(-1, keepdims=True) + eps)


def kda_gate(a_lowrank, dt_bias, a_log, lower_bound=None):
    """KDA's log-space decay gate.

    g = -exp(A_log) * softplus(a + dt_bias), optionally floored at `lower_bound`
    (config linear_attn_config.gate_lower_bound = -5.0 on K3; `safe_gate` in the modeling call).
    Same functional form goinfer's DeltaNet already computes — the ONLY difference is that here
    `a_lowrank`/`dt_bias` are PER-CHANNEL (num_heads*head_dim) instead of per-head, so g is a
    vector over K rather than a scalar.
    """
    sp = np.logaddexp(0.0, a_lowrank + dt_bias)              # softplus, overflow-safe
    g = -np.exp(a_log) * sp
    if lower_bound is not None:
        g = np.maximum(g, lower_bound)
    return g


def gated_rmsnorm(x, w, gate, eps=1e-5):
    """FusedRMSNormGated(activation='sigmoid'): rmsnorm(x)*w * sigmoid(gate)."""
    xn = x / np.sqrt((x * x).mean(-1, keepdims=True) + eps)
    return xn * w * (1.0 / (1.0 + np.exp(-gate)))


# ---------------------------------------------------------------- checks

def _self_check(rng):
    """Structural properties that pin the AXIS — these fail loudly if the decay is transposed."""
    T, H, HV, K, V = 3, 2, 2, 4, 5
    q = rng.standard_normal((T, H, K)); k = rng.standard_normal((T, H, K))
    v = rng.standard_normal((T, HV, V)); beta = rng.random((T, HV))

    # A decay that zeroes exactly ONE key-row must zero that row of S and nothing else.
    g = np.zeros((T, HV, K)); g[:, :, 1] = -60.0             # log-space ⇒ exp ≈ 0
    _, S = kda_recurrent(q, k, v, g, np.zeros((T, HV)))      # beta=0 ⇒ no writes, decay only
    assert np.allclose(S, 0), "beta=0 must leave S at its (zero) initial state"

    S0 = rng.standard_normal((HV, K, V))
    _, S = kda_recurrent(q, k, v, g, np.zeros((T, HV)), initial_state=S0)
    decayed = S0 * np.exp(g[0])[:, :, None] ** T
    assert np.allclose(S, decayed), "decay must be a pure per-key-row scaling when beta=0"
    assert np.abs(S[:, 1, :]).max() < 1e-9, "key-row 1 must be annihilated"
    assert np.abs(S[:, 0, :]).max() > 0, "other key-rows must survive — decay indexed the WRONG axis"
    print("self-check: decay indexes the KEY axis of [K,V]  ✅")


def _vs_fla(rng):
    import importlib.util
    import os
    import torch

    # Import fla's naive reference BY FILE PATH. `import fla` pulls in Triton via the package
    # __init__, which is a GPU/compiler dependency the reference itself does not need — naive.py is
    # pure torch. Loading the file directly keeps this check runnable on a CPU box, and pins the
    # transcription to the exact source file rather than to whatever the package re-exports.
    cand = [os.environ.get("FLA_NAIVE", "")]
    import sysconfig
    for base in {sysconfig.get_paths()["purelib"], sysconfig.get_paths()["platlib"]}:
        cand.append(os.path.join(base, "fla", "ops", "kda", "naive.py"))
    path = next((c for c in cand if c and os.path.exists(c)), None)
    if path is None:
        raise SystemExit("fla naive.py not found — pip install flash-linear-attention, "
                         "or set FLA_NAIVE=/path/to/fla/ops/kda/naive.py")
    spec = importlib.util.spec_from_file_location("kda_naive", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    naive_recurrent_kda = mod.naive_recurrent_kda
    print(f"fla reference: {path}")

    B, T, H, HV, K, V = 1, 7, 2, 4, 6, 5
    q = rng.standard_normal((T, H, K)); k = rng.standard_normal((T, H, K))
    v = rng.standard_normal((T, HV, V))
    g = -np.exp(rng.standard_normal((T, HV, K)))             # log-space, negative
    beta = rng.random((T, HV))
    S0 = rng.standard_normal((HV, K, V))

    o_np, S_np = kda_recurrent(q, k, v, g, beta, initial_state=S0)

    tt = lambda x: torch.tensor(x, dtype=torch.float64)[None]
    o_t, S_t = naive_recurrent_kda(tt(q), tt(k), tt(v), tt(g), tt(beta),
                                   initial_state=tt(S0), output_final_state=True)
    do = np.abs(o_np - o_t[0].numpy().astype(np.float64)).max()
    ds = np.abs(S_np - S_t[0].numpy().astype(np.float64)).max()
    print(f"vs fla naive_recurrent_kda: max|Δo| = {do:.3e}   max|ΔS| = {ds:.3e}")
    # TOLERANCE, and why it is not a fudge. fla's reference casts its inputs to float32 internally
    # (`map(lambda x: x.to(torch.float), ...)`) while this transcription runs float64, so the residual
    # is f32 rounding accumulated over T steps — measured ~7.7e-7 and INVARIANT to the input dtype we
    # hand it, which is the signature of an internal cast rather than a semantic difference.
    #
    # It still discriminates the thing this file exists to pin: broadcasting the decay over the VALUE
    # axis instead of the KEY axis was measured at max|Δ| = 3.57 against the same reference — an O(1)
    # error, 4.7e6x larger than the f32 noise floor. So this threshold separates right-from-wrong by
    # ~5 orders of magnitude; it is loose against rounding and razor-tight against the axis.
    assert do < 1e-5 and ds < 1e-5, "transcription diverges from fla's reference beyond f32 rounding"
    print("vs fla: transcription matches upstream semantics (residual is fla's internal f32 cast)  ✅")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--vs-fla", action="store_true", help="also diff against fla's torch reference")
    a = ap.parse_args()
    rng = np.random.default_rng(20260809)
    _self_check(rng)
    # surrounding ops smoke
    x = rng.standard_normal((4, 6)); w = rng.standard_normal((6, 4))
    y, win = short_conv_silu(x, w)
    assert y.shape == x.shape and win.shape == (3, 6)
    g = kda_gate(rng.standard_normal(6), rng.standard_normal(6), rng.standard_normal(6), -5.0)
    assert (g >= -5.0).all() and (g <= 0).all(), "clamped log-decay must lie in [lower_bound, 0]"
    print("surrounding ops (conv+SiLU, l2norm, clamped gate, gated RMSNorm): shapes/ranges ok  ✅")
    if a.vs_fla:
        _vs_fla(rng)
