#!/usr/bin/env python
"""Pin a SCALED Gemma-4 MoE fixture — the composition gate the C' expert cache never had.

WHY THIS EXISTS
===============
`TestGemma4MoE_cacheExpertsBitExact_scaled` has never run. It skips when
GOINFER_MOE_SCALED_FIXTURE is unset (it always was), and pointing it at the real 26B fails
structurally: that test needs BOTH arms resident, and the cache-OFF arm must hold ~11.4 GB of
experts in VRAM, which is the exact thing the cache exists to avoid. So C' bit-exactness has
only ever been gated at toy width (gemma4-moe-tiny: 2 layers, hidden 256, 4 experts).

That is the gap A' fell into. A' zero-copy was correct in isolation — mapped reads bit-exact at
K up to 4096 words, offsets to 64 MB, N to 4096 rows — and `gemv_w4a8_moe` still mis-read it in
the composed forward at width (255/256 logits wrong). The recorded lesson: "isolation proves the
primitive, never the composition." C' currently rests on the same class of evidence A' had when
it was wrong. This fixture is the composition gate.

TWO PROPERTIES MUST STAY REAL, AND ONLY ONE OF THEM IS A DIMENSION
==================================================================
1. GEOMETRY: hidden_size and moe_intermediate_size. THIS IS THE SENSITIVE AXIS.
   The A' post-mortem (docs/task-moe-streaming.md) rules out the others explicitly: "The
   divergence is not offset, K, occupancy, EXPERT COUNT, or the allocation/VRAM-layout change."
   It was measured both ways -- a many-expert/small-K fixture (32 experts, top-8, hidden 256)
   was 0/256 bit-exact, and hidden 2048 / moe_inter 768 was 255/256 wrong.

   So: SHRINK EXPERT COUNT AND LAYERS. NEVER SHRINK hidden OR moe_intermediate_size.
   The obvious-looking reduction -- keep 128 experts, cut hidden -- reproduces the configuration
   that already passed while broken. It would load, run, pass, and prove nothing.

2. WEIGHT DISTRIBUTION: per-group scale spread. Geometry alone is not enough.
   v0.9.0 shipped a compiler-fused multiply-add that caused 84% token-stream divergence on real
   models and was INVISIBLE on random-weight fixtures, "because uniform data rounds the same way
   in both orders; it only appears on real weights, whose group scales span orders of magnitude."
   A perfect-geometry fixture with random weights can reproduce that false negative exactly --
   one axis over from the one avoided above.

   Measured on the real 26B (per-group absmax, groups of 32 along K):

       tensor class                    log2 std   dynamic range
       real experts (gate_up, down)      0.30      13.2x  (3.7 octaves)
       real q_proj / embed_tokens        0.49      35-39x (5.1-5.3 octaves)
       HF random init normal(0, 0.02)    0.27       5.0x  (2.3 octaves)

   NOTE the correction: v0.9.0's "orders of magnitude" overstates it for the EXPERT weights,
   which span 13x, not >=100x. The requirement is still real -- random init is 5x, and the
   1.4-3 octave gap is where fused vs unfused rounding diverges -- but quote 13x, not "orders".

METHOD: empirical scale transplant (not distribution fitting)
============================================================
Because the geometry rule above forces hidden/moe_inter/head_dim/head counts to their REAL
values, every linear in this fixture has shapes IDENTICAL to the real 26B's -- only the expert
count, layer count and vocab shrink. So real per-row group-scale vectors can be imposed
directly rather than fitted:

    1. take random weights, reshape to [rows, K/32, 32]
    2. normalize each group to unit absmax
    3. multiply each group by a scale sampled from the REAL 26B's corresponding tensor,
       sampling WHOLE ROWS so within-row correlation (outlier output channels) survives

A fitted marginal would reproduce the histogram and destroy that correlation. This does not.

WHAT THIS FIXTURE DOES NOT COVER
================================
- HOST BUFFER SIZE. C's pinned host source here is 4 x 32 x 3.35 MB = 428 MB (381 MB weights +
  48 MB f16 scales), against ~11.4 GB weights-only on the real 26B -- a ~27x shortfall. Judged
  immaterial because the A' exclusion list rules out offset explicitly, and the mapped-read
  primitive was already proven bit-exact at 64 MB offsets, which this exceeds. Stated rather
  than left implicit: if a future failure turns out to be sensitive to total allocation size,
  this fixture cannot see it, and layer count is the cheap axis to grow (disk and CPU-reference
  time scale linearly with it).
- DEPTH-AMPLIFIED divergence. 4 layers, not 30. Safe here only because both gates compare
  bit-for-bit float32 LOGITS, where a single wrong expert read shows up immediately with no
  amplification needed. If either assertion is ever weakened to token identity, this depth
  reduction is no longer sound and must be revisited.
- TRAINED ROUTING. Scales are real; the weights under them are random, so expert selection is
  not a trained distribution. Cache HIT RATES from this fixture are meaningless. Bit-exactness
  is not, since it holds per-read regardless of which experts get picked.

DETERMINISTIC. Two seeds drive everything: torch.manual_seed(0) for the base weights and a
SEPARATE Generator(1234) for the scale sampling and the norm strengthening, so the transplant
cannot perturb the base draw. Verified by regenerating and comparing:

    model.safetensors sha256 = a56ed8bba8ca5125aacf325ab19b9492c5bec9b642227e97f86abc360a018154

That matters because these gates assert bit-identity: a fixture that drifted between machines
would make a real failure look like an environment difference, and an environment difference
look like a real failure. If the hash moves, the donor checkpoint or torch changed -- find out
which before trusting a red gate.

    ~/.venv-vl/bin/python scripts/pin_gemma4_moe_scaled.py
    -> testdata/gemma4-moe-scaled/            (~1.9 GB, bf16, gitignored)
"""
import json
import os

import torch
from safetensors import safe_open
from transformers import Gemma4TextConfig
from transformers.models.gemma4.modeling_gemma4 import Gemma4ForCausalLM

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT = os.path.join(HERE, "..", "testdata", "gemma4-moe-scaled")
REAL = os.path.expanduser("~/models/gemma-4-26b-a4b-it")

# REAL geometry, preserved exactly (the sensitive axis). Only n_experts/layers/vocab shrink.
HIDDEN, MOE_INTER, INTER = 2816, 704, 2112
N_EXPERTS, TOP_K, N_LAYERS, VOCAB = 32, 8, 4, 4096
# 3 sliding + 1 full: both attention geometries present, including the K=V global (head_dim 512).
LAYER_TYPES = ["sliding_attention"] * 3 + ["full_attention"]

CFG = dict(
    vocab_size=VOCAB, hidden_size=HIDDEN, num_hidden_layers=N_LAYERS,
    num_attention_heads=16, num_key_value_heads=8, head_dim=256,
    intermediate_size=INTER, rms_norm_eps=1e-6, tie_word_embeddings=True,
    max_position_embeddings=4096, sliding_window=8,  # prompt 16 > 8 => sliding layers clip
    layer_types=LAYER_TYPES,
    hidden_activation="gelu_pytorch_tanh", final_logit_softcapping=30.0,
    hidden_size_per_layer_input=0,  # PLE-free, like the real 26B
    rope_local_base_freq=10000.0, rope_theta=1000000.0,
    enable_moe_block=True, num_experts=N_EXPERTS, top_k_experts=TOP_K,
    moe_intermediate_size=MOE_INTER,
    global_head_dim=512, num_global_key_value_heads=2, attention_k_eq_v=True,
)


def tensor_class(name):
    """Map a parameter name to a scale-donor CLASS.

    Keying the pool by K alone is WRONG and the first run of this script proved it: expert
    gate_up, q_proj and the dense mlp gate/up all have K=2816, so a K-only pool let the wide
    attention rows dominate and the generated experts came out at 243x spread against a real
    26B target of ~20x. Over-dispersing is not "matching the empirical distribution" -- it makes
    a different fixture, one whose quantization error is dominated by an artefact.
    """
    if "experts.gate_up_proj" in name:
        return "expert_gate_up"
    if "experts.down_proj" in name:
        return "expert_down"
    if "embed_tokens" in name or "lm_head" in name:
        return "embed"
    if "router" in name:
        return "router"
    if "self_attn.o_proj" in name:
        return "attn_out"
    if "self_attn." in name:
        return "attn_in"
    if "mlp.down_proj" in name:
        return "mlp_out"
    if "mlp." in name:
        return "mlp_in"
    return "other"


def real_scale_pool(limit_rows=8192):
    """Per-row group-scale vectors from the REAL 26B, keyed by (class, K).

    Sampled across several layers, not just layer 0: a single layer understates the spread
    (layer 0 alone reads 13.2x on experts; layers 0/9/22 together read 20.5x).
    """
    idx = json.load(open(os.path.join(REAL, "model.safetensors.index.json")))["weight_map"]
    names = [k for k in idx if k.endswith(".weight") or k.endswith(("gate_up_proj", "down_proj"))]
    # language_model ONLY. The 26B ships as Gemma4ForConditionalGeneration, so 356 of its tensors
    # belong to the SigLIP vision tower (hidden 1152) -- a different model with a different scale
    # distribution. Left in, they polluted attn_in/attn_out with K=1152 donors.
    keep = [n for n in names if tensor_class(n) != "other" and "language_model" in n]
    # A few layers rather than all 30 -- enough for the spread, cheap to read. MUST include a
    # full_attention layer (the real 5:1 interleave puts them at 5, 11, 17, ...): the global
    # layers are the only source of K=8192 o_proj donors, and sampling 0/9/22 (all sliding) left
    # the fixture's own global layer with no donor.
    wanted_layers = {"0", "5", "9", "22", "23"}

    def layer_of(n):
        p = n.split(".")
        return p[p.index("layers") + 1] if "layers" in p else None

    keep = [n for n in keep if layer_of(n) is None or layer_of(n) in wanted_layers]
    pool, opened = {}, {}
    for name in keep:
        path = os.path.join(REAL, idx[name])
        f = opened.setdefault(path, safe_open(path, framework="pt"))
        sl = f.get_slice(name)
        shape = sl.get_shape()
        if len(shape) == 3:                       # [experts, rows, K] -- take a few experts
            w = torch.cat([sl[e:e + 1].to(torch.float32)[0] for e in (0, 31, 127) if e < shape[0]])
        else:
            w = sl[0:limit_rows].to(torch.float32)
        K = w.shape[-1]
        if K % 32:
            continue
        s = w.reshape(-1, K // 32, 32).abs().amax(-1)
        s = s[s.amin(-1) > 0][:limit_rows]
        if s.numel():
            pool.setdefault((tensor_class(name), K), []).append(s)
    out = {k: torch.cat(v) for k, v in pool.items()}
    for k in sorted(out):
        r = out[k]
        print(f"  donors {k[0]:15s} K={k[1]:5d}: {r.shape[0]:6d} rows  "
              f"log2std={torch.log2(r).std():.2f}  range={float(r.max() / r.min()):6.1f}x")
    return out


def transplant(name, w, pool, gen):
    """Impose REAL per-group scales on random weights, sampling whole donor rows.

    Whole rows, not independent groups: within-row correlation (an output channel that is
    uniformly hot) is structure a fitted marginal would destroy.
    """
    K = w.shape[-1]
    if K % 32:
        return None
    cls = tensor_class(name)
    donors = pool.get((cls, K))
    if donors is None:
        return None
    flat = w.reshape(-1, K // 32, 32)
    unit = flat / flat.abs().amax(-1, keepdim=True).clamp_min(1e-12)
    pick = torch.randint(0, donors.shape[0], (flat.shape[0],), generator=gen)
    w.copy_((unit * donors[pick].unsqueeze(-1)).reshape(w.shape))
    return cls


def strengthen(model, gen):
    """HF init leaves norms/scalars at identity, so a bug applying them would not move the
    output. Seeded separate generator, exactly as the other pinners."""
    n = 0
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith(("layernorm.weight", "q_norm.weight", "k_norm.weight")) or name == "model.norm.weight":
                p.normal_(1.0, 0.1, generator=gen)
                n += 1
        for name, b in model.named_buffers():
            if name.endswith("layer_scalar"):
                b.normal_(1.0, 0.1, generator=gen)
                n += 1
    return n


def main():
    torch.manual_seed(0)
    print(f"sampling real per-group scales from {REAL} ...")
    pool = real_scale_pool()

    config = Gemma4TextConfig(**CFG)
    model = Gemma4ForCausalLM(config).eval().to(torch.float32)

    gen = torch.Generator().manual_seed(1234)
    done, skipped = [], []
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.dim() < 2 or "norm" in name:
                continue
            (done if transplant(name, p.data, pool, gen) else skipped).append(name)
        n_norm = strengthen(model, gen)

    print(f"scale transplant: {len(done)} tensors matched a class+K donor, {len(skipped)} skipped")
    for s in skipped:
        print(f"  SKIPPED (no donor for this class+K): {s}")
    if skipped:
        raise SystemExit("refusing to write a fixture with untransplanted tensors -- a random-weight "
                         "tensor is exactly the false negative this fixture exists to avoid")
    print(f"strengthened {n_norm} degenerate norm/scalar params")

    # Verify achieved vs real target, per class. A fixture that silently missed its distribution
    # is the same class of defect as one that silently missed its geometry.
    print("achieved distribution (target = the donor row above):")
    seen = set()
    with torch.no_grad():
        for name, p in model.named_parameters():
            cls = tensor_class(name)
            if p.dim() < 2 or cls in seen or cls == "other":
                continue
            seen.add(cls)
            K = p.shape[-1]
            if K % 32:
                continue
            s = p.data.reshape(-1, K // 32, 32).abs().amax(-1).flatten()
            s = s[s > 0]
            tgt = pool.get((cls, K))
            t = (f"target log2std={torch.log2(tgt).std():.2f} range={float(tgt.max() / tgt.min()):.1f}x"
                 if tgt is not None else "no target")
            print(f"  {cls:15s} K={K:5d}: log2std={torch.log2(s).std():.2f} "
                  f"range={float(s.max() / s.min()):6.1f}x   {t}")

    os.makedirs(CKPT, exist_ok=True)
    model.to(torch.bfloat16).save_pretrained(CKPT, safe_serialization=True)
    print(f"saved -> {CKPT}  hidden={HIDDEN} moe_inter={MOE_INTER} experts={N_EXPERTS} "
          f"top_k={TOP_K} layers={N_LAYERS}")


if __name__ == "__main__":
    main()
