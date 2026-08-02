#!/usr/bin/env python
"""Pin a tiny-random Gemma 4 26B-A4B-style MoE (gemma4, enable_moe_block=true)
forward + greedy decode as a goinfer parity golden — the independent HF oracle for
the Phase-2 forward (docs/task-gemma4-moe.md).

Builds a SMALL random text model (Gemma4ForCausalLM on a Gemma4TextConfig) with the
parallel dense-MLP + top-2-of-4-expert MoE FFN sub-block, one sliding + one full layer,
gelu-tanh GeGLU everywhere, final-logit softcap — CPU fp32, which goinfer must
reproduce. Dumps the resolved arch config so goinfer's loader reads identical values.

The router is CONSTRUCTED (construct_router_margins), not random: a random router gives
near-tied top-k logits that int4 quant flips 30% of the time, and no resident kernel can
match a fixture whose routing is itself a coin flip. Since hidden ≫ tokens, a least-squares
fit of router.proj to the fixture's own router inputs reproduces any target logit per token
exactly, giving every top-k decision a wide, quant-robust margin (the real resident-MoE gate).

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
# PIN_OUT redirects the checkpoint dir (default = the committed golden fixture). A scaled
# fixture for the A′ latency characterization sets PIN_OUT + PIN_HIDDEN/PIN_MOE_INTER/
# PIN_NUM_EXPERTS/PIN_TOPK to get realistic per-expert GEMV sizes; the golden JSON it writes
# alongside is scratch (uncommitted) and not consumed by the latency test.
CKPT = os.getenv("PIN_OUT", os.path.join(HERE, "..", "testdata", "gemma4-moe-tiny"))
OUT = os.path.join(os.path.dirname(CKPT), "gemma4_moe_forward_golden.json") if os.getenv("PIN_OUT") else \
    os.path.join(HERE, "..", "testdata", "gemma4_moe_forward_golden.json")

CFG = dict(
    # hidden/intermediate/moe_intermediate are multiples of 32 (int4 group size). hidden=256 (not 64)
    # keeps W4A8 int8-activation rounding out of the degenerate regime, but note the RESIDENT gate is
    # int4-vs-int4, NOT int4-vs-f32 — the int4-vs-f32 "floor" here is only ~0.79 yet the analogous
    # Split-A dense fixture gates resident parity at 0.979 with an f32-floor of 0.88. What actually
    # gates a resident MoE kernel is 100% routing agreement + a wide routing MARGIN (see
    # construct_router_margins + TestQuantNoiseFloor_gemma4MoE), which hidden does not set — the
    # least-squares router does. The env knobs (PIN_HIDDEN/PIN_MOE_INTER/PIN_HEAD_DIM/PIN_MOE) exist
    # only to reproduce the sizing sweep; the defaults ARE the committed golden's config.
    vocab_size=256, hidden_size=int(os.getenv("PIN_HIDDEN", "256")), num_hidden_layers=2,
    num_attention_heads=4, num_key_value_heads=2, head_dim=int(os.getenv("PIN_HEAD_DIM", "16")),
    intermediate_size=int(os.getenv("PIN_HIDDEN", "256")), rms_norm_eps=1e-6, tie_word_embeddings=True,
    max_position_embeddings=128, sliding_window=4,
    layer_types=["sliding_attention", "full_attention"],
    hidden_activation="gelu_pytorch_tanh", final_logit_softcapping=30.0,
    hidden_size_per_layer_input=0,  # PLE-free, like the 12B/26B
    rope_local_base_freq=10000.0, rope_theta=1000000.0,
    # uniform attention (no global-wide head / K=V) — that path is covered by the
    # dense gemma4 goldens; this golden isolates the NEW FFN sub-block.
    # the parallel dense + MoE FFN:
    # top-2-of-4, not top-2-of-8: the identical code path (router + indexed expert GEMV + weighted
    # combine) with HALF the near-tie boundaries. No kernel coverage is lost by having fewer experts.
    enable_moe_block=(os.getenv("PIN_MOE", "1") == "1"), num_experts=int(os.getenv("PIN_NUM_EXPERTS", "4")),
    top_k_experts=int(os.getenv("PIN_TOPK", "2")), moe_intermediate_size=int(os.getenv("PIN_MOE_INTER", "64")),
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


def construct_router_margins(model, prompt):
    """CONSTRUCT confident routing, don't amplify random ties. A random router gives near-tied
    logits whose top-k boundary is a coin flip quantization noise decides — and scaling can't fix it
    (gap and noise scale together). So fit router.proj to the fixture's OWN inputs (cheating in a
    model, correct in a kernel-gating fixture): make each token's intended top-2 experts win by a
    wide margin on the captured router input. Done SEQUENTIALLY per layer — each layer's router input
    depends on the previous layer's now-constructed router — so the margins hold at inference."""
    router_names = [n for n, m in model.named_modules() if n.endswith("router.proj")]
    g = torch.Generator().manual_seed(4321)
    for name in router_names:  # layer order: each capture sees the earlier layers already constructed
        mod = dict(model.named_modules())[name]
        cap = {}

        def hook(m, inp, cap=cap):
            x = inp[0].detach()
            cap["rn"] = x.reshape(-1, x.shape[-1])  # [tokens, hidden] (MoE flattens batch×seq)

        h = mod.register_forward_pre_hook(hook)
        with torch.no_grad():
            model(torch.tensor([prompt], dtype=torch.long), use_cache=False)
        h.remove()
        rn = cap["rn"]                                 # [tokens, hidden]
        seq, hidden = rn.shape
        nE = mod.weight.shape[0]
        # SOLVE for the router logits instead of accumulating directions. Accumulating (proj[e] += c·v_t)
        # cross-contaminates: an expert row shared by several tokens carries every one of their directions,
        # and a competing token's contribution eats into another token's 2nd-vs-3rd margin — cranking the
        # coefficient amplifies the pollution equally, so it never converges. Because hidden(256) ≫ tokens,
        # the rn_t are linearly independent, so the min-norm least-squares solution proj[e] = pinv(rn) @ tgt_e
        # reproduces ANY target logit per token EXACTLY (rn @ proj[e] == tgt_e), with zero cross-talk. Set a
        # clean, identical target per token — winner 8, second 5, losers 0 — so every one of the 16 decisions
        # has the SAME wide softmax margin, far past the 0.02 gate and independent of how the tokens overlap.
        pinv = torch.linalg.pinv(rn)                   # [hidden, tokens]
        tgt = torch.zeros(seq, nE)
        for t in range(seq):
            # The gate margin is prob(2nd_selected) − prob(1st_rejected). Both selected logits sit near 7
            # and both rejected sit at 0, so the boundary is huge (prob(2nd)≈0.27 vs prob(3rd)≈2e-4). A
            # dominating winner (e.g. 8 vs 5) would STEAL softmax mass from the 2nd and shrink that boundary
            # — what matters is the selected pair BOTH towering over the rejected pair, not the winner over
            # the 2nd. int4 quant erodes the f32 margin by ~0.03, so ~0.27 leaves wide headroom past 0.02.
            tgt[t, t % nE] = 7.0                        # winner logit
            tgt[t, (t + 1) % nE] = 6.0                  # 2nd logit — close to winner, far above the zeros
        proj = (pinv @ tgt).T                           # [nE, hidden]: rn @ proj.T == tgt
        with torch.no_grad():
            mod.weight.copy_(proj)


def main():
    torch.manual_seed(0)
    config = Gemma4TextConfig(**CFG)
    model = Gemma4ForCausalLM(config).eval().to(torch.float32)
    counts = strengthen(model)
    construct_router_margins(model, PROMPT)

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
