#!/usr/bin/env python
"""Build three tiny-random Laguna checkpoints + pin their text goldens — the G6
parity fixtures (mirrors pin_glm_tiny / pin_mellum2).

THREE, not one, because Laguna's three released generations differ in ways that a
single fixture cannot exercise:

  * xs21 — `gating: "per-head"`, sliding/full 3:1 interleave, PER-LAYER query
           heads, YaRN(factor 32) on full layers + plain RoPE on sliding, one
           dense prefix layer.
  * xs2  — `gating: true`. Its OWN module hardcodes per-head gating and ignores
           the field, so this pins that the granularity comes from the tensor
           shape rather than the config spelling.
  * m1   — `gating: "per-element"`, NO sliding window, uniform heads, routed
           scaling 1.0, three dense prefix layers. The per-element gate path.

Each variant is generated against ITS OWN generation's modeling_laguna.py — the
XS.2 module is a different file from the XS-2.1/M.1 one (34KB vs 41KB), so using
one module for all three would silently validate goinfer against the wrong
reference. The vendored .py files are copied into each checkpoint dir so the
fixtures reproduce offline.

    ~/.venv-vl/bin/python scripts/pin_laguna_tiny.py
    -> testdata/laguna_{xs21,xs2,m1}_tiny_text_golden.json   (tracked)
    -> testdata/laguna-{xs21,xs2,m1}-tiny/                   (gitignored)
"""
import json
import os
import shutil
import urllib.request

import torch
from transformers import AutoConfig, AutoModelForCausalLM

HERE = os.path.dirname(os.path.abspath(__file__))
TESTDATA = os.path.join(HERE, "..", "decoder", "testdata")
# XS.2 is already on disk (the T3 target); the other two generations' code is fetched.
LOCAL_XS2 = "/home/francis/models/laguna-xs2"

PROMPT = [2, 7, 42, 100, 5, 200, 13, 88]
N_NEW = 6

COMMON = dict(
    model_type="laguna",
    vocab_size=256, hidden_size=64, intermediate_size=128, head_dim=16,
    num_key_value_heads=2, rms_norm_eps=1e-6, attention_bias=False,
    tie_word_embeddings=False, max_position_embeddings=512,
    num_experts=8, num_experts_per_tok=2, moe_intermediate_size=32,
    shared_expert_intermediate_size=32, hidden_act="silu",
    moe_apply_router_weight_on_input=False, router_aux_loss_coef=0.0,
)

YARN_FULL = lambda factor, omax: dict(
    rope_type="yarn", rope_theta=500000.0, factor=factor,
    original_max_position_embeddings=omax, beta_fast=64.0, beta_slow=1.0,
    partial_rotary_factor=0.5,
)
PLAIN_SLIDING = dict(rope_type="default", rope_theta=10000.0, partial_rotary_factor=1.0)

VARIANTS = {
    # (repo whose modeling code defines this generation, config overrides)
    "xs21": ("poolside/Laguna-XS-2.1", dict(
        num_hidden_layers=4, num_attention_heads=4,
        num_attention_heads_per_layer=[4, 8, 8, 8],
        layer_types=["full_attention", "sliding_attention", "sliding_attention", "sliding_attention"],
        mlp_layer_types=["dense", "sparse", "sparse", "sparse"],
        mlp_only_layers=[0], decoder_sparse_step=1,
        sliding_window=16, gating="per-head", gating_types=["per_head"] * 4,
        norm_topk_prob=True, moe_routed_scaling_factor=2.5,
        rope_parameters={"full_attention": YARN_FULL(32.0, 256), "sliding_attention": PLAIN_SLIDING},
    )),
    "xs2": (LOCAL_XS2, dict(
        num_hidden_layers=4, num_attention_heads=4,
        num_attention_heads_per_layer=[4, 8, 8, 8],
        layer_types=["full_attention", "sliding_attention", "sliding_attention", "sliding_attention"],
        mlp_layer_types=["dense", "sparse", "sparse", "sparse"],
        sliding_window=16, gating=True, partial_rotary_factor=0.5,
        moe_routed_scaling_factor=2.5,
        rope_parameters={"full_attention": YARN_FULL(64.0, 128), "sliding_attention": PLAIN_SLIDING},
    )),
    # M.1's OWN modeling_laguna.py cannot be loaded standalone: it ships with
    # package-relative imports (`from ...cache_utils import DynamicCache`), i.e. it
    # was exported as an in-library module rather than remote code. Diffing it
    # against XS-2.1's shows the ONLY differences are those import lines plus a
    # transformers>=5.12 conversion-mapping shim — the model code is identical — so
    # XS-2.1's module is the faithful reference for M.1's config.
    "m1": ("poolside/Laguna-XS-2.1", dict(
        num_hidden_layers=4, num_attention_heads=4,
        mlp_layer_types=["dense", "dense", "sparse", "sparse"],
        mlp_only_layers=[0, 1], decoder_sparse_step=1,
        sliding_window=0, gating="per-element", norm_topk_prob=True,
        moe_routed_scaling_factor=1.0,
        rope_parameters={"full_attention": dict(
            rope_type="yarn", rope_theta=500000.0, factor=64.0,
            original_max_position_embeddings=128, beta_fast=64.0, beta_slow=1.0,
            partial_rotary_factor=1.0)},
    )),
}

CODE_FILES = ("configuration_laguna.py", "modeling_laguna.py")


def fetch_code(src, dst):
    """Put this generation's own configuration/modeling .py into dst."""
    os.makedirs(dst, exist_ok=True)
    for fn in CODE_FILES:
        out = os.path.join(dst, fn)
        if os.path.isdir(src):
            shutil.copyfile(os.path.join(src, fn), out)
        else:
            url = f"https://huggingface.co/{src}/resolve/main/{fn}"
            with urllib.request.urlopen(url, timeout=60) as r, open(out, "wb") as f:
                f.write(r.read())


def build(tag, src, overrides):
    ckpt = os.path.join(TESTDATA, f"laguna-{tag}-tiny")
    code = os.path.join(TESTDATA, f".laguna-code-{tag}")
    fetch_code(src, code)

    cfg_dict = dict(COMMON, **overrides)
    # Write a config.json next to that generation's code so AutoConfig resolves the
    # LagunaConfig defined by THAT module, not a sibling generation's.
    cfg_dict["auto_map"] = {
        "AutoConfig": "configuration_laguna.LagunaConfig",
        "AutoModelForCausalLM": "modeling_laguna.LagunaForCausalLM",
    }
    cfg_dict["architectures"] = ["LagunaForCausalLM"]
    with open(os.path.join(code, "config.json"), "w") as f:
        json.dump(cfg_dict, f, indent=1)

    cfg = AutoConfig.from_pretrained(code, trust_remote_code=True)
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True).eval().to(torch.float32)

    with torch.no_grad():
        ids = torch.tensor([PROMPT], dtype=torch.long)
        last_logits = model(input_ids=ids, use_cache=False).logits[0, -1].float().tolist()
        cur, cont = list(PROMPT), []
        for _ in range(N_NEW):
            o = model(input_ids=torch.tensor([cur], dtype=torch.long), use_cache=False)
            cont.append(int(o.logits[0, -1].argmax()))
            cur.append(cont[-1])

    # Record the gate's actual granularity from the SHAPE, which is the thing
    # goinfer keys on — for xs2 this is expected to disagree with `gating: true`.
    g = model.model.layers[0].self_attn.g_proj.weight.shape[0]
    heads0 = cfg_dict.get("num_attention_heads_per_layer", [cfg_dict["num_attention_heads"]])[0]
    granularity = "per-head" if g == heads0 else "per-element"

    golden = {
        "note": f"tiny-random LagunaForCausalLM ({tag}), text forward; CPU fp32; "
                f"reference module from {src}",
        "config": cfg_dict, "prompt_ids": PROMPT,
        "argmax": int(torch.tensor(last_logits).argmax()),
        "last_logits": last_logits, "n_new": N_NEW, "continuation_ids": cont,
        "g_proj_rows_layer0": g, "gate_granularity": granularity,
        "declared_gating": overrides.get("gating"),
    }
    out = os.path.join(TESTDATA, f"laguna_{tag}_tiny_text_golden.json")
    with open(out, "w") as f:
        json.dump(golden, f)

    model.save_pretrained(ckpt, safe_serialization=True)
    for fn in CODE_FILES:
        shutil.copyfile(os.path.join(code, fn), os.path.join(ckpt, fn))
    print(f"[{tag}] argmax={golden['argmax']} cont={cont}")
    print(f"       g_proj rows={g} -> {granularity} (config said {overrides.get('gating')!r})")
    print(f"       -> {out}\n       -> {ckpt}")


def main():
    os.makedirs(TESTDATA, exist_ok=True)
    for tag, (src, ov) in VARIANTS.items():
        build(tag, src, ov)


if __name__ == "__main__":
    main()
