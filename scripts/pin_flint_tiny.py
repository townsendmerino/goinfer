#!/usr/bin/env python3
"""Build a tiny-random Flint (FlintForCausalLM) checkpoint + forward and window-boundary
goldens (F2 of docs/completed/task-flint-2026-10.md).

Reference module is the upstream transformers PR #49824 implementation
(src/transformers/models/flint/{configuration_flint.py,modeling_flint.py},
merged in commit 0ea108daf169dd66a1800e48a56d2251d4ff76bd), fetched at that exact
commit and verified by SHA-256 before execution.

Exercises every architectural property of RecursiveSI/flint at a fast, deterministic
CI size (~1.5 MB in float32):
  - 6 layers: 2 dense lead layers (mlp_only_layers=[0, 1], SwiGLU) + 4 sparse MoE
    layers (layers 2..5: 8 routed experts, top-2 sigmoid router with norm_topk_prob=True
    and no router bias + 1 ungated shared expert).
  - Hybrid full/sliding attention: layers [0, 1, 5] are full_attention with NoPE
    (no_rope_layer_indices=[0, 1, 5]); layers [2, 3, 4] are sliding_attention with
    split-half RoPE (rope_theta=10000.0).
  - Per-head QK RMSNorm (q_norm, k_norm over head_dim=16) with non-unit weights.
  - Per-head learned attention sinks (self_attn.softmax_offset in the released
    safetensors layout; self_attn.sinks in transformers).
  - Untied embeddings (tie_word_embeddings=False).

Also pins the F2 window-boundary golden on the same weights with sliding_window=513
(the real Flint window size: current token + 512 preceding tokens) across a 1,100-token
prompt at lengths [16, 256, 511, 512, 513, 514, 515, 800, 1025, 1026, 1027, 1100].

    python3 scripts/pin_flint_tiny.py
    -> testdata/flint-tiny/{config.json,model.safetensors}
    -> testdata/flint_tiny_golden.json
"""
import hashlib
import json
import os
import sys
import types
import urllib.request

import torch
from safetensors.torch import save_file

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
TD = os.path.join(ROOT, "testdata")
CKPT = os.path.join(TD, "flint-tiny")
GOLDEN = os.path.join(TD, "flint_tiny_golden.json")
CODE_CACHE = os.path.join(ROOT, "decoder", "testdata", ".flint-code")

CODE_REVISION = "0ea108daf169dd66a1800e48a56d2251d4ff76bd"
CODE_SHA256 = {
    "configuration_flint.py": "849eee3068846081077a7568663355802c8c4eec3741d92cd578d222bd144d00",
    "modeling_flint.py": "869d318cec68654ae4f6132a6d1462cff4d6f99ae811d75e6c9e7247dd99d772",
}

PROMPT = [1, 7, 42, 100, 5, 200, 13, 88, 3, 71, 9, 17, 60, 33, 2, 88]  # len 16 > sliding_window 8
N_NEW = 6

WINDOW_513 = 513
WINDOW_513_LEN = 1100
WINDOW_513_LENGTHS = [16, 256, 511, 512, 513, 514, 515, 800, 1025, 1026, 1027, 1100]

CFG = dict(
    architectures=["FlintForCausalLM"],
    model_type="flint",
    vocab_size=256,
    hidden_size=64,
    intermediate_size=128,
    num_hidden_layers=6,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=16,
    hidden_act="silu",
    max_position_embeddings=2048,
    initializer_range=0.02,
    rms_norm_eps=1e-6,
    use_cache=True,
    tie_word_embeddings=False,
    rope_theta=10000.0,
    use_sliding_window=True,
    sliding_window=8,
    attention_bias=False,
    attention_dropout=0.0,
    attention_sink=True,
    softmax_type="learnable",
    decoder_sparse_step=1,
    moe_intermediate_size=32,
    shared_expert_intermediate_size=32,
    num_experts=8,
    num_local_experts=8,
    num_experts_per_tok=2,
    norm_topk_prob=True,
    router_score_function="sigmoid",
    router_expert_bias=False,
    mlp_only_layers=[0, 1],
    layer_types=[
        "full_attention",
        "full_attention",
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "full_attention",
    ],
    no_rope_layer_indices=[0, 1, 5],
    bos_token_id=1,
    eos_token_id=[2, 3],
    dtype="float32",
)


def load_upstream_flint():
    os.makedirs(CODE_CACHE, exist_ok=True)
    for fn, want_sha in CODE_SHA256.items():
        path = os.path.join(CODE_CACHE, fn)
        if os.path.exists(path):
            data = open(path, "rb").read()
            if hashlib.sha256(data).hexdigest() == want_sha:
                continue
        url = (
            f"https://raw.githubusercontent.com/huggingface/transformers/"
            f"{CODE_REVISION}/src/transformers/models/flint/{fn}"
        )
        with urllib.request.urlopen(url, timeout=60) as r:
            data = r.read()
        got_sha = hashlib.sha256(data).hexdigest()
        if got_sha != want_sha:
            raise SystemExit(f"{fn} sha256 mismatch: got {got_sha}, want {want_sha}")
        with open(path, "wb") as f:
            f.write(data)

    pkg = types.ModuleType("transformers.models.flint")
    pkg.__path__ = [CODE_CACHE]
    pkg.__package__ = "transformers.models.flint"
    sys.modules["transformers.models.flint"] = pkg

    cfg_path = os.path.join(CODE_CACHE, "configuration_flint.py")
    cfg_mod = types.ModuleType("transformers.models.flint.configuration_flint")
    cfg_mod.__file__ = cfg_path
    cfg_mod.__package__ = "transformers.models.flint"
    sys.modules["transformers.models.flint.configuration_flint"] = cfg_mod
    exec(compile(open(cfg_path, "rb").read(), cfg_path, "exec"), cfg_mod.__dict__)

    mod_path = os.path.join(CODE_CACHE, "modeling_flint.py")
    mod_mod = types.ModuleType("transformers.models.flint.modeling_flint")
    mod_mod.__file__ = mod_path
    mod_mod.__package__ = "transformers.models.flint"
    sys.modules["transformers.models.flint.modeling_flint"] = mod_mod
    exec(compile(open(mod_path, "rb").read(), mod_path, "exec"), mod_mod.__dict__)

    return cfg_mod.FlintConfig, mod_mod.FlintForCausalLM, mod_mod.FlintRMSNorm, mod_mod.FlintAttention


def to_real_checkpoint_state_dict(sd: dict[str, torch.Tensor], moe_inter: int) -> dict[str, torch.Tensor]:
    """Convert transformers 5.x in-memory state_dict keys (self_attn.sinks and fused 3D
    mlp.experts.gate_up_proj/down_proj) to RecursiveSI/flint's released safetensors
    layout (self_attn.softmax_offset and per-expert mlp.experts.{e}.{gate,up,down}_proj.weight).
    """
    out = {}
    for k, v in sd.items():
        if k.endswith(".self_attn.sinks"):
            out[k[: -len(".sinks")] + ".softmax_offset"] = v.contiguous()
        elif k.endswith(".mlp.experts.gate_up_proj"):
            prefix = k[: -len(".gate_up_proj")]
            for e in range(v.shape[0]):
                out[f"{prefix}.{e}.gate_proj.weight"] = v[e, :moe_inter, :].contiguous()
                out[f"{prefix}.{e}.up_proj.weight"] = v[e, moe_inter:, :].contiguous()
        elif k.endswith(".mlp.experts.down_proj"):
            prefix = k[: -len(".down_proj")]
            for e in range(v.shape[0]):
                out[f"{prefix}.{e}.down_proj.weight"] = v[e].contiguous()
        else:
            out[k] = v.contiguous()
    return out


def main():
    FlintConfig, FlintForCausalLM, FlintRMSNorm, FlintAttention = load_upstream_flint()

    cfg = FlintConfig(
        vocab_size=CFG["vocab_size"],
        hidden_size=CFG["hidden_size"],
        intermediate_size=CFG["intermediate_size"],
        num_hidden_layers=CFG["num_hidden_layers"],
        num_attention_heads=CFG["num_attention_heads"],
        num_key_value_heads=CFG["num_key_value_heads"],
        head_dim=CFG["head_dim"],
        hidden_act=CFG["hidden_act"],
        max_position_embeddings=CFG["max_position_embeddings"],
        initializer_range=CFG["initializer_range"],
        rms_norm_eps=CFG["rms_norm_eps"],
        use_cache=CFG["use_cache"],
        tie_word_embeddings=CFG["tie_word_embeddings"],
        rope_parameters={"rope_type": "default", "rope_theta": CFG["rope_theta"]},
        sliding_window=CFG["sliding_window"],
        attention_dropout=CFG["attention_dropout"],
        decoder_sparse_step=CFG["decoder_sparse_step"],
        moe_intermediate_size=CFG["moe_intermediate_size"],
        shared_expert_intermediate_size=CFG["shared_expert_intermediate_size"],
        num_experts_per_tok=CFG["num_experts_per_tok"],
        num_experts=CFG["num_experts"],
        norm_topk_prob=CFG["norm_topk_prob"],
        mlp_only_layers=CFG["mlp_only_layers"],
        layer_types=CFG["layer_types"],
        attention_bias=CFG["attention_bias"],
    )

    torch.manual_seed(0)
    model = FlintForCausalLM(cfg).eval().to(torch.float32)

    # Give RMSNorms non-unit weights and attention sinks non-trivial spread so any
    # skipped norm, swapped Q/K norm, or dropped sink shifts logits by > 1e-1.
    with torch.no_grad():
        for mod in model.modules():
            if isinstance(mod, FlintRMSNorm):
                mod.weight.copy_(1.0 + 0.25 * torch.randn_like(mod.weight))
            elif isinstance(mod, FlintAttention):
                mod.sinks.copy_(0.5 * torch.randn_like(mod.sinks))

    with torch.no_grad():
        ids = torch.tensor([PROMPT], dtype=torch.long)
        last_logits = model(input_ids=ids, use_cache=False).logits[0, -1].float().tolist()
        cur, cont = list(PROMPT), []
        for _ in range(N_NEW):
            o = model(input_ids=torch.tensor([cur], dtype=torch.long), use_cache=False)
            cont.append(int(o.logits[0, -1].argmax()))
            cur.append(cont[-1])

    # F2.2 window-boundary golden: run the exact same weights with sliding_window=513
    # over 1,100 tokens, recording full logits at 512, 513, 514, 1100 and checkpoints.
    model.config.sliding_window = WINDOW_513
    for layer in model.model.layers:
        if layer.self_attn.sliding_window is not None:
            layer.self_attn.sliding_window = WINDOW_513

    win_ids = [(i * 31 + 7) % CFG["vocab_size"] for i in range(WINDOW_513_LEN)]
    with torch.no_grad():
        win_out = model(input_ids=torch.tensor([win_ids], dtype=torch.long), use_cache=False).logits[0].float()

    window_checkpoints = []
    for length in WINDOW_513_LENGTHS:
        row = win_out[length - 1].tolist()
        window_checkpoints.append({
            "len": length,
            "pos": length - 1,
            "argmax": int(win_out[length - 1].argmax()),
            "logits": row,
        })

    # Restore sliding_window=8 on the saved config.
    model.config.sliding_window = CFG["sliding_window"]
    for layer in model.model.layers:
        if layer.self_attn.sliding_window is not None:
            layer.self_attn.sliding_window = CFG["sliding_window"]

    os.makedirs(CKPT, exist_ok=True)
    real_sd = to_real_checkpoint_state_dict(model.state_dict(), CFG["moe_intermediate_size"])
    save_file(real_sd, os.path.join(CKPT, "model.safetensors"))
    with open(os.path.join(CKPT, "config.json"), "w") as f:
        json.dump(CFG, f, indent=2)
        f.write("\n")

    golden = {
        "note": (
            "tiny-random FlintForCausalLM (transformers PR #49824 @ "
            + CODE_REVISION[:10]
            + "), CPU fp32; includes 16-token prompt + 6-token greedy continuation at "
            "sliding_window=8, and 1,100-token window-boundary golden at sliding_window=513 "
            "(lengths 512, 513, 514, 1100 and intermediate positions)."
        ),
        "code_revision": CODE_REVISION,
        "code_sha256": CODE_SHA256,
        "config": CFG,
        "prompt_ids": PROMPT,
        "argmax": int(torch.tensor(last_logits).argmax()),
        "last_logits": last_logits,
        "n_new": N_NEW,
        "continuation_ids": cont,
        "window_513": {
            "sliding_window": WINDOW_513,
            "prompt_len": WINDOW_513_LEN,
            "checkpoints": window_checkpoints,
        },
    }
    with open(GOLDEN, "w") as f:
        json.dump(golden, f)

    st_size = os.path.getsize(os.path.join(CKPT, "model.safetensors"))
    print(f"flint-tiny: argmax={golden['argmax']} cont={cont} safetensors={st_size} B")
    for cp in window_checkpoints:
        if cp["len"] in (512, 513, 514, 1100):
            print(f"  window_513 len={cp['len']} (pos={cp['pos']}): argmax={cp['argmax']}")
    print(f"-> {CKPT}\n-> {GOLDEN}")


if __name__ == "__main__":
    main()
