#!/usr/bin/env python
"""Convert the tiny-random GLM-4.5 (Glm4Moe) HF checkpoint to a tiny F32 GGUF, for
the goinfer GGUF-loader weightDiff gate. F32 (no quant) so the GGUF loader's output
is byte-comparable to the bit-exact safetensors loader — any divergence is the
LOADER (tensor naming, the dense/MoE split, the e_score_correction_bias, the
ungated shared expert), not quant noise.

Tensor names + metadata keys come from the official `gguf` library's GLM4_MOE
mapping (the same TensorNameMap convert_hf_to_gguf.py uses), so the fixture matches
the real llama.cpp container, not a hand-invented one. GLM uses GPT-NeoX RoPE, so —
like qwen2/qwen3/gemma — there is NO q/k permute to replicate (ggufQKPermuted false).

    ~/.venv-vl/bin/python scripts/pin_glm_tiny_gguf.py
    -> testdata/glm-tiny.gguf
"""
import json
import os

import gguf
import numpy as np
from safetensors import safe_open

HERE = os.path.dirname(__file__)
CKPT = os.path.join(HERE, "..", "testdata", "glm-tiny")
OUT = os.path.join(HERE, "..", "testdata", "glm-tiny.gguf")

ARCH = gguf.MODEL_ARCH.GLM4_MOE
ARCH_NAME = gguf.MODEL_ARCH_NAMES[ARCH]  # "glm4moe"


def main():
    with open(os.path.join(CKPT, "config.json")) as f:
        cfg = json.load(f)

    tensors = {}
    with safe_open(os.path.join(CKPT, "model.safetensors"), framework="np") as st:
        for k in st.keys():
            tensors[k] = st.get_tensor(k)

    n_layer = cfg["num_hidden_layers"]
    n_expert = cfg["n_routed_experts"]
    head_dim = cfg["head_dim"]
    rot = int(cfg["partial_rotary_factor"] * head_dim)

    w = gguf.GGUFWriter(OUT, ARCH_NAME)
    w.add_block_count(n_layer)
    w.add_context_length(cfg["max_position_embeddings"])
    w.add_embedding_length(cfg["hidden_size"])
    w.add_feed_forward_length(cfg["intermediate_size"])
    w.add_head_count(cfg["num_attention_heads"])
    w.add_head_count_kv(cfg["num_key_value_heads"])
    w.add_key_length(head_dim)
    w.add_value_length(head_dim)
    w.add_layer_norm_rms_eps(cfg["rms_norm_eps"])
    w.add_rope_dimension_count(rot)
    w.add_rope_freq_base(cfg["rope_parameters"]["rope_theta"])
    # DeepSeek-style MoE descriptors.
    w.add_expert_count(n_expert)
    w.add_expert_used_count(cfg["num_experts_per_tok"])
    w.add_expert_shared_count(cfg["n_shared_experts"])
    w.add_expert_feed_forward_length(cfg["moe_intermediate_size"])
    w.add_leading_dense_block_count(cfg["first_k_dense_replace"])
    w.add_expert_weights_scale(cfg["routed_scaling_factor"])
    w.add_expert_weights_norm(cfg["norm_topk_prob"])
    w.add_vocab_size(cfg["vocab_size"])

    tmap = gguf.get_tensor_name_map(ARCH, n_layer)

    # HF-suffix -> GGUF-suffix translator using the official map. For the unmapped
    # stacked-expert tensors (HF stores per-expert; GGUF stacks) we stack by hand.
    def put(hf_name, arr, suffix=".weight"):
        base = hf_name[: -len(suffix)] if hf_name.endswith(suffix) else hf_name
        gg = tmap.get_name(base, try_suffixes=(".weight", ".bias"))
        if gg is None:
            raise KeyError(f"no GGUF mapping for {hf_name!r} (base {base!r})")
        w.add_tensor(gg + suffix, np.ascontiguousarray(arr.astype(np.float32)))

    put("model.embed_tokens.weight", tensors["model.embed_tokens.weight"])
    put("model.norm.weight", tensors["model.norm.weight"])
    put("lm_head.weight", tensors["lm_head.weight"])

    for i in range(n_layer):
        pre = f"model.layers.{i}."
        put(pre + "input_layernorm.weight", tensors[pre + "input_layernorm.weight"])
        put(pre + "post_attention_layernorm.weight", tensors[pre + "post_attention_layernorm.weight"])
        for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
            put(pre + f"self_attn.{proj}.weight", tensors[pre + f"self_attn.{proj}.weight"])
        put(pre + "self_attn.q_norm.weight", tensors[pre + "self_attn.q_norm.weight"])
        put(pre + "self_attn.k_norm.weight", tensors[pre + "self_attn.k_norm.weight"])

        if i < cfg["first_k_dense_replace"]:
            for proj in ("gate_proj", "up_proj", "down_proj"):
                put(pre + f"mlp.{proj}.weight", tensors[pre + f"mlp.{proj}.weight"])
            continue

        # MoE layer: router + e_score_correction_bias + stacked experts + shared.
        put(pre + "mlp.gate.weight", tensors[pre + "mlp.gate.weight"])
        # exp_probs_b: HF source base is "...mlp.gate.e_score_correction"; it is a
        # bias term, so llama.cpp stores it with a .bias suffix.
        bias = tensors[pre + "mlp.gate.e_score_correction_bias"]
        gg_bias = tmap.get_name(pre + "mlp.gate.e_score_correction")
        if gg_bias is None:
            raise KeyError("no GGUF mapping for e_score_correction bias")
        w.add_tensor(gg_bias + ".bias", np.ascontiguousarray(bias.astype(np.float32)))

        for kind in ("gate_proj", "up_proj", "down_proj"):
            stack = np.stack([tensors[pre + f"mlp.experts.{e}.{kind}.weight"] for e in range(n_expert)], axis=0)
            gg = tmap.get_name(pre + f"mlp.experts.{kind}")  # stacked-expert source base
            if gg is None:
                raise KeyError(f"no GGUF mapping for stacked experts {kind}")
            w.add_tensor(gg + ".weight", np.ascontiguousarray(stack.astype(np.float32)))

        for proj in ("gate_proj", "up_proj", "down_proj"):
            put(pre + f"mlp.shared_experts.{proj}.weight", tensors[pre + f"mlp.shared_experts.{proj}.weight"])

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {OUT}  (arch={ARCH_NAME}, {n_layer} layers, {n_expert} experts)")


if __name__ == "__main__":
    main()
