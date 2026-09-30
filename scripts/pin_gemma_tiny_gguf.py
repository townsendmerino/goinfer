#!/usr/bin/env python3
"""Convert the tiny Gemma 1 / Gemma 2 checkpoints (scripts/pin_gemma_tiny.py) to F32 GGUFs, for the GGUF loader gate.

    ~/.venv-peft/bin/python scripts/pin_gemma_tiny_gguf.py   ->  testdata/gemma1-tiny.gguf, testdata/gemma2-hd-tiny.gguf

F32, so the GGUF loader's output can be compared with the safetensors loader's on the same weights: any divergence is
the loader (names, the norm offset, the metadata it reads or derives), not quantization. Mirrors llama.cpp's
convert_hf_to_gguf.py for these two families, with tensor names from the gguf library's own TensorNameMap:
  - arch "gemma" / "gemma2"; the head is tied, so no output.weight;
  - every *norm.weight is stored + 1 (GemmaModel.modify_tensors: llama.cpp's RMSNorm has no (1+w)); the loader
    subtracts it back;
  - Gemma 2 writes sliding_window and both softcaps; neither writes query_pre_attn_scalar or layer types, which the
    loader derives as llama.cpp's own does.
No tokenizer metadata: the loader does not need it, as for testdata/glm-tiny.gguf. The real-checkpoint check is the
independent one: a llama.cpp-converted GGUF against the HF weights.
"""
import json, os
import gguf
import numpy as np
from safetensors import safe_open

TD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "testdata")
ARCHS = {"gemma1": gguf.MODEL_ARCH.GEMMA, "gemma2-hd": gguf.MODEL_ARCH.GEMMA2}  # gemma2-hd: query_pre_attn_scalar == head_dim

for name, arch in ARCHS.items():
    ckpt = os.path.join(TD, f"{name}-tiny")
    cfg = json.load(open(os.path.join(ckpt, "config.json")))
    with safe_open(os.path.join(ckpt, "model.safetensors"), framework="np") as st:
        tensors = {k: st.get_tensor(k) for k in st.keys()}
    n = cfg["num_hidden_layers"]
    out = os.path.join(TD, f"{name}-tiny.gguf")
    w = gguf.GGUFWriter(out, gguf.MODEL_ARCH_NAMES[arch])
    w.add_context_length(cfg["max_position_embeddings"])
    w.add_embedding_length(cfg["hidden_size"])
    w.add_block_count(n)
    w.add_feed_forward_length(cfg["intermediate_size"])
    w.add_head_count(cfg["num_attention_heads"])
    w.add_head_count_kv(cfg["num_key_value_heads"])
    w.add_key_length(cfg["head_dim"])
    w.add_value_length(cfg["head_dim"])
    w.add_layer_norm_rms_eps(cfg["rms_norm_eps"])
    w.add_vocab_size(cfg["vocab_size"])
    if name.startswith("gemma2"):
        w.add_sliding_window(cfg["sliding_window"])
        w.add_attn_logit_softcapping(cfg["attn_logit_softcapping"])
        w.add_final_logit_softcapping(cfg["final_logit_softcapping"])
    tmap = gguf.get_tensor_name_map(arch, n)
    for hf, arr in tensors.items():
        if hf == "lm_head.weight":
            continue  # tied
        base = hf[: -len(".weight")]
        gg = tmap.get_name(base, try_suffixes=(".weight",))
        if gg is None:
            raise KeyError(f"no GGUF mapping for {hf!r}")
        a = arr.astype(np.float32)
        if hf.endswith("norm.weight"):
            a = a + 1  # llama.cpp's Gemma convention
        w.add_tensor(gg + ".weight", np.ascontiguousarray(a))
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
    print(f"wrote {out} ({len(tensors)} tensors)")
