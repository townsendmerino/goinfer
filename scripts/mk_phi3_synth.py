#!/usr/bin/env python3
"""Write a random Phi3ForCausalLM checkpoint with phi3-mini-4k's exact per-layer dims
(hidden 3072, 32 heads == 32 kv heads, head_dim 96, inter 8192, vocab 32064) but few layers,
as safetensors + config.json, using only numpy. Optional --outlier plants a massive activation
channel (an input_layernorm weight spike on channel 7 in layer 0 plus a large embedding value
for token 1) to emulate the BOS massive-activation regime of real LLMs."""
import json, os, struct, sys
import numpy as np

out = sys.argv[1]
L = int(sys.argv[2]) if len(sys.argv) > 2 else 4
outlier = "--outlier" in sys.argv
H, NH, NKV, HD, I, V = 3072, 32, 32, 96, 8192, 32064
rng = np.random.default_rng(1234)
os.makedirs(out, exist_ok=True)

def nrm(shape, std=0.02):
    return (rng.standard_normal(shape, dtype=np.float32) * std).astype(np.float32)

tensors = {}
tensors["model.embed_tokens.weight"] = nrm((V, H))
tensors["lm_head.weight"] = nrm((V, H))
tensors["model.norm.weight"] = np.ones(H, np.float32)
for l in range(L):
    p = f"model.layers.{l}."
    tensors[p + "input_layernorm.weight"] = np.ones(H, np.float32)
    tensors[p + "post_attention_layernorm.weight"] = np.ones(H, np.float32)
    tensors[p + "self_attn.qkv_proj.weight"] = nrm((NH * HD + 2 * NKV * HD, H))
    tensors[p + "self_attn.o_proj.weight"] = nrm((H, NH * HD))
    tensors[p + "mlp.gate_up_proj.weight"] = nrm((2 * I, H))
    tensors[p + "mlp.down_proj.weight"] = nrm((H, I))
if outlier:
    # A handful of channels carry a "massive activation": the embedding of token 1 (<s>) has a
    # few huge entries, and layer 0's norm weight amplifies channel 7 for EVERY token, so the
    # int8 activation scale is dominated by one channel at every position, like a real LLM.
    e = tensors["model.embed_tokens.weight"]
    e[1, 7] = 400.0
    e[1, 1500] = -250.0
    for l in range(L):
        tensors[f"model.layers.{l}.input_layernorm.weight"][7] = 40.0
        tensors[f"model.layers.{l}.post_attention_layernorm.weight"][7] = 40.0

# safetensors: u64 header length, JSON header, raw little-endian data.
header = {}
off = 0
order = list(tensors.keys())
for k in order:
    n = tensors[k].nbytes
    header[k] = {"dtype": "F32", "shape": list(tensors[k].shape), "data_offsets": [off, off + n]}
    off += n
hj = json.dumps(header, separators=(",", ":")).encode()
pad = (8 - len(hj) % 8) % 8
hj += b" " * pad
with open(os.path.join(out, "model.safetensors"), "wb") as f:
    f.write(struct.pack("<Q", len(hj)))
    f.write(hj)
    for k in order:
        f.write(np.ascontiguousarray(tensors[k]).tobytes())

cfg = {
    "architectures": ["Phi3ForCausalLM"], "model_type": "phi3", "dtype": "float32",
    "hidden_size": H, "intermediate_size": I, "num_hidden_layers": L,
    "num_attention_heads": NH, "num_key_value_heads": NKV,
    "max_position_embeddings": 4096, "rms_norm_eps": 1e-05, "rope_theta": 10000.0,
    "hidden_act": "silu", "tie_word_embeddings": False, "vocab_size": V,
    "bos_token_id": 1, "eos_token_id": 32000, "pad_token_id": 32000,
    "attention_dropout": 0.0, "initializer_range": 0.02, "original_max_position_embeddings": 4096,
    "resid_pdrop": 0.0, "embd_pdrop": 0.0, "use_cache": True,
}
json.dump(cfg, open(os.path.join(out, "config.json"), "w"), indent=1)
print("wrote", out, "layers", L, "bytes", off + 8 + len(hj), "outlier", outlier)
