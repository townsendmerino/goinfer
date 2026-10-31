"""Write a generated Qwen1.5-MoE-A2.7B checkpoint at the real model's widths with fewer layers (OP4m-d3's D3-3 option (b)).

The config and tokenizer files are the real model's, with num_hidden_layers (and max_window_layers) set to LAYERS. Every
tensor has the real shape and is random f16 (norms near 1, weights normal times 0.02, biases small), written one tensor at a
time so memory stays small. A speed cell needs the real widths, not the real values.

usage: gen_qwen15moe.py SRC_DIR OUT_DIR LAYERS
"""
import json
import os
import shutil
import struct
import sys

import numpy as np

src, out, layers = sys.argv[1], sys.argv[2], int(sys.argv[3])
os.makedirs(out, exist_ok=True)
cfg = json.load(open(os.path.join(src, "config.json")))
cfg["num_hidden_layers"] = layers
cfg["max_window_layers"] = layers
json.dump(cfg, open(os.path.join(out, "config.json"), "w"), indent=2)
for f in ["tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "generation_config.json"]:
    if os.path.exists(os.path.join(src, f)):
        shutil.copy(os.path.join(src, f), out)

H, V = cfg["hidden_size"], cfg["vocab_size"]
E, MI, SI = cfg["num_experts"], cfg["moe_intermediate_size"], cfg["shared_expert_intermediate_size"]
NH, NKV = cfg["num_attention_heads"], cfg["num_key_value_heads"]
HD = H // NH
tensors = [("model.embed_tokens.weight", [V, H], "w"), ("model.norm.weight", [H], "n"), ("lm_head.weight", [V, H], "w")]
for l in range(layers):
    p = f"model.layers.{l}."
    tensors += [
        (p + "input_layernorm.weight", [H], "n"), (p + "post_attention_layernorm.weight", [H], "n"),
        (p + "self_attn.q_proj.weight", [NH * HD, H], "w"), (p + "self_attn.q_proj.bias", [NH * HD], "b"),
        (p + "self_attn.k_proj.weight", [NKV * HD, H], "w"), (p + "self_attn.k_proj.bias", [NKV * HD], "b"),
        (p + "self_attn.v_proj.weight", [NKV * HD, H], "w"), (p + "self_attn.v_proj.bias", [NKV * HD], "b"),
        (p + "self_attn.o_proj.weight", [H, NH * HD], "w"),
        (p + "mlp.gate.weight", [E, H], "w"),
        (p + "mlp.shared_expert.gate_proj.weight", [SI, H], "w"), (p + "mlp.shared_expert.up_proj.weight", [SI, H], "w"),
        (p + "mlp.shared_expert.down_proj.weight", [H, SI], "w"), (p + "mlp.shared_expert_gate.weight", [1, H], "w"),
    ]
    for e in range(E):
        q = p + f"mlp.experts.{e}."
        tensors += [(q + "gate_proj.weight", [MI, H], "w"), (q + "up_proj.weight", [MI, H], "w"), (q + "down_proj.weight", [H, MI], "w")]

header, off = {}, 0
for name, shape, _ in tensors:
    n = int(np.prod(shape)) * 2
    header[name] = {"dtype": "F16", "shape": shape, "data_offsets": [off, off + n]}
    off += n
hb = json.dumps(header, separators=(",", ":")).encode()
hb += b" " * ((8 - len(hb) % 8) % 8)
rng = np.random.default_rng(20261031)
path = os.path.join(out, "model.safetensors")
with open(path, "wb") as f:
    f.write(struct.pack("<Q", len(hb)))
    f.write(hb)
    for i, (name, shape, kind) in enumerate(tensors):
        n = int(np.prod(shape))
        if kind == "n":
            a = 1 + rng.standard_normal(n, dtype=np.float32) * 0.02
        elif kind == "b":
            a = rng.standard_normal(n, dtype=np.float32) * 0.01
        else:
            a = rng.standard_normal(n, dtype=np.float32) * 0.02
        a.astype(np.float16).tofile(f)
        if i % 500 == 0:
            print(f"{i}/{len(tensors)} tensors, {f.tell() >> 20} MiB", flush=True)
print(f"wrote {path}: {len(tensors)} tensors, {off >> 20} MiB of data, {layers} layers", flush=True)
