import json, os, sys, numpy as np
from safetensors.numpy import save_file
out, image, patch, hidden, heads, inter, layers = sys.argv[1], *map(int, sys.argv[2:8])
r = np.random.default_rng(1)
g = image // patch
def lin(o, i): return (r.standard_normal((o, i)) / np.sqrt(i)).astype(np.float32)
def vec(n, s=0.02): return (s * r.standard_normal(n)).astype(np.float32)
t = {"embeddings.patch_embedding.weight": (r.standard_normal((hidden, 3, patch, patch)) / np.sqrt(3*patch*patch)).astype(np.float32),
     "embeddings.patch_embedding.bias": vec(hidden), "embeddings.position_embedding.weight": vec(g*g*hidden, 0.5).reshape(g*g, hidden),
     "post_layernorm.weight": 1 + vec(hidden, 0.1), "post_layernorm.bias": vec(hidden)}
for l in range(layers):
    p = f"encoder.layers.{l}."
    for n in ["layer_norm1", "layer_norm2"]:
        t[p+n+".weight"], t[p+n+".bias"] = 1 + vec(hidden, 0.1), vec(hidden)
    for n in ["q_proj", "k_proj", "v_proj", "out_proj"]:
        t[p+"self_attn."+n+".weight"], t[p+"self_attn."+n+".bias"] = lin(hidden, hidden), vec(hidden)
    t[p+"mlp.fc1.weight"], t[p+"mlp.fc1.bias"] = lin(inter, hidden), vec(inter)
    t[p+"mlp.fc2.weight"], t[p+"mlp.fc2.bias"] = lin(hidden, inter), vec(hidden)
os.makedirs(out, exist_ok=True)
save_file(t, os.path.join(out, "model.safetensors"))
json.dump({"model_type": "siglip_vision_model", "hidden_size": hidden, "image_size": image, "intermediate_size": inter,
           "layer_norm_eps": 1e-6, "num_attention_heads": heads, "num_channels": 3, "num_hidden_layers": layers,
           "patch_size": patch}, open(os.path.join(out, "config.json"), "w"))
print(out, g*g, "patches")
