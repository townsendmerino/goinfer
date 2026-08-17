#!/usr/bin/env python
"""Build a REAL-WEIGHT layer-slice oracle for Laguna-XS.2 — the strongest numeric
gate this box can hold for a 33B model.

WHY A SLICE. T3 means cosine/argmax against a full reference forward of a released
checkpoint. Laguna-XS.2 is ~63GB in bf16; a reference forward alongside goinfer's
own copy does not fit in 62GB of RAM. A slice keeps the thing that matters — REAL
trained weights, with their real routing distributions, real gate magnitudes and
real QK-norm scales — and drops only depth. Tiny goldens use random weights, where
the router is near-uniform and the softplus gate sits in a narrow band; a real
slice exercises the same code on the distributions the model actually produces.

The slice is layers [0, N): the dense prefix layer PLUS enough MoE layers to cover
both attention types (layer 0 full_attention, layers 1..3 sliding_attention), which
is exactly the geometry that made per-layer query heads necessary.

    ~/.venv-vl/bin/python scripts/pin_laguna_slice.py
    -> decoder/testdata/laguna_xs2_slice_golden.json   (tracked)
    -> decoder/testdata/laguna-xs2-slice/              (gitignored, ~7GB)
"""
import json
import os
import shutil

import torch
from safetensors import safe_open
from safetensors.torch import save_file

SRC = "/home/francis/models/laguna-xs2"
HERE = os.path.dirname(os.path.abspath(__file__))
TESTDATA = os.path.join(HERE, "..", "decoder", "testdata")
OUT_CKPT = os.path.join(TESTDATA, "laguna-xs2-slice")
OUT_GOLDEN = os.path.join(TESTDATA, "laguna_xs2_slice_golden.json")

N_LAYERS = 4          # layer 0 = dense + full_attention; 1..3 = MoE + sliding
PROMPT = [2, 1547, 913, 24, 88, 7, 100, 2001]
N_NEW = 4


def main():
    idx = json.load(open(os.path.join(SRC, "model.safetensors.index.json")))["weight_map"]
    keep = {}
    for name, shard in idx.items():
        if name.startswith("model.layers."):
            li = int(name.split(".")[2])
            if li >= N_LAYERS:
                continue
        keep.setdefault(shard, []).append(name)

    tensors = {}
    for shard, names in sorted(keep.items()):
        path = os.path.join(SRC, shard)
        with safe_open(path, framework="pt") as f:
            for n in names:
                tensors[n] = f.get_tensor(n)
        print(f"  read {len(names):4d} tensors from {shard}")

    cfg = json.load(open(os.path.join(SRC, "config.json")))
    cfg["num_hidden_layers"] = N_LAYERS
    for k in ("layer_types", "num_attention_heads_per_layer", "mlp_layer_types"):
        if isinstance(cfg.get(k), list):
            cfg[k] = cfg[k][:N_LAYERS]
    cfg.pop("_source_repo", None)

    os.makedirs(OUT_CKPT, exist_ok=True)
    save_file(tensors, os.path.join(OUT_CKPT, "model.safetensors"), metadata={"format": "pt"})
    json.dump(cfg, open(os.path.join(OUT_CKPT, "config.json"), "w"), indent=1)
    for fn in ("configuration_laguna.py", "modeling_laguna.py", "tokenizer.json",
               "tokenizer_config.json", "special_tokens_map.json"):
        src = os.path.join(SRC, fn)
        if os.path.exists(src):
            shutil.copyfile(src, os.path.join(OUT_CKPT, fn))
    total = sum(t.numel() * t.element_size() for t in tensors.values())
    print(f"  wrote {len(tensors)} tensors, {total/1e9:.2f} GB -> {OUT_CKPT}")

    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        OUT_CKPT, trust_remote_code=True, dtype=torch.float32).eval()
    with torch.no_grad():
        ids = torch.tensor([PROMPT], dtype=torch.long)
        last = model(input_ids=ids, use_cache=False).logits[0, -1].float()
        cur, cont = list(PROMPT), []
        for _ in range(N_NEW):
            o = model(input_ids=torch.tensor([cur], dtype=torch.long), use_cache=False)
            cont.append(int(o.logits[0, -1].argmax()))
            cur.append(cont[-1])

    g0 = model.model.layers[0].self_attn.g_proj.weight.shape[0]
    golden = {
        "note": f"REAL Laguna-XS.2 weights, first {N_LAYERS} layers, CPU fp32 reference",
        "source": SRC, "n_layers": N_LAYERS,
        "prompt_ids": PROMPT, "n_new": N_NEW,
        "argmax": int(last.argmax()), "last_logits": last.tolist(),
        "continuation_ids": cont, "g_proj_rows_layer0": g0,
    }
    json.dump(golden, open(OUT_GOLDEN, "w"))
    print(f"  argmax={golden['argmax']} cont={cont} g_proj0={g0}")
    print(f"  -> {OUT_GOLDEN}")


if __name__ == "__main__":
    main()
