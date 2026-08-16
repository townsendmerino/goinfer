#!/usr/bin/env python
"""Convert a z-lab DFlash drafter checkpoint from bf16 to f32 safetensors — the P10
increment-2 fixture, and the same move 05 made for the EAGLE-3 head (`.bin` → f32
safetensors) so the Go loader reads one format and parity runs at full precision.

DFlash ships ONLY the denoiser trunk: 5 Qwen3-style decoder layers + `fc` (fuses the 5
captured target hidden states) + `hidden_norm` + `norm`. There is no `embed_tokens`, no
`lm_head`, no Markov head and no confidence head — it reuses the TARGET's embedding and
LM head at both ends (verified: the 58 tensors below account for the published file size
to the byte). That is the structural difference from DSpark, and it is why this converts
in one pass with no vocab-sized tensors.

    ~/.venv-vl/bin/python scripts/convert_dflash_f32.py \
        ~/models/qwen3-4b-dflash ~/models/qwen3-4b-dflash-f32

Checkpoint: z-lab/Qwen3-4B-DFlash-b16 (MIT, documented; paired with Qwen/Qwen3-4B).
"""
import json
import os
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def main():
    src = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~/models/qwen3-4b-dflash")
    dst = os.path.expanduser(sys.argv[2] if len(sys.argv) > 2 else "~/models/qwen3-4b-dflash-f32")
    os.makedirs(dst, exist_ok=True)

    tensors, n_params = {}, 0
    with safe_open(os.path.join(src, "model.safetensors"), framework="pt") as f:
        for k in f.keys():
            t = f.get_tensor(k).to(torch.float32).contiguous()
            tensors[k] = t
            n_params += t.numel()
    save_file(tensors, os.path.join(dst, "model.safetensors"), metadata={"format": "pt"})

    # Carry the config across unchanged except dtype — the Go loader reads block_size,
    # target_layer_ids and mask_token_id out of it.
    cfg = json.load(open(os.path.join(src, "config.json")))
    cfg["dtype"] = "float32"
    cfg["torch_dtype"] = "float32"
    with open(os.path.join(dst, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
        f.write("\n")

    print(f"{len(tensors)} tensors, {n_params:,} params -> {dst}")
    print(f"  f32 bytes {n_params * 4:,} (+ header)")


if __name__ == "__main__":
    main()
