#!/usr/bin/env python
"""Real-checkpoint gate for aikit/vision's Gemma4Encoder (P7 Phase A) — loads
the REAL google/gemma-4-E2B-it vision tower + embed_vision weights (not a tiny
random model) and runs a forward pass on fixed synthetic pre-patchified input,
dumping the projected soft-token output. The Go Gemma4Encoder loads the SAME
real checkpoint directory and must reproduce it (cosine gate, matched f32
precision on both sides — this session's own "matched precision" discipline).

Deliberately loads ONLY the vision_tower.*/embed_vision.* tensors via
safetensors' safe_open (not safetensors.torch.load_file, which would
materialize the whole ~10GB file including the full text decoder) — the real
E2B checkpoint at ~/models/gemma-4-E2B-unq.

    ~/g4venv/bin/python scripts/pin_gemma4_vision_real.py
    -> testdata/gemma4_vision_real_golden.json.gz (gzipped per CLAUDE.md's fixed-size-real-golden convention)
"""
import gzip
import json
import os

import torch
from safetensors import safe_open
from transformers.models.gemma4.configuration_gemma4 import Gemma4Config
from transformers.models.gemma4.modeling_gemma4 import Gemma4MultimodalEmbedder, Gemma4VisionModel

CKPT_DIR = os.path.expanduser("~/models/gemma-4-E2B-unq")
OUT = os.path.join(os.path.dirname(__file__), "..", "testdata", "gemma4_vision_real_golden.json.gz")

PATCH_SIZE = 16
POOLING_KERNEL_SIZE = 3
GRID = 6  # 96x96 at patch_size=16 -> 6x6=36 patches -> pooled 2x2=4 soft tokens


def main():
    with open(os.path.join(CKPT_DIR, "config.json")) as f:
        cfg_json = json.load(f)
    full_cfg = Gemma4Config(**cfg_json)
    vision_cfg = full_cfg.vision_config
    text_cfg = full_cfg.text_config
    print(f"vision_config: hidden={vision_cfg.hidden_size} layers={vision_cfg.num_hidden_layers} "
          f"heads={vision_cfg.num_attention_heads} head_dim={vision_cfg.head_dim} "
          f"use_clipped_linears={vision_cfg.use_clipped_linears} standardize={vision_cfg.standardize}")

    vision_model = Gemma4VisionModel(vision_cfg)
    embedder = Gemma4MultimodalEmbedder(vision_cfg, text_cfg)

    vt_sd, ev_sd = {}, {}
    st_path = os.path.join(CKPT_DIR, "model.safetensors")
    print(f"reading {st_path} (selective load: vision_tower.*/embed_vision.* only)")
    with safe_open(st_path, framework="pt") as f:
        for k in f.keys():
            if k.startswith("model.vision_tower."):
                vt_sd[k[len("model.vision_tower."):]] = f.get_tensor(k).float()
            elif k.startswith("model.embed_vision."):
                ev_sd[k[len("model.embed_vision."):]] = f.get_tensor(k).float()
    print(f"  loaded {len(vt_sd)} vision_tower tensors, {len(ev_sd)} embed_vision tensors")

    missing_vt, unexpected_vt = vision_model.load_state_dict(vt_sd, strict=False)
    missing_ev, unexpected_ev = embedder.load_state_dict(ev_sd, strict=False)
    if missing_vt or unexpected_vt:
        print(f"  WARNING vision_tower missing={missing_vt} unexpected={unexpected_vt}")
    if missing_ev or unexpected_ev:
        print(f"  WARNING embed_vision missing={missing_ev} unexpected={unexpected_ev}")
    vision_model.eval()
    embedder.eval()

    num_patches = GRID * GRID
    patch_dim = 3 * PATCH_SIZE * PATCH_SIZE
    gen = torch.Generator().manual_seed(1)
    patches = torch.rand(1, num_patches, patch_dim, generator=gen, dtype=torch.float32)

    xs, ys = torch.meshgrid(torch.arange(GRID), torch.arange(GRID), indexing="xy")
    position_ids = torch.stack([xs.reshape(-1), ys.reshape(-1)], dim=-1).unsqueeze(0).long()

    with torch.no_grad():
        vis_out = vision_model(pixel_values=patches, pixel_position_ids=position_ids)
        pooled = vis_out.last_hidden_state
        projected = embedder(pooled)

    golden = {
        "note": "REAL google/gemma-4-E2B-it vision_tower+embed_vision weights, synthetic patches, CPU fp32",
        "checkpoint": "gemma-4-E2B-it (via ~/models/gemma-4-E2B-unq)",
        "config": {
            "hidden_size": vision_cfg.hidden_size,
            "intermediate_size": vision_cfg.intermediate_size,
            "num_hidden_layers": vision_cfg.num_hidden_layers,
            "num_attention_heads": vision_cfg.num_attention_heads,
            "head_dim": vision_cfg.head_dim,
            "patch_size": PATCH_SIZE,
            "pooling_kernel_size": POOLING_KERNEL_SIZE,
            "position_embedding_size": vision_cfg.position_embedding_size,
            "rms_norm_eps": vision_cfg.rms_norm_eps,
            "use_clipped_linears": vision_cfg.use_clipped_linears,
        },
        "text_hidden_size": text_cfg.hidden_size,
        "num_patches": num_patches,
        "patches_shape": list(patches.shape),
        "patches": patches.flatten().tolist(),
        "position_ids_shape": list(position_ids.shape),
        "position_ids": position_ids.flatten().tolist(),
        "projected_shape": list(projected.shape),
        "projected": projected.flatten().float().tolist(),
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with gzip.open(OUT, "wt") as f:
        json.dump(golden, f)
    print(f"wrote {OUT}")
    print(f"  num_patches={num_patches} projected_shape={golden['projected_shape']}")


if __name__ == "__main__":
    main()
