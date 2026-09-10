#!/usr/bin/env python
"""Real-checkpoint gate for aikit/vision's Gemma4Encoder on the STANDARDIZE=TRUE geometry
(P7 real-validation pass) — the real google/gemma-4-26b-a4b-it vision tower + embed_vision
weights (hidden 1152, 27 layers, head_dim 72, use_clipped_linears=False), the twin of
pin_gemma4_vision_real.py (which pins the standardize=False E2B geometry; that one must stay
untouched as the existing regression pin). Loading a real 26B-A4B checkpoint's vision tower
through this script's own reference forward FIRST surfaced that goinfer's Go encoder had
`vision_config.standardize=True` unimplemented (a hard refusal at load time) — the fix adds the
missing `(x - std_bias) * std_scale` affine (Gemma4VisionModel.forward's own placement,
immediately after the pooler's root-hidden_size scaling); this script pins the real numeric
result the Go side must now reproduce.

Deliberately loads ONLY the vision_tower.*/embed_vision.* tensors via safetensors' safe_open
(not safetensors.torch.load_file, which would materialize the whole ~50GB sharded file) — the
real 26B-A4B checkpoint at ~/models/gemma-4-26b-a4b-it.

    ~/.venv-vl/bin/python scripts/pin_gemma4_vision_real_26b.py
    -> testdata/gemma4_vision_real_26b_golden.json.gz
"""
import gzip
import json
import os

import torch
from safetensors import safe_open
from transformers.models.gemma4.configuration_gemma4 import Gemma4Config
from transformers.models.gemma4.modeling_gemma4 import Gemma4MultimodalEmbedder, Gemma4VisionModel

CKPT_DIR = os.path.expanduser("~/models/gemma-4-26b-a4b-it")
OUT = os.path.join(os.path.dirname(__file__), "..", "testdata", "gemma4_vision_real_26b_golden.json.gz")

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
    if not vision_cfg.standardize:
        raise SystemExit("this checkpoint's vision_config.standardize is False — wrong pin script "
                          "(use pin_gemma4_vision_real.py for the standardize=False geometry)")

    vision_model = Gemma4VisionModel(vision_cfg)
    embedder = Gemma4MultimodalEmbedder(vision_cfg, text_cfg)

    vt_sd, ev_sd = {}, {}
    index_path = os.path.join(CKPT_DIR, "model.safetensors.index.json")
    with open(index_path) as f:
        weight_map = json.load(f)["weight_map"]
    shard_files = sorted({v for k, v in weight_map.items() if k.startswith("model.vision_tower.") or k.startswith("model.embed_vision.")})
    print(f"reading {len(shard_files)} shard(s) (selective load: vision_tower.*/embed_vision.* only)")
    for shard in shard_files:
        with safe_open(os.path.join(CKPT_DIR, shard), framework="pt") as f:
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
        "note": "REAL google/gemma-4-26b-a4b-it vision_tower+embed_vision weights "
                "(standardize=True geometry), synthetic patches, CPU fp32",
        "checkpoint": "gemma-4-26b-a4b-it",
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
            "standardize": vision_cfg.standardize,
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
