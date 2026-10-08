#!/usr/bin/env python3
"""G-S10b's real half (docs/tasks/task-multimodal-support-2026-10.md, S10): Qwen3-VL-2B's vision tower, every stage, from
transformers on the four F2a images. HF's own processor makes the pixel values (so the tower is compared on identical
inputs; preprocessing parity is a separate question), the tower runs in float32 with sdpa, and forward hooks record:
the input to block 0 (the patch embed plus the interpolated position rows), each block's output, the main merger's rows
and each DeepStack set. Written per image to <artifacts>/<img>.<stage>.f32 (float32 little-endian) with
<artifacts>/golden.json naming the grids and shapes; aikit's TestQwen3VisionEncoder_realDeepstack reads them.

Run (nobara, ~/.venv-vl): python3 scripts/pin_qwen3vl_tower_real.py --model ~/models/qwen3-vl-2b-instruct \
         --images testdata --artifacts ~/goinfer-logs/qwen3vl-tower
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import transformers
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

IMAGES = ["gemma3_preprocess_image.png", "qwen25vl_preprocess_image.png", "glm_ocr/formula.png", "glm_ocr/table.png"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--artifacts", required=True)
    a = ap.parse_args()
    model_dir = os.path.expanduser(a.model)
    if model_dir.startswith("/Volumes/") or model_dir.startswith("/srv/models"):
        sys.exit(f"{model_dir} is on the archive (CLAUDE.md)")
    art = os.path.expanduser(a.artifacts)
    os.makedirs(art, exist_ok=True)
    torch.set_num_threads(max(1, os.cpu_count() - 2))
    proc = AutoProcessor.from_pretrained(model_dir)
    model = Qwen3VLForConditionalGeneration.from_pretrained(model_dir, dtype=torch.float32, attn_implementation="sdpa").eval()
    vis = model.model.visual
    golden = {"transformers": transformers.__version__, "torch": torch.__version__, "hidden": vis.config.hidden_size,
              "out_hidden": vis.config.out_hidden_size, "deepstack": list(vis.config.deepstack_visual_indexes),
              "depth": len(vis.blocks), "images": []}
    for img in IMAGES:
        name = img.replace("/", "_").removesuffix(".png")
        im = Image.open(os.path.join(a.images, img)).convert("RGB")
        enc = proc.image_processor(images=[im], return_tensors="pt")
        px, grid = enc["pixel_values"].float(), enc["image_grid_thw"]
        st = {}
        hooks = [vis.blocks[0].register_forward_pre_hook(lambda mod, args, kw=None: st.__setitem__("embed", args[0].clone()))]
        for L, blk in enumerate(vis.blocks):
            hooks.append(blk.register_forward_hook(lambda mod, i, o, L=L: st.__setitem__(f"block{L}", (o[0] if isinstance(o, tuple) else o).clone())))
        with torch.no_grad():
            out = vis(px, grid)
        for h in hooks:
            h.remove()
        px.numpy().astype("<f4").tofile(os.path.join(art, name + ".pixels.f32"))
        for k, v in st.items():
            v.numpy().astype("<f4").tofile(os.path.join(art, f"{name}.{k}.f32"))
        out.pooler_output.numpy().astype("<f4").tofile(os.path.join(art, name + ".merged.f32"))
        for k, d in enumerate(out.deepstack_features):
            d.numpy().astype("<f4").tofile(os.path.join(art, f"{name}.deep{k}.f32"))
        golden["images"].append({"image": img, "name": name, "grid": grid[0].tolist(), "patches": int(px.shape[0]),
                                 "merged": int(out.pooler_output.shape[0])})
        print(f"{img}: grid {grid[0].tolist()}, {px.shape[0]} patches, {out.pooler_output.shape[0]} merged rows", file=sys.stderr)
    json.dump(golden, open(os.path.join(art, "golden.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
