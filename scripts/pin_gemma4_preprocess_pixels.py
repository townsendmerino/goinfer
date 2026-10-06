#!/usr/bin/env python3
"""F1c of docs/multimodal.md "Finishing this doc": transformers' Gemma4ImageProcessor (defaults: max_soft_tokens 280,
resample=3, torchvision backend), processor only, on four repo images that are not at their target size. Records each
image's patch count, positions, and the SHA-256 of its valid patches as uint8 (round(x*255), row-major patches,
(row, col, channel) within one), so aikit's vision.Gemma4Preprocess is held to it bit for bit without committing
megabytes of pixels.

Run: python3 scripts/pin_gemma4_preprocess_pixels.py --out testdata/gemma4-preprocess-pixels.json
"""
import argparse
import hashlib
import json

import numpy as np
import torch
import torchvision
import transformers
from PIL import Image
from transformers.models.gemma4.image_processing_gemma4 import Gemma4ImageProcessor

IMAGES = ["testdata/gemma3_preprocess_image.png", "testdata/qwen25vl_preprocess_image.png",
          "testdata/glm_ocr/formula.png", "testdata/glm_ocr/table.png"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    proc = Gemma4ImageProcessor()
    out = {"transformers": transformers.__version__, "torch": torch.__version__, "torchvision": torchvision.__version__,
           "processor": {k: getattr(proc, k, None) for k in ["max_soft_tokens", "resample", "patch_size", "pooling_kernel_size"]},
           "images": []}
    out["processor"]["resample"] = int(out["processor"]["resample"])
    for path in IMAGES:
        img = Image.open(path).convert("RGB")
        r = proc(images=[img], return_tensors="pt")
        pv, pos = r["pixel_values"][0], r["image_position_ids"][0]
        valid = pos[:, 0] >= 0
        pv, pos = pv[valid].numpy(), pos[valid].numpy()
        u8 = np.round(pv * 255).astype(np.uint8)
        if np.abs(u8.astype(np.float32) / 255 - pv).max() > 1e-6:
            raise SystemExit(f"{path}: pixels are not k/255")
        out["images"].append({"path": path, "w": img.width, "h": img.height, "n_patches": int(len(pos)),
                              "positions": pos.astype(int).tolist(), "sha256": hashlib.sha256(u8.tobytes()).hexdigest()})
        print(f"{path}: {img.width}x{img.height}, {len(pos)} patches")
    with open(a.out, "w") as f:
        json.dump(out, f)


if __name__ == "__main__":
    main()
