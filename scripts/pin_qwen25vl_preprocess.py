#!/usr/bin/env python
"""Pin the Qwen2.5-VL image preprocessing (image bytes -> pixel_values + grid_thw)
against the HF Qwen2_5_VLImageProcessor — the P5.3 gate.

Two stages are tangled in the processor: smart_resize (resize to a multiple of
patch*merge within min/max pixels, PIL BICUBIC) and the exact part (rescale +
CLIP-normalize + the patchify rearrange into [n_patches, C*temporal*patch*patch]).
PIL-bicubic parity is a separate, fiddly port — so this pins a **pre-sized** image
(dims already multiples of patch*merge and within bounds) where PIL.resize is a
no-op (PIL returns a copy when size == input size). That gives goinfer a BIT-EXACT
gate on the normalize + patchify, decoupled from resize parity (a tolerance gate on
a non-pre-sized image is a follow-on, mirroring the Gemma 3 P1 strategy).

Writes the deterministic input PNG + the reference pixel_values/grid_thw.

    ~/.venv-vl/bin/python scripts/pin_qwen25vl_preprocess.py
    -> testdata/qwen25vl_preprocess_image.png
    -> testdata/qwen25vl_preprocess_golden.json
"""
import io
import json
import os

import numpy as np
from PIL import Image
from transformers import AutoImageProcessor

HERE = os.path.dirname(__file__)
MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"  # processor config (patch 14, merge 2, temporal 2, CLIP norm)
IMG = os.path.join(HERE, "..", "testdata", "qwen25vl_preprocess_image.png")
OUT = os.path.join(HERE, "..", "testdata", "qwen25vl_preprocess_golden.json")

# Pre-sized so smart_resize is identity: H,W multiples of patch*merge=28, pixel
# count within the processor's min/max. 56×84 → grid 4×6 (matches the image golden).
H, W = 56, 84


def main():
    proc = AutoImageProcessor.from_pretrained(MODEL, use_fast=False)  # PIL-based (no torchvision dep)
    # Force no rounding surprises: pin min/max so 56×84 (4704 px) neither up- nor
    # down-scales. (patch grid 4×6 = 24 patches.)
    proc.min_pixels = 4 * 28 * 28
    proc.max_pixels = 64 * 28 * 28

    # Deterministic RGB pattern (no external asset, reproducible bytes).
    yy, xx = np.mgrid[0:H, 0:W]
    r = (xx * 255 // W).astype(np.uint8)
    g = (yy * 255 // H).astype(np.uint8)
    b = ((xx + yy) * 255 // (W + H)).astype(np.uint8)
    arr = np.stack([r, g, b], axis=-1)
    img = Image.fromarray(arr, "RGB")
    img.save(IMG)

    # Re-load from the saved PNG so the bytes match exactly what goinfer decodes.
    with open(IMG, "rb") as f:
        png_bytes = f.read()
    img = Image.open(io.BytesIO(png_bytes)).convert("RGB")

    out = proc(images=img, return_tensors="np")
    pv = out["pixel_values"]              # [n_patches, C*temporal*patch*patch]
    grid = out["image_grid_thw"]          # [[t,h,w]]
    assert grid.tolist() == [[1, H // 14, W // 14]], grid.tolist()

    golden = {
        "note": "Qwen2.5-VL image preprocess; pre-sized 56×84 (smart_resize no-op) → exact pixel_values",
        "image_png": "qwen25vl_preprocess_image.png",
        "image_height": H, "image_width": W,
        "patch_size": 14, "spatial_merge_size": 2, "temporal_patch_size": 2,
        "image_mean": [float(x) for x in proc.image_mean],
        "image_std": [float(x) for x in proc.image_std],
        "grid_thw": grid.tolist(),
        "pixel_values_shape": list(pv.shape),
        "pixel_values": pv.reshape(-1).astype(float).tolist(),
    }
    with open(OUT, "w") as f:
        json.dump(golden, f)
    print(f"wrote {IMG}\nwrote {OUT}")
    print(f"  grid_thw={grid.tolist()}  pixel_values_shape={list(pv.shape)}  mean={golden['image_mean']}")


if __name__ == "__main__":
    main()
