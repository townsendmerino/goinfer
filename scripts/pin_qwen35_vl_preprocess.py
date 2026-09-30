#!/usr/bin/env python
"""Pin the Qwen3.5+ image preprocessing (P8a, gate G0b) — numpy only, no PIL/torchvision.

For a GRID-ALIGNED image the HF processor's smart_resize and resize are identities, so
pixel_values is the fused normalize + patchify of the image bytes, and goinfer must reproduce it
BIT FOR BIT. This writes one small image + golden (patch 16, merge 2, temporal 2, mean = std = 0.5,
the real Qwen3.5 values; 64x96 so the golden is tiny and committed) and, with --g2 DIR, the three
real-size G2 images and their pixel_values (outside the repo).

    python scripts/pin_qwen35_vl_preprocess.py
    -> testdata/qwen35vl_preprocess_image.png, testdata/qwen35vl_preprocess_golden.json
    python scripts/pin_qwen35_vl_preprocess.py --g2 ~/models/qwen35vl_g2
    -> DIR/img_{A,B,C}.png, DIR/img_{A,B,C}.pv.f32 (raw little-endian), DIR/images.json
"""
import argparse
import json
import os

import numpy as np

from qwen35vl_images import hf_pixel_values, make_image, write_png

HERE = os.path.dirname(os.path.abspath(__file__))
TESTDATA = os.path.join(HERE, "..", "testdata")
G2_IMAGES = {"A": (256, 256), "B": (384, 256), "C": (320, 512)}  # (h, w), 64 / 96 / 160 merged tokens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--g2")
    a = ap.parse_args()
    if a.g2:
        d = os.path.expanduser(a.g2)
        os.makedirs(d, exist_ok=True)
        meta = {}
        for k, (h, w) in G2_IMAGES.items():
            img = make_image(k, h, w)
            pv, grid = hf_pixel_values(img)
            write_png(os.path.join(d, f"img_{k}.png"), img)
            pv.astype("<f4").tofile(os.path.join(d, f"img_{k}.pv.f32"))
            meta[k] = dict(h=h, w=w, grid_thw=grid, n_merged=grid[1] * grid[2] // 4)
            print(k, h, w, grid, pv.shape)
        json.dump(meta, open(os.path.join(d, "images.json"), "w"), indent=1)
        return
    img = make_image("S", 64, 96)
    pv, grid = hf_pixel_values(img)
    write_png(os.path.join(TESTDATA, "qwen35vl_preprocess_image.png"), img)
    json.dump(dict(patch_size=16, merge_size=2, temporal_patch_size=2, image_mean=[0.5] * 3,
                   image_std=[0.5] * 3, grid_thw=[grid], pixel_values_shape=list(pv.shape),
                   pixel_values=pv.reshape(-1).tolist()),
              open(os.path.join(TESTDATA, "qwen35vl_preprocess_golden.json"), "w"))
    print("wrote small golden", pv.shape, grid)


if __name__ == "__main__":
    main()
