#!/usr/bin/env python3
"""Phase V's resize option (docs/tasks/task-embeddinggemma2.md): a small golden for embeddinggemma2's bicubic.

The reference processor (Gemma4ImageProcessor) resizes the uint8 image with torchvision's antialiased bicubic,
tvF.resize(img, [h, w], BICUBIC, antialias=True). This records, for seeded synthetic RGB images (noise, which clamps
hard, and a smooth gradient) over downscales, upscales, mixed and single-axis resizes, the input pixels and
torchvision's output, both as base64 HWC uint8, so embeddinggemma2's resizeBicubicRGB is held to it bit for bit
without the real checkpoint.

Run: python3 scripts/pin_embeddinggemma2_resize.py --out testdata/embeddinggemma2-resize/golden.json
(torch and torchvision; recorded in the golden.)
"""
import argparse
import base64
import json
import os

import numpy as np
import torch
import torchvision
import torchvision.transforms.v2.functional as tvF
from torchvision.transforms import InterpolationMode

CASES = [  # name, (h, w), (th, tw)
    ("down", (75, 100), (32, 48)),
    ("down-large", (120, 90), (16, 16)),
    ("up", (20, 30), (48, 64)),
    ("mixed", (37, 53), (80, 16)),
    ("width-only", (40, 100), (40, 33)),
    ("height-only", (90, 60), (33, 60)),
]


def image(kind, h, w, rng):
    if kind == "noise":
        return rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
    y, x = np.mgrid[0:h, 0:w]
    g = np.stack([255 * x / max(w - 1, 1), 255 * y / max(h - 1, 1), 127.5 + 127.5 * np.sin((x + y) / 5)], -1)
    return np.clip(np.round(g), 0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rng = np.random.default_rng(20261006)
    out = {"torch": torch.__version__, "torchvision": torchvision.__version__, "cases": []}
    for kind in ("noise", "gradient"):
        for name, (h, w), (th, tw) in CASES:
            src = image(kind, h, w, rng)
            t = torch.from_numpy(src).permute(2, 0, 1).contiguous()
            r = tvF.resize(t, size=[th, tw], interpolation=InterpolationMode.BICUBIC, antialias=True)
            dst = r.permute(1, 2, 0).contiguous().numpy()
            assert dst.dtype == np.uint8 and dst.shape == (th, tw, 3)
            out["cases"].append({"name": f"{kind}/{name}", "h": h, "w": w, "th": th, "tw": tw,
                                 "src": base64.b64encode(src.tobytes()).decode(),
                                 "dst": base64.b64encode(dst.tobytes()).decode()})
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f)


if __name__ == "__main__":
    main()
