"""Deterministic procedural test images for the Qwen3.5+ vision gates (P8a), and a dependency-free
PNG writer. No PIL/torchvision is available in the HF venvs on this box, and the point of a gate
image is that goinfer and HF see the SAME bytes, so the generator is numpy-only and the PNG is
written with zlib. Sizes are multiples of 32 (patch 16 x merge 2) and >= 65536 px, so the HF
processor's smart_resize is the identity and no resampler is exercised (the resize path is a
separate, disclosed gap — docs/measurements/p8a-qwen35-vl-2026-09/preregistration.md, G0b)."""
import struct
import zlib

import numpy as np


def make_image(kind, h, w):
    """uint8 [h, w, 3]."""
    rng = np.random.RandomState({"A": 11, "B": 22, "C": 33, "S": 44}[kind])
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    base = np.stack([
        128 + 100 * np.sin(xx / (w / 6.0) + rng.rand() * 6),
        128 + 100 * np.sin(yy / (h / 5.0) + rng.rand() * 6),
        128 + 100 * np.cos((xx + yy) / ((h + w) / 8.0)),
    ], axis=-1)
    img = base.copy()
    for _ in range(6):  # a few solid rectangles and discs so there is something to describe
        r, c = rng.randint(0, h - h // 4), rng.randint(0, w - w // 4)
        rh, rw = rng.randint(h // 10, h // 4), rng.randint(w // 10, w // 4)
        col = rng.randint(0, 256, 3)
        if rng.rand() < 0.5:
            img[r:r + rh, c:c + rw] = col
        else:
            m = (yy - (r + rh / 2)) ** 2 + (xx - (c + rw / 2)) ** 2 < (min(rh, rw) / 2) ** 2
            img[m] = col
    return np.clip(img, 0, 255).astype(np.uint8)


def write_png(path, arr):
    h, w, _ = arr.shape
    raw = b"".join(b"\x00" + arr[y].tobytes() for y in range(h))

    def chunk(tag, data):
        c = struct.pack(">I", len(data)) + tag + data
        return c + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n")
        f.write(chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)))
        f.write(chunk(b"IDAT", zlib.compress(raw, 9)))
        f.write(chunk(b"IEND", b""))


def hf_pixel_values(img, patch=16, merge=2, temporal=2,
                    mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
    """Transcription of Qwen2VLImageProcessor's torchvision path for a grid-aligned image:
    fused normalize (x - mean*255)/(std*255) in f32 (image_processing_backends.py
    _fuse_mean_std_and_rescale_factor), then patchify() exactly as in image_processing_qwen2_vl.py.
    Returns (pixel_values [gh*gw, C*T*P*P] f32, [t, gh, gw])."""
    h, w, _ = img.shape
    assert h % (patch * merge) == 0 and w % (patch * merge) == 0
    x = img.astype(np.float32).transpose(2, 0, 1)                      # C,H,W  (uint8 -> f32)
    scale = 1.0 / (1.0 / 255.0)  # HF: mean * (1.0 / rescale_factor), rescale_factor = 1/255
    m = np.array(mean, np.float32) * np.float32(scale)
    s = np.array(std, np.float32) * np.float32(scale)
    x = (x - m[:, None, None]) / s[:, None, None]
    x = x[None]                                                          # B,C,H,W
    gh, gw = h // patch, w // patch
    p = x.reshape(1, 3, gh // merge, merge, patch, gw // merge, merge, patch)
    p = p.transpose(0, 2, 5, 3, 6, 1, 4, 7)      # B, gh/m, gw/m, m, m, C, P, P
    p = np.broadcast_to(p[:, :, :, :, :, :, None, :, :],
                        p.shape[:6] + (temporal,) + p.shape[6:])
    p = np.ascontiguousarray(p).reshape(1, gh * gw, 3 * temporal * patch * patch)
    return p[0].astype(np.float32), [1, gh, gw]
