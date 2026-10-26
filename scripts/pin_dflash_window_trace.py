#!/usr/bin/env python3
"""The DFlash drafter's trunk output from z-lab's reference code, for goinfer's per-layer masks
(docs/measurements/dflash-window-2026-10/README.md, G1 tier 1).

One block of 16 positions over fused context rows (after fc and hidden_norm) of each length in CTX, through the drafter's
layers and final norm, in f32 on the CPU. The inputs are not stored: both sides build them from hashed() below, so the golden
holds only the outputs. Run with a Python that has torch and transformers (on nobara, ~/g4venv):

    ~/g4venv/bin/python scripts/pin_dflash_window_trace.py --drafter ~/models/qwen35-9b-dflash \
        --out testdata/dflash_window_qwen35_9b_golden.json.gz
"""
import argparse
import gzip
import importlib.util
import json
import os
import tempfile
import urllib.request

import numpy as np
import torch

ZLAB_COMMIT = "07ebd93db9f472af339b644bb70221ad8428328a"
ZLAB_URL = f"https://raw.githubusercontent.com/z-lab/dflash/{ZLAB_COMMIT}/dflash/model.py"
CTX = (64, 4000, 4500)  # under, near and past the 4,096 window
BLOCK = 16


def hashed(seed: int, n: int) -> np.ndarray:
    """n floats in [-0.5, 0.5], a hash of (seed, index): decoder's dflashWindowHashed computes the same values."""
    i = np.arange(n, dtype=np.uint64) + np.uint64(seed) * np.uint64(0x9E3779B9)
    m = np.uint64(0xFFFFFFFF)
    h = (i * np.uint64(2654435761)) & m
    h ^= h >> np.uint64(15)
    h = (h * np.uint64(2246822519)) & m
    h ^= h >> np.uint64(13)
    return ((h % np.uint64(2001)).astype(np.float64) / 2000.0 - 0.5).astype(np.float32)


def load_zlab(path):
    if path is None:
        path = os.path.join(tempfile.mkdtemp(), "model.py")
        urllib.request.urlretrieve(ZLAB_URL, path)
    spec = importlib.util.spec_from_file_location("zlab_dflash_model", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drafter", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--zlab", help=f"z-lab's dflash/model.py at {ZLAB_COMMIT[:8]} (fetched when omitted)")
    a = ap.parse_args()
    zl = load_zlab(a.zlab)
    model = zl.DFlashDraftModel.from_pretrained(a.drafter, torch_dtype=torch.float32).eval()
    h = model.config.hidden_size
    cases = []
    with torch.no_grad():
        for n in CTX:
            fused = torch.from_numpy(hashed(1, n * h).reshape(1, n, h))
            block = torch.from_numpy(hashed(2, BLOCK * h).reshape(1, BLOCK, h))
            pos = torch.arange(n + BLOCK)[None]
            pe = model.rotary_emb(block, pos)
            hs = block
            for layer in model.layers:  # the forward after fc and hidden_norm: the context rows are already fused
                hs = layer(target_hidden=fused, hidden_states=hs, attention_mask=None, position_ids=pos,
                           position_embeddings=pe)
                if isinstance(hs, tuple):
                    hs = hs[0]
            out = model.norm(hs)[0]
            cases.append({"ctx": n, "out": out.tolist()})
            print(f"ctx {n}: out[0][:4] = {out[0, :4].tolist()}")
    rec = {"zlab_commit": ZLAB_COMMIT, "drafter": os.path.basename(os.path.normpath(a.drafter)), "block": BLOCK,
           "layer_types": getattr(model.config, "layer_types", None),
           "sliding_window": getattr(model.config, "sliding_window", None), "cases": cases}
    with gzip.open(a.out, "wt") as f:
        json.dump(rec, f)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
