#!/usr/bin/env python3
"""G-S10q-c's controls (docs/tasks/task-multimodal-support-2026-10.md, "S10, Qwen3-VL MoE"), run by day before the 30B is
queued: the layer-streaming reference (scripts/q3vlmoe_hf_stream.py) on the tiny Qwen3-VL MoE (testdata/qwen3vlmoe-tiny,
its experts on disk in the 4.57 layout, so HF's own conversion is exercised too) against an ordinary Hugging Face float32
forward of the same checkpoint, on two image sequences.
  control 1: stream against ordinary, relative max|diff| <= 1e-4 over every graded position;
  control 2: each planted defect in the stream (nodeepstack, swap, norenorm) breaks control 1's bar.

    ~/.venv-vl/bin/python -I scripts/q3vlmoe_stream_controls.py <tiny dir> <work dir>
"""
import json
import os
import shutil
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import q3vlmoe_hf_stream as S  # noqa: E402

tiny, work = sys.argv[1], sys.argv[2]
ck = os.path.join(work, "qwen3vlmoe-tiny-ctl")
shutil.rmtree(ck, ignore_errors=True)
shutil.copytree(tiny, ck)
cfg = json.load(open(os.path.join(ck, "config.json")))
vc = cfg["vision_config"]
# an image processor at the tiny tower's patch: patch 4, merge 2, temporal 2 (the real Qwen3-VL processor's class)
json.dump({"image_processor_type": "Qwen2VLImageProcessorFast", "patch_size": vc["patch_size"], "merge_size": vc["spatial_merge_size"],
           "temporal_patch_size": vc["temporal_patch_size"], "image_mean": [0.5, 0.5, 0.5], "image_std": [0.5, 0.5, 0.5],
           "do_resize": True, "size": {"shortest_edge": 16, "longest_edge": 64 * 64}},
          open(os.path.join(ck, "preprocessor_config.json"), "w"))
rng = np.random.default_rng(3)
seqs = []
for k, (h, w) in enumerate([(32, 48), (40, 24)]):
    p = os.path.join(work, f"ctl{k}.png")
    Image.fromarray(rng.integers(0, 256, (h, w, 3)).astype(np.uint8), "RGB").save(p)
    _, grid = S.image_inputs(__import__("transformers").AutoImageProcessor.from_pretrained(ck), p)
    n = int(grid.prod()) // (vc["spatial_merge_size"] ** 2)
    ids = [5, 17, cfg["vision_start_token_id"]] + [cfg["image_token_id"]] * n + [cfg["vision_end_token_id"], 41, 52]
    seqs.append({"ids": ids, "path": rng.integers(0, 490, 6).tolist(), "image": p})
ref = np.concatenate(S.ordinary(ck, seqs))
scale = float(np.abs(ref).max())
bar = 1e-4
results = {}
for defect in ["", "nodeepstack", "swap", "norenorm"]:
    got = np.concatenate(S.stream(ck, seqs, defect))
    rel = float(np.abs(got - ref).max()) / scale
    results[defect or "stream"] = rel
    print(f"[controls] {defect or 'stream (control 1)':28s} relative max|diff| {rel:.3e}  ({'within' if rel <= bar else 'beyond'} {bar:g})", flush=True)
ok = results["stream"] <= bar and all(results[d] > bar for d in ["nodeepstack", "swap", "norenorm"])
print("[controls] " + ("PASS: control 1 within the bar, every planted defect beyond it" if ok else "FAIL"), flush=True)
json.dump(results, open(os.path.join(work, "controls.json"), "w"), indent=1)
sys.exit(0 if ok else 1)
