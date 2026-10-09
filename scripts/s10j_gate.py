#!/usr/bin/env python3
"""G-S10j step 1, the instrument gate: does the Go dump reproduce the served first-token distribution?
For each arm (dsoff = the dump's "off", dson = "on") every one of the server's top-5 tokens must have a dump probability within 0.01 absolute.
Usage: ~/g4venv/bin/python -I scripts/s10j_gate.py <model dir> <served reference json> <dump dir>"""
import json, sys
import numpy as np
from transformers import AutoTokenizer
model_dir, ref_path, dump = sys.argv[1:4]
tok = AutoTokenizer.from_pretrained(model_dir)
ref = json.load(open(ref_path))
meta = json.load(open(f"{dump}/table.png.meta.json")); V = meta["vocab"]
ok = True
for served, arm in (("dsoff", "off"), ("dson", "on")):
    x = np.fromfile(f"{dump}/table.png.{arm}.f32", dtype="<f4").reshape(meta["steps"], V)[0].astype(np.float64)
    p = np.exp(x - x.max()); p /= p.sum()
    worst = 0.0
    parts = []
    for t, ps in ref[served]["run"]["top5"].items():
        i = tok.convert_tokens_to_ids(t)
        if i is None or i == tok.unk_token_id: i = tok.encode(t, add_special_tokens=False)[0]
        d = abs(float(p[i]) - ps); worst = max(worst, d)
        parts.append(f"{t!r} served {ps:.4f} dump {p[i]:.4f}")
    held = worst <= 0.01; ok &= held
    print(f"{arm:3s} ({served}): worst |diff| {worst:.4f} -> {'HELD' if held else 'NOT HELD'}   " + "; ".join(parts))
print("INSTRUMENT GATE:", "HELD" if ok else "NOT HELD")
sys.exit(0 if ok else 1)
