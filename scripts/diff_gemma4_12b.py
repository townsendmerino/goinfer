#!/usr/bin/env python3
"""Diff goinfer's 12B layer trace against the HF reference: per-layer cosine +
norm ratio on the last-position residual stream, and cosine on the layer 4/5
q/k/v probes. A sharp cosine cliff localizes a forward bug to its layer.

    ~/g4venv/bin/python scripts/diff_gemma4_12b.py
"""
import json, os, numpy as np

T = os.path.expanduser("~/mycode/goinfer/testdata")
hf = json.load(open(f"{T}/gemma4_12b_trace.json"))
gi = json.load(open(f"{T}/gemma4_12b_goinfer_trace.json"))

def cos(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

GLOBAL = set(range(5, 48, 6))  # full_attention layers: 5,11,17,23,29,35,41,47
print(f"HF argmax={hf['argmax']}  goinfer argmax={gi['argmax']}  ok={gi['argmax_ok']}")
# Alignment: HF output_hidden_states[0]=embedding, [i+1]=after layer i.
# goinfer trace: "-1"=embedding, "i"=after layer i.  => HF[i+1] <-> goinfer[i].
print("\nlayer  type  cosine    |gi|/|hf|")
worst = (1.0, None)
for i in range(-1, 48):
    gk = str(i)
    hk = str(i + 1)  # off-by-one: goinfer layer i == HF hidden_states[i+1]
    if hk not in hf["hidden_last"] or gk not in gi["hidden_last"]:
        continue
    h, g = hf["hidden_last"][hk], gi["hidden_last"][gk]
    c = cos(h, g)
    ratio = np.linalg.norm(g) / (np.linalg.norm(h) + 1e-12)
    typ = "GLB" if i in GLOBAL else ("emb" if i < 0 else "loc")
    flag = "  <-- DROP" if c < 0.97 else ""
    print(f"{i:5d}  {typ}   {c:.5f}   {ratio:6.3f}{flag}")
    if i >= 0 and c < worst[0]:
        worst = (c, i)
print(f"\nworst layer cosine: {worst[0]:.5f} at layer {worst[1]}")

print("\n=== layer 4/5 q/k/v probe cosine (goinfer vs HF) ===")
for li in (4, 5):
    for name in ("q", "k", "v"):
        gk = f"{li}.{name}"
        # HF probe stores post-norm q/k (q_norm/k_norm) and post-v_norm v, flattened
        hfkey = {"q": f"L{li}.q_norm", "k": f"L{li}.k_norm", "v": f"L{li}.v_norm"}[name]
        if gk not in gi["qkv"] or hfkey not in hf["probe"]:
            continue
        hv = np.asarray(hf["probe"][hfkey], float).reshape(-1)
        gv = np.asarray(gi["qkv"][gk], float).reshape(-1)
        note = ""
        if hv.shape != gv.shape:
            note = f"  (shape hf={hv.shape} gi={gv.shape})"
        n = min(len(hv), len(gv))
        print(f"L{li}.{name:1}  cos={cos(hv[:n], gv[:n]):.5f}{note}")
