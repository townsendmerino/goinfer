#!/usr/bin/env python3
"""The Q1 split's derived readout (README.md, "Q1 split, read"): from the two goinfer arms' full logit rows (not
committed: 18.5 MB each and incompressible) to each arm's per-step top-20, the KL and raw-logit cosine between them, and
the steps where an arm's argmax is not the forced token.
Usage: read.py <dir holding cuda-rows.f32, cpu-rows.f32, forced.json> <tokenizer.json> <out.json>"""
import json
import sys

import numpy as np

d, tokpath, out = sys.argv[1:4]
forced = json.load(open(d + "/forced.json"))
steps = forced["name_step"] + 1
tok = json.load(open(tokpath))
vocab = {i: s for s, i in tok["model"]["vocab"].items()}
for a in tok.get("added_tokens", []):
    vocab[a["id"]] = a["content"]


def lsm(x):
    x = x - x.max()
    return x - np.log(np.exp(x).sum())


R = {a: np.fromfile(f"{d}/{a}-rows.f32", dtype="<f4").reshape(steps, -1).astype(np.float64) for a in ("cuda", "cpu")}
res = {"steps": []}
for s in range(steps):
    row = {"step": s, "forced": forced["forced"][s] if s < len(forced["forced"]) else None}
    L = {a: lsm(R[a][s]) for a in R}
    for a in R:
        top = np.argsort(-L[a])[:20]
        row[a] = [[int(i), vocab.get(int(i), "?"), round(float(L[a][i]), 4)] for i in top]
    p = np.exp(L["cpu"])
    row["kl_cpu_cuda"] = round(float((p * (L["cpu"] - L["cuda"])).sum()), 5)
    a, b = R["cuda"][s], R["cpu"][s]
    row["cos_raw"] = round(float(a @ b / np.sqrt((a @ a) * (b @ b))), 5)
    res["steps"].append(row)
with open(out, "w") as fh:
    json.dump(res, fh, indent=1)
for r in res["steps"]:
    f = r["forced"]
    am = {a: r[a][0][0] for a in R}
    flag = ""
    if f is not None and any(am[a] != f for a in R):
        flag = "  forced %r; argmax cuda %r, cpu %r" % (vocab[f], vocab[am["cuda"]], vocab[am["cpu"]])
    print(f"step {r['step']:2d} KL(P||C) {r['kl_cpu_cuda']:.4f} cos {r['cos_raw']:.5f}{flag}")
