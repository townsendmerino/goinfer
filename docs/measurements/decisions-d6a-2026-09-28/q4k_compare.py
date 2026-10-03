#!/usr/bin/env python3
"""D6a owed q4k measurement: the registered comparison (docs/measurements/decisions-d6a-2026-09-28.md, "The owed q4k measurement").

Per-item argmax agreement with the transformers f32 reference (testdata/decisions/route_a_b0.jsonl), overall and per kind, for goinfer's Route A at
--quant q4k on the CPU (results/b0cmp-goinfer-bare-q4k.jsonl) beside the int4 CUDA arm already on record (results/b0cmp-goinfer-bare.jsonl); the paired
discordant counts with an exact two-sided sign test; and top-1 against gold, OOD-weighted (noul 9,767 / choice 3,219 / score 72) with a Wilson 95% interval
on the noul cell. Run from the repo root: python3 docs/measurements/decisions-d6a-2026-09-28/q4k_compare.py
"""
import json, math, os
T = "testdata/decisions/"
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results") + "/"
load = lambda p: {json.loads(l)["id"]: json.loads(l) for l in open(p)}
items, ref = load(T + "items.jsonl"), load(T + "route_a_b0.jsonl")
q4, i4 = load(R + "b0cmp-goinfer-bare-q4k.jsonl"), load(R + "b0cmp-goinfer-bare.jsonl")
dist = lambda r: r["distribution"] if "distribution" in r else r["p"]
a = lambda p: max(range(len(p)), key=lambda i: p[i])


def wil(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


W = {"noul": 9767, "choice": 3219, "score": 72}
for kind in ("noul", "choice", "score", "all"):
    ids = [k for k, v in items.items() if kind == "all" or v["kind"] == kind]
    n = len(ids)
    ag4 = [a(dist(ref[k])) == a(dist(q4[k])) for k in ids]
    agi = [a(dist(ref[k])) == a(dist(i4[k])) for k in ids]
    only4 = sum(x and not y for x, y in zip(ag4, agi)); onlyi = sum(y and not x for x, y in zip(ag4, agi))
    m, k = only4 + onlyi, min(only4, onlyi)
    p = min(1, 2 * sum(math.comb(m, j) for j in range(k + 1)) / 2 ** m) if m else 1
    lo, hi = wil(sum(ag4), n)
    tv4 = sum(0.5 * sum(abs(x - y) for x, y in zip(dist(ref[i]), dist(q4[i]))) for i in ids) / n
    tvi = sum(0.5 * sum(abs(x - y) for x, y in zip(dist(ref[i]), dist(i4[i]))) for i in ids) / n
    print(f"{kind:6} n={n:3} argmax agree q4k {sum(ag4)/n:.3f} [{lo:.2f},{hi:.2f}] vs int4 {sum(agi)/n:.3f}; TV q4k {tv4:.3f} vs int4 {tvi:.3f}; "
          f"discordant q4k-only {only4}, int4-only {onlyi}, sign-test p={p:.2g}")
for name, S in (("transformers f32", ref), ("goinfer q4k CPU", q4), ("goinfer int4 CUDA", i4)):
    per = {}
    for kind in W:
        ids = [x for x, v in items.items() if v["kind"] == kind]
        per[kind] = sum(a(dist(S[x])) == a(items[x]["target"]) for x in ids) / len(ids)
    w = sum(per[k] * W[k] for k in W) / sum(W.values())
    ids = [x for x, v in items.items() if v["kind"] == "noul"]
    k = sum(a(dist(S[x])) == a(items[x]["target"]) for x in ids)
    print(f"{name:18} top-1 vs gold {{noul {per['noul']:.3f}, choice {per['choice']:.3f}, score {per['score']:.3f}}}  OOD-weighted {w:.3f}   noul {k}/51 Wilson95 ({wil(k,51)[0]:.2f}, {wil(k,51)[1]:.2f})")
