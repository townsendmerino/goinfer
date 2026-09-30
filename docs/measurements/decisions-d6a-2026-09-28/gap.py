#!/usr/bin/env python3
"""Why D6a's own OOD sample read ~14 points below the 150 fixture items for goinfer's bare-v1 label scoring.

    python3 gap.py > results/gap.txt      (run from this directory; reads only committed files)

Inputs: results/eval-B.jsonl (arm B, the 872-row OOD sample), results/b0cmp-goinfer-bare.jsonl (goinfer on the 150 D0
fixture items, calibration split) and testdata/decisions/route_a_b0.jsonl (transformers f32 on the same 150 items).
Four steps: (1) per-kind top-1 on both samples, which puts the whole gap on noul; (2) noul base rate, prediction rate and
per-class accuracy; (3) noul by source environment; (4) the reference against goinfer on the fixture's noul items, per
source, and a counterfactual that removes that per-source P(true) shift from the OOD sample. Step 4 is an ESTIMATE: the
shifts come from 7-17 items per source.
"""
import collections, json, os

HERE = os.path.dirname(os.path.abspath(__file__))
PROP = {"noul": 9767, "choice": 3219, "score": 72}


def load(p):
    return [json.loads(l) for l in open(os.path.join(HERE, p))]


def src(i):
    return "v3_*" if i.startswith("v3_") else i.split(":")[0]


def argmax(xs):
    return max(range(len(xs)), key=lambda i: xs[i])


oos = load("results/eval-B.jsonl")
fix = load("results/b0cmp-goinfer-bare.jsonl")
ref = {x["id"]: x for x in load("../../../testdata/decisions/route_a_b0.jsonl")}
gold = lambda x: x["target"][1] > x["target"][0]

print("(1) per-kind top-1, goinfer bare-v1")
for name, rows in (("OOD sample (D6a arm B)", oos), ("fixture (calibration split)", fix)):
    per = {}
    for k in PROP:
        r = [x for x in rows if x["kind"] == k]
        per[k] = sum(argmax(x["distribution"]) == argmax(x["target"]) for x in r) / len(r)
    w = sum(per[k] * PROP[k] for k in PROP) / sum(PROP.values())
    print(f"  {name:28s} " + "  ".join(f"{k} {per[k]:.3f}" for k in PROP) + f"   OOD-weighted {w:.3f}")

print("\n(2) noul: base rate and bias")
for name, rows in (("OOD sample", oos), ("fixture", fix)):
    r = [x for x in rows if x["kind"] == "noul"]
    g = [gold(x) for x in r]
    p = [x["distribution"][1] > 0.5 for x in r]
    at = sum(a for a, b in zip(p, g) if b) / sum(g)
    af = sum(not a for a, b in zip(p, g) if not b) / (len(r) - sum(g))
    print(f"  {name:12s} n={len(r):3d} gold-true {sum(g)/len(r):.3f} pred-true {sum(p)/len(r):.3f} "
          f"acc|true {at:.3f} acc|false {af:.3f} always-false {1-sum(g)/len(r):.3f}")

print("\n(3) noul by source environment")
for name, rows in (("OOD sample", oos), ("fixture", fix)):
    r = [x for x in rows if x["kind"] == "noul"]
    d = collections.defaultdict(list)
    for x in r:
        d[src(x["id"])].append(x)
    print(f"  {name}")
    for s, xs in sorted(d.items(), key=lambda kv: -len(kv[1])):
        g = [gold(x) for x in xs]
        p = [x["distribution"][1] > 0.5 for x in xs]
        print(f"    {s:22s} n={len(xs):3d} share {len(xs)/len(r):.2f} gold-true {sum(g)/len(xs):.2f} "
              f"pred-true {sum(p)/len(xs):.2f} top-1 {sum(a == b for a, b in zip(p, g))/len(xs):.2f}")

print("\n(4) reference (transformers f32) vs goinfer (Q4_K_M int4) on the fixture's noul items")
d = collections.defaultdict(list)
for x in fix:
    if x["kind"] == "noul":
        d[src(x["id"])].append((x, ref[x["id"]]))
shift = {}
for s, xs in sorted(d.items(), key=lambda kv: -len(kv[1])):
    g = [gold(x) for x, _ in xs]
    rp = [r["p"][1] for _, r in xs]
    gp = [x["distribution"][1] for x, _ in xs]
    shift[s] = sum(b - a for a, b in zip(rp, gp)) / len(xs)
    print(f"  {s:22s} n={len(xs):3d} ref pred-true {sum(v > 0.5 for v in rp)/len(xs):.2f} "
          f"goinfer pred-true {sum(v > 0.5 for v in gp)/len(xs):.2f}  mean P(true) shift {shift[s]:+.3f}")
allshift = sum(sum(x["distribution"][1] - r["p"][1] for x, r in xs) for xs in d.values()) / sum(len(xs) for xs in d.values())
r = [x for x in oos if x["kind"] == "noul"]
cf = sum(((x["distribution"][1] - shift.get(src(x["id"]), allshift)) > 0.5) == gold(x) for x in r) / len(r)
print(f"  all sources: mean shift {allshift:+.3f}")
print(f"  counterfactual (ESTIMATE): OOD noul top-1 {sum((x['distribution'][1] > 0.5) == gold(x) for x in r)/len(r):.3f} "
      f"as measured -> {cf:.3f} with each source's shift removed (unlisted sources: the all-source mean)")
