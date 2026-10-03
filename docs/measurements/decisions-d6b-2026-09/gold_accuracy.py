#!/usr/bin/env python3
"""D6b, AFTER THE FACT (2026-10-02), not part of the verdict: each arm's top-1 accuracy against GOLD on the 84 gold rows, paired against the
transformers f32 reference's own accuracy, with a bootstrap of the paired difference. The registered grader (grade.py) measures agreement with the f32
reference and calibration; it does not report accuracy against gold, which is what the default-precision decision most depends on.
    python3 docs/measurements/decisions-d6b-2026-09/gold_accuracy.py     (run from the repo root)"""
import json, random
def load(p): return {json.loads(l)['id']: json.loads(l) for l in open(p)}
items = load('testdata/decisions/items.jsonl'); ref = load('testdata/decisions/jev9b_ref_f32.jsonl')
d = 'docs/measurements/decisions-d6b-2026-09/results/'
arms = {a: load(d + a + '.jsonl') for a in ('f32', 'int8int8', 'int4', 'int4-cuda')}
am = lambda v: max(range(len(v)), key=lambda i: v[i])
gold = [i for i, x in items.items() if x['gold']]
dist = {'reference': {i: ref[i]['p'] for i in ref}}
for a, rows in arms.items(): dist[a] = {i: r['distribution'] for i, r in rows.items()}
acc = {n: [am(rows[i]) == am(items[i]['target']) for i in gold] for n, rows in dist.items()}
print(f"gold rows: {len(gold)}")
for n, ok in acc.items(): print(f"{n:10s} accuracy vs gold: {sum(ok)}/{len(ok)} = {sum(ok)/len(ok):.3f}")
rnd = random.Random(1)
print("paired against the reference:")
for a in arms:
    diff = [int(x) - int(y) for x, y in zip(acc[a], acc['reference'])]
    b = sorted(sum(rnd.choice(diff) for _ in diff) / len(diff) for _ in range(5000))
    print(f"  {a:10s}: gained {diff.count(1):2d}, lost {diff.count(-1):2d} of {len(diff)}; mean {sum(diff)/len(diff):+.3f}, 95% bootstrap [{b[125]:+.3f}, {b[4875]:+.3f}]")
for kind in ('choice', 'noul', 'score'):
    g = [i for i in gold if items[i]['kind'] == kind]
    print(f"  gold {kind:6s} n={len(g):2d}: " + "  ".join(f"{n} {sum(am(dist[n][i]) == am(items[i]['target']) for i in g)}" for n in dist))
