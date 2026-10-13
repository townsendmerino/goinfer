#!/usr/bin/env python3
"""Grades run-kv16-speed.sh against the bands pre-registered in docs/tasks/task-cuda-f16-kv-2026-10.md.

Per cell (model, depth): the ratio of the f16 arm's median per-run decode rate to the f32 arm's, pooled over both arm orders. Ships when every cell is >= 0.98, parked when the
worst is in 0.95-0.98, killed when any is below 0.95. Also prints each order alone, so an order effect shows. A cell with no rate on either arm is VOID, not a pass.
usage: grade-speed.py <outdir>   (reads <outdir>/*.json)
"""
import glob, json, os, statistics, sys

out = sys.argv[1]
rates = {}  # (model, depth, arm, order) -> [per-run means]
for f in sorted(glob.glob(os.path.join(out, "*.json"))):
    order = "f16first" if "f16first" in f else "f32first"
    for c in json.load(open(f)):
        if c.get("kind") or not c.get("engine") or c.get("error"):
            continue
        arm = "f16" if c["engine"] == "goinfer" else "f32"
        rates.setdefault((c["model"], c["depth"], arm, order), []).extend(c.get("runs") or [])
cells = sorted({(m, d) for (m, d, _, _) in rates})
worst, void = None, 0
print(f"{'cell':<18}{'f16':>9}{'f32':>9}{'ratio':>8}   f16-first  f32-first   n(f16/f32)")
for m, d in cells:
    a = {arm: [x for o in ("f16first", "f32first") for x in rates.get((m, d, arm, o), [])] for arm in ("f16", "f32")}
    if not a["f16"] or not a["f32"]:
        print(f"{m} d={d:<12} VOID (a missing arm)")
        void += 1
        continue
    r = statistics.median(a["f16"]) / statistics.median(a["f32"])
    per = []
    for o in ("f16first", "f32first"):
        x, y = rates.get((m, d, "f16", o)), rates.get((m, d, "f32", o))
        per.append(f"{statistics.median(x) / statistics.median(y):.3f}" if x and y else "  - ")
    print(f"{m + ' d=' + str(d):<18}{statistics.median(a['f16']):9.1f}{statistics.median(a['f32']):9.1f}{r:8.3f}   {per[0]:>9}  {per[1]:>9}   {len(a['f16'])}/{len(a['f32'])}")
    worst = r if worst is None else min(worst, r)
if worst is None or void:
    print(f"VERDICT: VOID ({void} void cells)")
elif worst >= 0.98:
    print(f"VERDICT: SHIPS (worst cell {worst:.3f} >= 0.98)")
elif worst >= 0.95:
    print(f"VERDICT: PARKED (worst cell {worst:.3f} in 0.95-0.98)")
else:
    print(f"VERDICT: KILLED (worst cell {worst:.3f} < 0.95)")
