#!/usr/bin/env python3
"""G-W3b of docs/tasks/task-cuda-windowed-kv-2026-10.md: pool the ABBA passes of G-W3 (F W W F) and its order-reversed twin (W F F W).

    python3 windowed_kv_pool.py <pass1 rows.json> <pass2 rows.json>

Per class the pooled ratio is median(W rates over all four W sessions) / median(F rates over all four F sessions). The rule is in the task doc, section 12, written before
pass 2 ran. This script only computes: the verdict line repeats the rule so the log is self-describing.
"""
import json
import statistics
import sys

passes = [json.load(open(p))["rows"] for p in sys.argv[1:3]]
worst_pass = 9.0
pooled = {}
for i, rows in enumerate(passes, 1):
    for cls in ("short", "long"):
        f = [x["rate"] for x in rows if x["arm"] == "F" and x["cls"] == cls]
        w = [x["rate"] for x in rows if x["arm"] == "W" and x["cls"] == cls]
        r = statistics.median(w) / statistics.median(f)
        worst_pass = min(worst_pass, r)
        print(f"pass {i} {cls}: W/F {r:.3f}")
texts = {}
for rows in passes:
    for x in rows:
        texts.setdefault((x["cls"], x["prompt"]), set()).add(x["text"])
ident = all(len(v) == 1 for v in texts.values())
for cls in ("short", "long"):
    f = [x["rate"] for rows in passes for x in rows if x["arm"] == "F" and x["cls"] == cls]
    w = [x["rate"] for rows in passes for x in rows if x["arm"] == "W" and x["cls"] == cls]
    pooled[cls] = statistics.median(w) / statistics.median(f)
    print(f"pooled {cls}: W/F {pooled[cls]:.3f} (F n={len(f)}, W n={len(w)})")
print(f"texts identical across both passes, every session: {ident}")
worst = min(pooled.values())
if not ident or worst < 0.97 or worst_pass < 0.97:
    v = "FAIL: windowed KV does not become the default"
elif worst >= 0.98 and worst_pass >= 0.98:
    v = "PASS: windowed KV becomes the default"
else:
    v = "AMBIGUOUS -> parked: the owner decides"
print(f"G-W3b (pooled worst class {worst:.3f}, worst single pass-class {worst_pass:.3f}): {v}")
