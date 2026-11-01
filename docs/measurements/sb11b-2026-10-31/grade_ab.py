#!/usr/bin/env python3
"""G-SB11b-5's grade (docs/tasks/task-subsystems-2026-10.md §8, "SB11b"): every cell bench_peer.py measured for both
goinfer (this build) and goinfer_old (main before it) in a pass is a pair, its ratio new/old of the cells' median decode
tok/s. PASS: the median ratio over every pair of both passes >= 0.99. FAIL: under 0.98. Between: AMBIGUOUS, parked.
A cell started under heavy load is void (the 2026-10-19 rule: 1-min load above 8.0 on nobara) and listed, not paired.
Usage: grade_ab.py pass1.json pass2.json"""
import json
import statistics
import sys

pairs, void = [], []
for path in sys.argv[1:]:
    cells = [c for c in json.load(open(path)) if c.get("engine") in ("goinfer", "goinfer_old") and c.get("runs")]
    by = {}
    for c in cells:
        load = (c.get("machine") or {}).get("loadavg") or [0]
        key = (c.get("phase"), c.get("backend"), c.get("model"), c.get("depth"), c.get("config"))
        if load[0] > 8.0:
            void.append((path, c["engine"], key, load[0]))
            continue
        by.setdefault(key, {})[c["engine"]] = statistics.median(c["runs"])
    for key, e in sorted(by.items(), key=str):
        if "goinfer" in e and "goinfer_old" in e:
            pairs.append((path, key, e["goinfer"], e["goinfer_old"], e["goinfer"] / e["goinfer_old"]))
for p in pairs:
    print(f"{p[0]} {p[1]}: new {p[2]:.2f} old {p[3]:.2f} tok/s, ratio {p[4]:.4f}")
for v in void:
    print(f"VOID {v[0]} {v[1]} {v[2]}: load {v[3]}")
if not pairs:
    print("VERDICT G-SB11b-5: no pairs")
    sys.exit(1)
m = statistics.median(p[4] for p in pairs)
verdict = "PASS" if m >= 0.99 else ("FAIL" if m < 0.98 else "AMBIGUOUS (parked)")
print(f"VERDICT G-SB11b-5: {verdict}: median new/old {m:.4f} over {len(pairs)} pairs")
