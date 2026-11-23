#!/usr/bin/env python3
"""Grades ttft_ab.py's output by the four-condition rule's speed condition (CLAUDE.md, "Measurement discipline"): per pair (one
block, both arms), base TTFT / prefetch TTFT; the prefetch is faster on a pair when that ratio is above 1. Reports the median ratio,
its range, and how many of the pairs are faster. The other three conditions (bit-identical outputs, memory not higher, replaces
code) are the README's.

  grade.py ttft.json
"""
import json, statistics, sys

rows = json.load(open(sys.argv[1]))
by = {}
for r in rows:
    if r.get("valid") and r.get("ttft_ms_median"):
        by.setdefault(r["block"], {})[r["arm"]] = r["ttft_ms_median"]
ratios = [v["base"] / v["prefetch"] for b, v in sorted(by.items()) if {"base", "prefetch"} <= set(v)]
if not ratios:
    sys.exit("no complete pairs")
faster = sum(x > 1 for x in ratios)
print("pairs %d; prefetch faster on %d; median base/prefetch %.4f (range %.4f-%.4f)" % (len(ratios), faster, statistics.median(ratios), min(ratios), max(ratios)))
print("base TTFT ms by block:", [round(by[b]["base"], 1) for b in sorted(by) if "base" in by[b]])
print("prefetch TTFT ms by block:", [round(by[b]["prefetch"], 1) for b in sorted(by) if "prefetch" in by[b]])
print("SPEED CONDITION", "MET" if len(ratios) >= 10 and faster >= 9 else "NOT MET", "(at least 9 of 10 pairs faster)")
