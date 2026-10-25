#!/usr/bin/env python3
"""KS-F2's G2 (README.md here): the small-gains rule per cell, from run-nobara.sh's JSONL (one line per run: arm A old, B new; tag warm1,
warm2 or pair0..pair9). A pair's ratio is B's decode ms per token over A's; B is faster when the ratio is below 1. Warm runs are dropped.

  SHIP  every cell has at least 9 of 10 pairs faster
  FAIL  any cell has at most 1 of 10 pairs faster
  PARK  otherwise

A run is VOID, and its pair is dropped, when it started at a load above 8.0, when it swapped (vmswap_peak_mib > 0), or when it started
with less available memory than the cell's need plus 2 GB. The need is the fit run's peak (need_mib, which run-nobara.sh writes on every
line); a record written before need_mib existed is checked against the largest peak any of the cell's own runs reached, which is at least
the fit's. A cell without 10
complete pairs is INCOMPLETE. It prints each pair, each cell's count and median ratio, and the verdict. G0, G1 and the memory statement
are graded in the record, not here. Usage: grade_g2.py <cell.jsonl> ...
"""
import json, os, statistics, sys

counts = []
for path in sys.argv[1:]:
    cell = os.path.basename(path).removeprefix("g2-").removesuffix(".jsonl")
    runs = {}
    lines = [json.loads(line) for line in open(path)]
    fallback = max(r.get("vmhwm_mib", 0) for r in lines)
    if any("need_mib" not in r for r in lines):
        print(f"{cell}: need_mib absent from some lines; their need is the cell's largest measured peak, {fallback} MiB")
    for r in lines:
        if not r["tag"].startswith("pair"):
            continue
        need = r.get("need_mib", fallback)
        why = []
        if r.get("load", 0) > 8.0:
            why.append(f"load {r['load']} > 8.0")
        if r.get("vmswap_peak_mib", 0) > 0:
            why.append(f"swap peak {r['vmswap_peak_mib']} MiB")
        if r.get("avail_mib", 0) < need + 2048:
            why.append(f"available {r['avail_mib']} MiB < need {need} + 2048")
        if why:
            print(f"{cell} {r['tag']} {r['arm']}: VOID ({'; '.join(why)})")
            continue
        runs.setdefault(r["tag"], {})[r["arm"]] = r
    ratios = []
    for tag in sorted(runs, key=lambda t: int(t[4:])):
        a, b = runs[tag].get("A"), runs[tag].get("B")
        if not a or not b:
            print(f"{cell} {tag}: incomplete ({sorted(runs[tag])})")
            continue
        ratio = b["decode_ms_per_token"] / a["decode_ms_per_token"]
        ratios.append(ratio)
        print(f"{cell} {tag}: A {a['decode_ms_per_token']:.2f} ms, B {b['decode_ms_per_token']:.2f} ms, ratio {ratio:.4f}"
              f" (loads {a['load']}, {b['load']}; swap peaks {a['vmswap_peak_mib']}, {b['vmswap_peak_mib']} MiB)")
    faster = sum(1 for x in ratios if x < 1)
    med = statistics.median(ratios) if ratios else float("nan")
    print(f"== {cell}: {faster} of {len(ratios)} pairs faster, median ratio {med:.4f} ({1 / med:.3f}x)\n")
    counts.append((cell, faster, len(ratios), med))

# Francis's standing merge rule (2026-10-25): median >= 1.05x (ratio <= 1/1.05) and at least 9 of 10 pairs faster, in every cell.
merge = all(f >= 9 and m <= 1 / 1.05 for _, f, _, m in counts)
print(f"merge bar (median >= 1.05x and >= 9 of 10 pairs faster, every cell): {'met' if merge else 'not met'}")
counts = [c[:3] for c in counts]
if any(n != 10 for _, _, n in counts):
    print("VERDICT INCOMPLETE: a cell has fewer than 10 complete pairs")
    sys.exit(1)
if all(f >= 9 for _, f, _ in counts):
    print("VERDICT SHIP (the small-gains rule: at least 9 of 10 pairs faster in every cell)")
elif any(f <= 1 for _, f, _ in counts):
    print("VERDICT FAIL (a cell has at most 1 of 10 pairs faster)")
    sys.exit(1)
else:
    print("VERDICT PARK (some cell has 2 to 8 of 10 pairs faster: the owner decides)")
    sys.exit(3)
