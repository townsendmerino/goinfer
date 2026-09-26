#!/usr/bin/env python3
"""R17 null-distribution table (docs/measurements/metal-decode-attn-r17-2026-09-25.md).

Parses every clean (executor-flushed) teacher-forced gate run and tabulates, per candidate, the mean KL over the chosen
prompts, its ratio to the exact arm, the paired per-prompt delta (mean, s.e., t), the lower-on count and the STRICT
§3.2 critC (mean <= exact AND lower on >= half the prompts), then summarises by group. Checks the exact arm is identical
(at printed precision) in every run.

  python3 null_table.py                           # all 10 prompts of set "a"
  python3 null_table.py --prompts=3,4,6,7,9,10    # only prompts whose set-A reference matches the snapshot text
"""
import math
import os
import re
import statistics
import sys

D = os.path.dirname(os.path.abspath(__file__))
args = sys.argv[1:]
subset = None
if args and args[0].startswith("--prompts="):
    subset = [int(x) for x in args.pop(0).split("=", 1)[1].split(",")]
logs = args or [os.path.join(D, f) for f in (
    "step2-fidelity-gate-flushed.log", "step2-null-gates.log", "step2-nudge-gates.log",
    "step2-nudge-matched-gates.log", "step2-vsum-gates.log")]

pr = re.compile(r'^\[(r17-gate|r2-gate3)\] set "a" K=3900 prompt +(\d+)/10 shipped\(agree=([\d.]+)% HF=(\d+)/64 '
                r'KL=([\d.]+)\) (\S+?)\(agree=([\d.]+)% HF=(\d+)/64 KL=([\d.]+)\)')
vr = re.compile(r'^=== (r17-gate|r2-gate3) \(candidate (\S+)\).*? — (.*) ===')
nudge = re.compile(r'^##### (?:MATCHED )?NUDGE \d+/\d+ k=(-?\d+)')
vsum = re.compile(r'^##### VSUM \d+/\d+ cand=(\S+) C=(\d+)')

runs, cur = [], {}
for lg in logs:
    if not os.path.exists(lg):
        continue
    label = None
    for line in open(lg, errors="replace"):
        m = nudge.match(line)
        if m:
            label = f"exact-nudge(k={int(m.group(1)):+d})"
            continue
        m = vsum.match(line)
        if m:
            label = f"{m.group(1)}(C={m.group(2)})" if m.group(1) == "exact-vchunk" else m.group(1)
            continue
        m = pr.match(line)
        if m:
            cur[int(m.group(2))] = (float(m.group(5)), float(m.group(9)))
            continue
        m = vr.match(line)
        if m:
            name = m.group(2) if m.group(1) == "r17-gate" else "attention_fa(S=prod, R2 gate)"
            if label and (name.startswith("exact-nudge") or name.startswith("exact-v")):
                name = label
            runs.append(dict(name=name, per=dict(cur)))
            cur, label = {}, None

exact_rows = {tuple(r['per'][p][0] for p in sorted(r['per'])) for r in runs}
print(f"{len(runs)} clean gate runs; exact arm identical (printed precision) in every run: {len(exact_rows) == 1}")
PS = subset or list(range(1, 11))
print(f"prompts used: {PS}" + ("" if subset is None else "  (subset)"))
print(f"{'candidate':34s} {'mean KL':>8s} {'ratio':>7s} {'paired Δ ± s.e.':>20s} {'t':>6s} {'lower':>6s}  strict critC")
rows = []
for r in runs:
    ex = statistics.mean(r['per'][p][0] for p in PS)
    c = statistics.mean(r['per'][p][1] for p in PS)
    d = [r['per'][p][1] - r['per'][p][0] for p in PS]
    mu, se = statistics.mean(d), statistics.stdev(d) / math.sqrt(len(d))
    lower = sum(1 for p in PS if r['per'][p][1] <= r['per'][p][0])
    strict = c <= ex and 2 * lower >= len(PS)
    rows.append((r['name'], c / ex, strict))
    print(f"{r['name']:34s} {c:8.4f} {c / ex:7.4f} {mu:+10.5f} ± {se:.5f} {mu / se if se else 0:6.2f} {lower:3d}/{len(PS):<2d}  "
          f"{'pass' if strict else 'FAIL'}")


def kof(n):
    return abs(int(n.split("k=")[1].rstrip(")"))) if "k=" in n else 0


def group(n):
    if n.startswith(('attention_fa', 'proto', 'r17-proto')):
        return 'split-kernel family (attention_fa + prototype, all S)'
    if n.startswith('exact-vchunk'):
        return 'exact, chunked V sum (removes the serial-V error)'
    if n.startswith('exact-vrev8'):
        return 'exact, 8-term V groups reversed (same class)'
    if n.startswith('exact-nudge') and 5 <= kof(n) <= 7:
        return 'exact, magnitude-matched scale nudges k=±5..7'
    if n.startswith(('exact-null', 'exact-nudge')):
        return 'exact, small rounding nulls (reversed q·k, k=±1..3)'
    return 'other'


print()
for g in ('exact, small rounding nulls (reversed q·k, k=±1..3)', 'exact, magnitude-matched scale nudges k=±5..7',
          'exact, 8-term V groups reversed (same class)', 'exact, chunked V sum (removes the serial-V error)',
          'split-kernel family (attention_fa + prototype, all S)'):
    rs = [x for x in rows if group(x[0]) == g]
    if rs:
        fails = sum(1 for x in rs if not x[2])
        print(f"  {g:55s} n={len(rs):2d}  ratio {min(x[1] for x in rs):.4f}..{max(x[1] for x in rs):.4f}  "
              f"mean {statistics.mean(x[1] for x in rs):.4f}  strict critC fails {fails}/{len(rs)}")
