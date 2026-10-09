#!/usr/bin/env python3
"""G-31a2's grading (docs/tasks/task-multimodal-support-2026-10.md, "G-31a"): the 31B's int4-against-int8int8 agreement against the sibling's.
PASS: 31B >= sibling - 5.0 points; PARKED: 5.0-10.0 below; FAIL: more than 10.0 below or any non-finite logit.
Usage: g31a_grade.py <sibling qc json> <31B qc json>"""
import json, math, sys
def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h
sib, big = (json.load(open(a)) for a in sys.argv[1:3])
for d in (sib, big): d['Splits'] = d['Splits'] or []  # Go writes a nil slice as null
for tag, d in (("sibling", sib), ("31B    ", big)):
    lo, hi = wilson(d['Agree'], d['Positions'])
    near = sum(1 for s in d['Splits'] if s['RefMargin'] < 0.1)
    print(f"{tag} {d['Label']}: {d['Agree']}/{d['Positions']} = {100 * d['Agreement']:.2f}% (95% CI {100 * lo:.1f}-{100 * hi:.1f}); splits {len(d['Splits'])}, of which {near} at an int8int8 margin < 0.1; non-finite {d['NonFinite']}; synthetic={d['Synthetic']}; "
          f"load {d['LoadInt8Sec']:.0f}s/{d['LoadInt4Sec']:.0f}s, {d['DecodeInt8SecPerTok']:.2f}/{d['DecodeInt4SecPerTok']:.2f} s per token (int8int8/int4)")
gap = 100 * (sib['Agreement'] - big['Agreement'])
if big['NonFinite'] or sib['NonFinite'] or big['Synthetic'] or sib['Synthetic']:
    v = "FAIL (non-finite logits, or a synthetic run)"
elif gap <= 5.0: v = "PASS"
elif gap <= 10.0: v = "PARKED"
else: v = "FAIL"
print(f"G-31a2: sibling minus 31B = {gap:+.2f} points (negative: the 31B agrees MORE than the sibling; the bar is <= +5.0) -> {v}")
