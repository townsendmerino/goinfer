#!/usr/bin/env python3
"""SB12 step 10's C6: compare two `gate gpu` runs (a base and a step run) by what the gate decided.

    python3 compare_runs.py BASE_LOGDIR STEP_LOGDIR

Each LOGDIR is a run-gate-gpu-cuda.sh log directory: gate-gpu.log and records/*.json. It compares the records' verdict, group
counts, emitted groups, vacuous and empty cells and notes, and each section's state lines (PASS, FAIL, SKIP) in order, colour
codes and heartbeats stripped. Exit 0 when all are equal, 1 otherwise, printing each difference.
"""
import glob, json, re, sys

def run(d):
    rec = json.load(open(sorted(glob.glob(d + "/records/*.json"))[-1]))
    log = re.sub(r"\x1b\[[0-9;]*m", "", open(d + "/gate-gpu.log", encoding="utf-8").read())
    sections, cur = {}, None
    for ln in log.split("\n"):
        m = re.match(r"^== (.*) ==$", ln)
        if m:
            cur = m.group(1)
            sections.setdefault(cur, [])
        elif cur and re.match(r"^  (PASS|FAIL|SKIP)  ", ln):
            sections[cur].append(ln.strip())
    keys = ("verdict", "groups", "expect", "emitted", "vacuous_cells", "empty_cells", "notes", "heavy_skipped")
    return {k: rec.get(k) for k in keys}, sections

(rb, sb), (rs, ss) = run(sys.argv[1]), run(sys.argv[2])
bad = 0
for k in rb:
    if rb[k] != rs[k]:
        bad += 1
        print(f"record {k}: base {rb[k]!r}\n             step {rs[k]!r}")
for h in sorted(set(sb) | set(ss)):
    if h in ("provenance", "verdict"):
        continue
    if sb.get(h) != ss.get(h):
        bad += 1
        print(f"section {h!r}:\n  base {sb.get(h)}\n  step {ss.get(h)}")
print(f"{'EQUAL' if bad == 0 else f'{bad} difference(s)'}: verdict {rb['verdict']} / {rs['verdict']}, groups {rb['groups']} / {rs['groups']}")
sys.exit(1 if bad else 0)
