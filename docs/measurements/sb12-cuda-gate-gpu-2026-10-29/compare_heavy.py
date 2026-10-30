#!/usr/bin/env python3
"""SB12 step 12's C6: each top-level test's result in a narrowed run's heavy cells against the same cells of a full-tier run.

    python3 compare_heavy.py BASE_LOGDIR STEP_LOGDIR

Both LOGDIRs hold `gate gpu -logdir` cell streams (gate_gpu_cuda-heavy-<group>.json, gate_gpu_cuda-drain.json,
gate_gpu_cuda-iso-<test>.json; a .gz is read when the plain file is absent). For every heavy cell the step run has, it prints
the tests whose final result differs and the tests in one run only. The drain group's cell in a narrowed run holds only the
drain tests inside the chosen groups, so for that cell the base's other tests are not listed. Exit 0 when nothing differs.
"""
import glob, gzip, json, os, re, sys

def results(path):
    op = gzip.open if path.endswith(".gz") else open
    final = {}
    with op(path, "rt", encoding="utf-8") as f:
        for ln in f:
            try:
                e = json.loads(ln)
            except ValueError:
                continue
            t = e.get("Test", "")
            if t and "/" not in t and e.get("Action") in ("pass", "fail", "skip"):
                final[t] = e["Action"]
    return final

def cells(d):
    out = {}
    for p in sorted(glob.glob(os.path.join(d, "gate_gpu_cuda-*.json*"))):  # a plain .json sorts before its .gz
        name = re.sub(r"^gate_gpu_|\.json(\.gz)?$", "", os.path.basename(p))
        if re.match(r"cuda-(heavy-|drain$|iso-)", name) and name not in out:
            out[name] = p
    return out

base, step = cells(sys.argv[1]), cells(sys.argv[2])
bad = same = 0
for name in sorted(step):
    s = results(step[name])
    if name not in base:
        print(f"{name}: not in the base run ({len(s)} test(s))")
        bad += 1
        continue
    b = results(base[name])
    for t in sorted(set(s) | set(b)):
        if t in s and t in b:
            if s[t] == b[t]:
                same += 1
            else:
                print(f"{name}: {t}: base {b[t]}, step {s[t]}")
                bad += 1
        elif t in s:
            print(f"{name}: {t}: step only ({s[t]})")
            bad += 1
        elif name != "cuda-drain":
            print(f"{name}: {t}: base only ({b[t]})")
            bad += 1
print(f"{same} test(s) with the same result; {bad} difference(s) across {len(step)} cell(s)")
sys.exit(1 if bad else 0)
