#!/usr/bin/env python3
"""Gate AX2b-ID: compare two tower-identity records (internal/towerid's JSON) case by case.

    ax2_compare.py <base.json> <head.json>

Pass: the same backend, GOOS and GOARCH, the same cases in the same order, and every case's sha256, values and device
flag equal. Anything else is FAIL; there is no tolerance (docs/tasks/task-subsystems-2026-10.md §7.8).
"""
import json
import sys


def main():
    a, b = (json.load(open(p)) for p in sys.argv[1:3])
    bad = 0
    for k in ("backend", "goos", "goarch"):
        if a[k] != b[k]:
            print(f"FAIL {k}: {a[k]} vs {b[k]}")
            bad += 1
    keys = lambda r: [(c["family"], c["input"]) for c in r["cases"]]
    if keys(a) != keys(b):
        print(f"FAIL cases differ: {sorted(set(keys(a)) ^ set(keys(b)))}")
        bad += 1
    for x, y in zip(a["cases"], b["cases"]):
        if x != y:
            print(f"FAIL {x['family']} {x['input']}: {x['sha256'][:16]} device={x['device']} vs {y['sha256'][:16]} device={y['device']}")
            bad += 1
    print(f"{a['backend']} {a['goos']}/{a['goarch']}: {len(a['cases'])} cases, {bad} differ: {'PASS' if bad == 0 else 'FAIL'}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
