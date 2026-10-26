#!/usr/bin/env python3
"""night_census.py — what the night queue on this machine ran, and how often a job was refused or voided.

Read-only. Reads scripts/night.py's per-night records: <runs>/<date>/SUMMARY.md (one table row per job, retries
included) and the job logs beside it.

  usage: night_census.py [--runs ~/goinfer-logs/night/runs] [--since YYYY-MM-DD]

A "refusal or void" is a log line matching one of PATTERNS: the fit guard declining a load, a pair voided by the
heavy-load rule, or a VOID-refused verdict. A log can match more than one pattern; it is counted once in the total.
"""
import argparse
import collections
import glob
import os
import re

PATTERNS = {
    "was NOT loaded": re.compile(r"was NOT loaded"),
    "would page to swap rather than run": re.compile(r"would page to swap rather than run"),
    "VOID pair": re.compile(r"(?i)\bvoid pair\b"),
    "VOID-refused": re.compile(r"VOID-refused"),
}
ROW = re.compile(r"^\| (\d+) \| ([^|]+) \| ([^|]+) \| ([^|]+) \| ([^|]+) \|.*`([^`]+\.log)`")
RAN = re.compile(r"(?:(\d+) h )?(\d+) min")
# Why a job that did not end ok did not: the first class whose pattern its log matches, in this order.
CAUSES = [
    ("memory guard refused the load", re.compile(r"was NOT loaded|would page to swap rather than run|fit check read REFUSED")),
    ("voided (heavy load or a guard)", re.compile(r"VOID")),
    ("the other box or the timing lock was busy", re.compile(r"HELD:|is measuring|timing lock")),
    ("a test failed", re.compile(r"--- FAIL|^FAIL\b", re.M)),
    ("harness or script error", re.compile(r"Traceback|No such file|command not found|error:", re.I)),
]


def minutes(s):
    m = RAN.search(s)
    return (int(m.group(1) or 0) * 60 + int(m.group(2))) if m else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="~/goinfer-logs/night/runs")
    ap.add_argument("--since", default="2026-10-06")
    a = ap.parse_args()
    root = os.path.expanduser(a.runs)
    days = sorted(d for d in os.listdir(root) if d >= a.since and os.path.isdir(os.path.join(root, d)))
    rows, ok, longest = 0, 0, []
    causes = collections.Counter()
    results = collections.Counter()
    for d in days:
        p = os.path.join(root, d, "SUMMARY.md")
        if not os.path.exists(p):
            continue
        for line in open(p, errors="replace"):
            m = ROW.match(line)
            if not m:
                continue
            rows += 1
            job, result, ran, log = m.group(2).strip(), m.group(3).strip(), m.group(4).strip(), m.group(6)
            ok += result == "ok"
            results[result.split(" (")[0].split(" rc=")[0]] += 1
            longest.append((minutes(ran), d, job, result))
            if result == "ok":
                continue
            text = open(log, errors="replace").read() if os.path.exists(log) else ""
            if result.startswith("timeout"):
                causes["timed out"] += 1
            elif result.startswith(("retry", "stopped")):
                causes["re-queued (its own check refused, or stopped)"] += 1
            else:
                causes[next((k for k, rx in CAUSES if rx.search(text)), "other")] += 1
    logs = sorted(glob.glob(os.path.join(root, "*", "*.log")))
    logs = [p for p in logs if os.path.basename(os.path.dirname(p)) >= a.since]
    hits = collections.Counter()
    any_hit = 0
    for p in logs:
        text = open(p, errors="replace").read()
        found = [k for k, rx in PATTERNS.items() if rx.search(text)]
        for k in found:
            hits[k] += 1
        any_hit += bool(found)
    print(f"# night queue on this machine since {a.since}: {len(days)} nights")
    print(f"job rows (retries included): {rows}; ok: {ok}; results: {dict(results)}")
    print("why the rows that were not ok were not (first matching cause in the job's log):")
    for k, n in causes.most_common():
        print(f"  {n:4d}  {k}")
    print(f"job logs: {len(logs)}; with a refusal or void: {any_hit}")
    for k, n in hits.most_common():
        print(f"  {n:4d} logs  {k}")
    print("\nlongest jobs:")
    for mins, d, job, result in sorted(longest, reverse=True)[:12]:
        print(f"  {mins // 60} h {mins % 60:02d} min  {d}  {job}  ({result})")


if __name__ == "__main__":
    main()
