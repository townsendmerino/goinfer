#!/usr/bin/env python3
"""breakdown.py — which commands the session time went to, from the Claude Code transcripts on this machine.

Read-only. It runs nothing and times nothing. It reuses test-efficiency-2026-09/transcript_miner.py's parser (`mine`):
a call's duration is its result's timestamp minus its call's; a background call's is its task notification's; a
foreground call longer than the tool's cap is BLOCKED and left out. Call durations OVERLAP (background polls run beside
other work), so these sums are not wall clock; test-wait-2026-10/session_wall.py is the wall-clock instrument.

  usage: breakdown.py [--since YYYY-MM-DD] [--projects GLOB ...]
         defaults: ~/.claude/projects/*goinfer* and *aikit* (subagent transcripts included), since 2026-10-12

Sections: `go test` calls by flag and target; wait/poll calls by what they waited on; build/lint by tool; served-gate
scripts; touched.py's duration distribution; watching PR checks. Patterns are regexes over the command text, and a
compound command can match more than one section, so the sections do not add up to a total.
"""
import argparse
import collections
import datetime as dt
import glob
import os
import re
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "test-efficiency-2026-09"))
import transcript_miner as tm  # noqa: E402


def hours(s):
    return f"{s / 3600:6.2f} h"


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default="2026-10-12")
    ap.add_argument("--projects", nargs="*", default=["~/.claude/projects/*goinfer*", "~/.claude/projects/*aikit*"])
    a = ap.parse_args()
    since = dt.datetime.fromisoformat(a.since + "T00:00:00").astimezone()
    files = []
    for pat in a.projects:
        for d in glob.glob(os.path.expanduser(pat)):
            files += glob.glob(d + "/*.jsonl") + glob.glob(d + "/*/subagents/*.jsonl")
    recs = [r for r in tm.mine(files, since) if r.get("secs")]
    print(f"# command breakdown since {a.since}: {len(files)} transcript files, {len(recs)} calls with a duration\n")

    # go test
    gt = [r for r in recs if re.search(r"\bgo test\b", r["cmd"])]
    print(f"## go test: {len(gt)} calls, {hours(sum(r['secs'] for r in gt))}")
    for name, rx in [("-count=1", r"-count[= ]1\b"), ("-run", r"\s-run[= ]"), ("-tags", r"-tags"),
                     ("-bench / -benchtime", r"-bench"), ("GOINFER_HEAVY_TESTS=1", r"GOINFER_HEAVY_TESTS=1"),
                     ("-race", r"-race")]:
        rs = [r for r in gt if re.search(rx, r["cmd"])]
        print(f"  with {name:22s} {len(rs):5d} calls {hours(sum(r['secs'] for r in rs))}")
    tgt = collections.defaultdict(lambda: [0, 0.0])
    for r in gt:
        pk = []
        for m in re.finditer(r"go test([^|;&\n]*)", r["cmd"]):
            pk += re.findall(r"(?<!\S)(\./[\w./-]*|\.\.\.)", m.group(1)) or ["(cwd)"]
        cd = re.findall(r"cd ([\w~./${}-]+)", r["cmd"])
        mod = next((m for m in ("metal", "cuda", "gpu") if cd and (cd[-1].endswith("/" + m) or ("/" + m + "/") in cd[-1])), "")
        if cd and "aikit" in cd[-1]:
            mod = "aikit"
        k = (mod + ":" if mod else "") + ",".join(sorted(set(pk)))
        tgt[k][0] += 1
        tgt[k][1] += r["secs"]
    print("  top targets by time:")
    for k, (n, s) in sorted(tgt.items(), key=lambda kv: -kv[1][1])[:15]:
        print(f"    {hours(s)} {n:5d} calls  {k[:100]}")

    # wait/poll
    wp = [r for r in recs if r["class"] == "wait/poll"]
    cats = collections.defaultdict(lambda: [0, 0.0])
    for r in wp:
        c = r["cmd"]
        if re.search(r"gh (pr checks|run (watch|list|view))", c):
            k = "CI (gh pr checks / gh run)"
        elif re.search(r"nobara", c):
            k = "nobara: remote runs and its night queue"
        elif re.search(r"date \+%H%M|-lt 0700|-lt 07", c):
            k = "a clock time"
        elif re.search(r"kill -0|pgrep|ps -p", c):
            k = "a local process"
        elif re.search(r"loadavg|load <|idle", c):
            k = "an idle or load gate"
        else:
            k = "other"
        cats[k][0] += 1
        cats[k][1] += r["secs"]
    print(f"\n## wait/poll: {len(wp)} calls, {hours(sum(r['secs'] for r in wp))} (call time; mostly background)")
    for k, (n, s) in sorted(cats.items(), key=lambda kv: -kv[1][1]):
        print(f"  {hours(s)} {n:5d}  {k}")

    # build/lint
    bl = [r for r in recs if r["class"] == "build/lint"]
    bc = collections.defaultdict(lambda: [0, 0.0])
    for r in bl:
        c = r["cmd"]
        k = next((t for t in ("staticcheck", "go vet", "go build", "gofmt", "go install") if t in c), "other (compound)")
        bc[k][0] += 1
        bc[k][1] += r["secs"]
    print(f"\n## build/lint: {len(bl)} calls, {hours(sum(r['secs'] for r in bl))}")
    for k, (n, s) in sorted(bc.items(), key=lambda kv: -kv[1][1]):
        print(f"  {hours(s)} {n:5d}  {k}")

    # served gates
    sg = [r for r in recs if r["class"] == "served gate"]
    sc = collections.defaultdict(lambda: [0, 0.0])
    for r in sg:
        m = re.search(r"(bench_peer\w*\.py|run-[\w.-]+\.sh)", r["cmd"])
        k = m.group(1) if m else "?"
        sc[k][0] += 1
        sc[k][1] += r["secs"]
    print(f"\n## served gate: {len(sg)} calls, {hours(sum(r['secs'] for r in sg))}")
    for k, (n, s) in sorted(sc.items(), key=lambda kv: -kv[1][1])[:10]:
        print(f"  {hours(s)} {n:5d}  {k}")

    # touched.py and CI watching
    for name, rx in [("scripts/touched.py", r"touched\.py"), ("gh pr checks --watch", r"gh pr checks[^|;&\n]*--watch"),
                     ("refresh_parity_hashes.sh", r"refresh_parity_hashes"), ("staticcheck", r"staticcheck")]:
        rs = [r["secs"] for r in recs if re.search(rx, r["cmd"])]
        if not rs:
            continue
        over5 = sum(1 for s in rs if s >= 300)
        print(f"\n## {name}: {len(rs)} calls, {hours(sum(rs))}; median {statistics.median(rs) / 60:.1f} min, "
              f"p90 {pct(rs, .9) / 60:.1f} min, p99 {pct(rs, .99) / 60:.1f} min, max {max(rs) / 60:.1f} min; "
              f"{over5} at 5 min or more")


if __name__ == "__main__":
    main()
