#!/usr/bin/env python3
"""census.py — where goinfer's verification wall-clock goes, from the records already on disk (TE0).

Read-only. It runs no test and times nothing: it reads what earlier runs left behind and prints five tables.

  1. served harness  — every bench_peer*-shaped JSON (first element {"kind": "provenance"}) under docs/measurements:
                       cells, cell wall ("secs" per record: server start + load + warm-up + timed runs + teardown),
                       and the part of it that was decode actually being timed (counts.tokens / mean(runs)).
  2. spans           — every "=== <timestamp> ... START" / "=== <timestamp> ... END" pair in a run log, and the
                       idle-gate lines inside it ("cell gate: loadavg", "waiting for idle"; each one is a 20 s sleep).
  3. go test         — "ok/FAIL <pkg> <secs>" package lines and "--- PASS/FAIL: TestX (<secs>)" top-level tests.
  4. parity churn    — September commits touching a parity shared set, and Deps-Hash-Refresh trailers (git, read-only).
  5. waste signals   — measurement records carrying a void / withdrawn / discarded / contaminated verdict, and records
                       that say a gate could not resolve its own bar.

usage: python3 docs/measurements/test-efficiency-2026-09/census.py [--since YYYY-MM-DD] [extra log dirs ...]
       (default log dirs: docs/measurements, ~/goinfer-logs, ~/gate-logs — whichever exist on this machine)

LIMITS, stated so nobody reads more into the tables than is there:
  - It sees only what was ARCHIVED. A daytime `go test` whose output went to a terminal is invisible here; so is
    nobara's ~/goinfer-bench unless its logs were committed. The vscode-claude transcripts on each machine
    (~/.claude/projects/) hold every command and are the complete source — a session on that machine can read them.
  - Most logs do not record their host; the served JSONs do (provenance.host), spans and go test lines mostly do not.
  - A 20 s sleep per idle-gate line is the harness's own interval (bench_peer.py gate_cell_idle, the run-*.sh loops).
"""
import argparse
import ast
import collections
import datetime as dt
import glob
import gzip
import json
import os
import re
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve()
REPO = HERE.parents[3]
MEAS = REPO / "docs" / "measurements"
TS = r"(\d{4}-\d\d-\d\d[ T]\d\d:\d\d:\d\d)"
SHARED_SETS = ("core", "loaders", "quant")


def read(p):
    try:
        if str(p).endswith(".gz"):
            with gzip.open(p, "rt", errors="replace") as f:
                return f.read()
        with open(p, errors="replace") as f:
            return f.read()
    except Exception:
        return ""


def served(since):
    outs = []
    for p in sorted(glob.glob(str(MEAS / "**" / "*.json"), recursive=True)):
        try:
            d = json.loads(read(p))
        except Exception:
            continue
        if not (isinstance(d, list) and d and isinstance(d[0], dict) and d[0].get("kind") == "provenance"):
            continue
        recs = [r for r in d[1:] if isinstance(r, dict) and r.get("secs")]
        utc = d[0].get("utc") or ""
        if not recs or (since and utc and utc < since):
            continue
        outs.append((utc, d[0].get("host") or "?", p, recs))
    print(f"## 1. served harness outputs{' since ' + since if since else ''}\n")
    if not outs:
        print("none found\n")
        return
    cells = sum(len(o[3]) for o in outs)
    wall = sum(r["secs"] for o in outs for r in o[3])
    dated = sorted(o[0] for o in outs if o[0])
    print(f"{len(outs)} outputs, {cells} cells, {wall / 3600:.1f} h of cell wall"
          f" (dated outputs span {dated[0][:10]} .. {dated[-1][:10]}; {len(outs) - len(dated)} undated)")
    host = collections.defaultdict(lambda: [0, 0, 0.0])
    for utc, h, p, recs in outs:
        k = "mac" if "macbook" in h.lower() else ("nobara" if "nobara" in h.lower() else h)
        host[k][0] += 1
        host[k][1] += len(recs)
        host[k][2] += sum(r["secs"] for r in recs)
    for k, v in sorted(host.items()):
        print(f"  {k:<8} {v[0]:>3} outputs {v[1]:>5} cells {v[2] / 3600:6.1f} h")
    by = collections.defaultdict(lambda: [0, 0.0, 0.0])
    tw = tm = 0.0
    for _, _, _, recs in outs:
        for r in recs:
            c = r.get("counts")
            try:
                c = ast.literal_eval(c) if isinstance(c, str) else c
            except Exception:
                c = None
            rates = [x for x in (r.get("runs") or []) if x]
            if not (c and c.get("tokens") and rates):
                continue
            meas = c["tokens"] / (sum(rates) / len(rates))
            k = (str(r.get("model")), str(r.get("engine")))
            by[k][0] += 1
            by[k][1] += r["secs"]
            by[k][2] += meas
            tw += r["secs"]
            tm += meas
    print(f"\ncells with token counts: {sum(v[0] for v in by.values())}; cell wall {tw / 3600:.2f} h, of which decode being "
          f"timed {tm / 3600:.2f} h ({100 * tm / tw:.0f}%) — the rest is server start, load, warm-up, prompt prefill, teardown")
    print(f"\n  {'model':<10} {'engine':<12} {'cells':>5} {'wall min':>9} {'timed':>6} {'overhead/cell':>14}")
    for k, v in sorted(by.items(), key=lambda kv: -kv[1][1])[:16]:
        print(f"  {k[0]:<10} {k[1]:<12} {v[0]:>5} {v[1] / 60:>9.1f} {100 * v[2] / v[1]:>5.0f}% {(v[1] - v[2]) / v[0]:>12.1f} s")
    print()


def spans_and_tests(roots, since):
    spans, pkgs, tests = [], [], []
    for root in roots:
        for p in glob.glob(os.path.join(root, "**", "*"), recursive=True):
            if not re.search(r"\.(log|txt)(\.gz)?$", p):
                continue
            t = read(p)
            if not t:
                continue
            rel = os.path.relpath(p, os.path.dirname(root))
            st = [(m.start(), m.group(1)) for m in re.finditer(r"^=== " + TS + r"[^\n]*?\bSTART\b", t, re.M)]
            en = [(m.start(), m.group(1)) for m in re.finditer(r"^=== " + TS + r"[^\n]*?\bEND\b", t, re.M)]
            for i, (pos, ts) in enumerate(st):
                nxt = [e for e in en if e[0] > pos]
                nextst = st[i + 1][0] if i + 1 < len(st) else float("inf")
                if not nxt or nxt[0][0] > nextst:
                    continue
                a = dt.datetime.fromisoformat(ts.replace(" ", "T"))
                b = dt.datetime.fromisoformat(nxt[0][1].replace(" ", "T"))
                secs = (b - a).total_seconds()
                if not (0 < secs < 86400) or (since and ts[:10] < since):
                    continue
                waits = len(re.findall(r"cell gate: loadavg|waiting for idle", t[pos:nxt[0][0]]))
                line = t[pos:t.find("\n", pos)]
                spans.append((ts[:16], secs, waits, rel, line))
            for m in re.finditer(r"^(ok|FAIL)\s+(\S+)\s+([\d.]+)s", t, re.M):
                pkgs.append((m.group(2).replace("github.com/townsendmerino/", ""), float(m.group(3)), rel))
            for m in re.finditer(r"^--- (PASS|FAIL): (Test\w+) \(([\d.]+)s\)", t, re.M):
                tests.append((m.group(2), float(m.group(3)), rel))
    print("## 2. START/END spans in run logs\n")
    tot = sum(s[1] for s in spans)
    gated = [s for s in spans if s[2]]
    print(f"{len(spans)} spans, {tot / 3600:.1f} h; {len(gated)} contain idle-gate waits: "
          f"{sum(s[1] for s in gated) / 60:.0f} min of wall, {sum(s[2] for s in gated) * 20 / 60:.0f} min of it waiting "
          f"({100 * sum(s[2] for s in gated) * 20 / max(1, sum(s[1] for s in gated)):.0f}%)")
    print("\nlongest 15:")
    for s in sorted(spans, key=lambda s: -s[1])[:15]:
        print(f"  {s[0]}  {s[1] / 60:5.1f} min  waits {s[2] * 20 / 60:4.1f} min  {s[3][:64]}")
    print("\n## 3. go test in archived logs\n")
    print(f"{len(pkgs)} package results, {sum(p[1] for p in pkgs) / 3600:.1f} h")
    agg = collections.defaultdict(list)
    for name, secs, _ in pkgs:
        agg[name].append(secs)
    for k, v in sorted(agg.items(), key=lambda kv: -sum(kv[1]))[:8]:
        v = sorted(v)
        print(f"  {k:<34} n={len(v):>3} median {v[len(v) // 2]:7.1f} s  max {v[-1]:8.1f} s  total {sum(v) / 3600:5.1f} h")
    agg2 = collections.defaultdict(list)
    for name, secs, _ in tests:
        agg2[name].append(secs)
    print(f"\n{len(tests)} top-level test results; the 15 longest single runs:")
    for k, v in sorted(agg2.items(), key=lambda kv: -max(kv[1]))[:15]:
        v = sorted(v)
        print(f"  {k:<46} n={len(v):>3} max {v[-1] / 60:6.1f} min  median {v[len(v) // 2] / 60:6.1f} min")
    print()


def git(*args):
    env = dict(os.environ, GIT_OPTIONAL_LOCKS="0")
    r = subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True, env=env)
    return r.stdout if r.returncode == 0 else ""


def parity_churn(since):
    print("## 4. parity churn\n")
    try:
        m = json.loads((REPO / "testdata" / "parity_manifest.json").read_text())
    except Exception as e:
        print(f"no manifest: {e}\n")
        return
    fam = m.get("families", {})
    sets = m.get("shared_sets", {})
    files = sorted({f for s in SHARED_SETS for f in (sets.get(s) or [])})
    uses = collections.Counter(u for v in fam.values() for u in (v.get("uses") or []))
    print(f"{len(fam)} families; {uses.get('core', 0)} use `core` ({len(sets.get('core') or [])} files)")
    s = since or "2026-09-01"
    commits = [c for c in git("log", f"--since={s}", "--format=%h", "--", *files).split() if c]
    trailers = git("log", f"--since={s}", "--format=%B").count("Deps-Hash-Refresh:")
    print(f"since {s}: {len(commits)} commits touched a core/loaders/quant file; {trailers} Deps-Hash-Refresh trailers\n")


def waste(since):
    print("## 5. waste signals in measurement records\n")
    pat_void = re.compile(r"\b(void|voided|withdrawn|discarded|contaminated)\b", re.I)
    pat_res = re.compile(r"resolving power|could not separate|cannot resolve", re.I)
    tag = (since or "2026-09")[:7]
    recs = [p for p in glob.glob(str(MEAS / "**" / "*.md"), recursive=True) if tag in p]
    nv = sum(1 for p in recs if pat_void.search(read(p)))
    nr = sum(1 for p in recs if pat_res.search(read(p)))
    print(f"{len(recs)} records with {tag} in their path: {nv} carry a void/withdrawn/discarded/contaminated verdict "
          f"somewhere; {nr} say a gate could not resolve its bar\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default="", help="YYYY-MM-DD (default: everything on disk)")
    ap.add_argument("dirs", nargs="*")
    a = ap.parse_args()
    roots = a.dirs or [str(MEAS)] + [d for d in (os.path.expanduser("~/goinfer-logs"), os.path.expanduser("~/gate-logs"))
                                     if os.path.isdir(d)]
    print(f"# verification census — {dt.datetime.now().astimezone():%Y-%m-%d %H:%M %Z}, repo {git('rev-parse', '--short', 'HEAD').strip()}"
          f", roots: {', '.join(roots)}\n")
    served(a.since)
    spans_and_tests(roots, a.since)
    parity_churn(a.since)
    waste(a.since)


if __name__ == "__main__":
    main()
