#!/usr/bin/env python3
"""Grades the MC4 candidate ("speculate when alone, batch under load", -spec-adaptive) on CUDA.

Pre-registered in docs/measurements/mc4-candidate-cuda-2026-10-01.md before the run; run.sh produced raw/.
Keys are <arm><clients>_<round>: arm b = batch (serve's defaults), s = spec-exclusive (-spec ngram), c = candidate
(-spec ngram -spec-adaptive); the staggered cell's "clients" is S. Every ratio is PAIRED (same round, same arm
order block), then the median of the 3 rounds. p99 is nearest rank over the cell's own requests/turns.

  python3 gates.py [raw-dir]
"""
import json, math, os, statistics, sys

RAW = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "raw")


def load(name):
    p = os.path.join(RAW, name)
    return json.load(open(p))["results"] if os.path.exists(p) else {}


def p99(xs):
    xs = sorted(xs)
    return xs[max(0, math.ceil(0.99 * len(xs)) - 1)]


def cell_copy(c):   # a bench_spec_copy cell: agg tok/s, request latencies (ms), reply hashes in client/request order
    reqs = [x for cl in c["per_client"] for x in cl]
    return {"agg": c["aggregate_tok_s"], "lat": [x["latency_s"] * 1000 for x in reqs if "latency_s" in x],
            "sha": [x.get("content_sha") for x in reqs], "errs": c["http_errors"]}


def cell_chat(r, n):   # a bench_w7_plain cell at n clients
    c = r[str(n)]
    turns = [t for cl in c["per_client"] for t in cl]
    return {"agg": c["aggregate_tok_s"], "lat": [t["latency_ms"] for t in turns],
            "sha": [t.get("content_sha") for t in turns], "errs": []}


def pairs(cells, num, den, rounds=(1, 2, 3)):
    """cells: {(arm, round): cell}. -> list of (num/den) aggregate ratios, and p99 ratios, per round"""
    agg, tail = [], []
    for r in rounds:
        a, b = cells.get((num, r)), cells.get((den, r))
        if not a or not b:
            continue
        agg.append(a["agg"] / b["agg"])
        tail.append(p99(a["lat"]) / p99(b["lat"]))
    return agg, tail


def med(xs):
    return statistics.median(xs) if xs else float("nan")


def split(results, shape):
    """-> {clients_label: {(arm, round): cell}}"""
    out = {}
    for key, v in results.items():
        if len(key) < 3 or key[0] not in "bsc" or "_" not in key:
            continue
        arm, rest = key[0], key[1:]
        lab, rnd = rest.split("_")
        out.setdefault(lab, {})[(arm, int(rnd))] = shape(v, lab)
    return out


copy15 = split(load("copy-15b.json"), lambda v, lab: cell_copy(v))
chat15 = split(load("chat-15b.json"), lambda v, lab: cell_chat(v, int(lab)))
stag15 = split(load("stag-15b.json"), lambda v, lab: cell_copy(v))
copy7 = split(load("copy-7b.json"), lambda v, lab: cell_copy(v))

verdict = {}
fails = []


def identity(name, cells):
    """gate 1: every reply of c (and s) equals b's, every round; b's rounds also agree with each other"""
    ref = None
    bad = {"c": 0, "s": 0}
    total = 0
    for (arm, r), c in sorted(cells.items()):
        if arm == "b" and ref is None:
            ref = c["sha"]
    if ref is None:
        return None
    for (arm, r), c in cells.items():
        for h, g in zip(c["sha"], ref):
            if arm == "b" and h != g:
                bad.setdefault("b", 0); bad["b"] += 1
            elif arm in ("c", "s") and h != g:
                bad[arm] += 1
        if len(c["sha"]) != len(ref):
            bad[arm] = bad.get(arm, 0) + abs(len(c["sha"]) - len(ref))
        total += len(c["sha"])
    return bad, total


print("== gate 1: identity (every reply equals batch's; HARD for the candidate)")
for label, grp in (("copy 1.5B", copy15), ("chat 1.5B", chat15), ("staggered copy 1.5B", stag15), ("copy 7B (reported)", copy7)):
    for lab, cells in sorted(grp.items()):
        r = identity(label, cells)
        if r is None:
            print(f"  {label} {lab}: NO batch cell"); continue
        bad, total = r
        errs = sum(len(c["errs"]) for c in cells.values())
        print(f"  {label} clients={lab}: {total} replies compared; differing: candidate {bad['c']}, spec-excl {bad['s']}, batch-vs-batch {bad.get('b', 0)}; http errors {errs}")
        if "7B" not in label and (bad["c"] or bad.get("b", 0) or errs):
            fails.append(f"1 identity/errors {label} {lab}")

print("\n== candidate (c) / batch (b), paired, median of 3   [spec-excl (s) / batch shown for reference]")
rows = []


def row(label, cells, hard=None, bar=None, tail_bar=None):
    ca, ct = pairs(cells, "c", "b")
    sa, st = pairs(cells, "s", "b")
    cs, _ = pairs(cells, "c", "s")
    m = med(ca)
    line = (f"  {label}: c/b agg {m:.3f} (pairs {' '.join(f'{x:.3f}' for x in ca)}); p99 c/b {med(ct):.3f}"
            f"; s/b agg {med(sa):.3f}, p99 {med(st):.3f}; c/s agg {med(cs):.3f}")
    ok = None
    if bar is not None:
        ok = m >= bar
        line += f"   [bar >= {bar}: {'PASS' if ok else 'FAIL'}{'' if hard else ' (not hard)'}]"
    if tail_bar is not None:
        tok = med(ct) <= tail_bar
        line += f"   [p99 bar <= {tail_bar}: {'PASS' if tok else 'FAIL'}]"
        ok = tok if ok is None else (ok and tok)
    print(line)
    return ok


print("== gate 2: alone (1 client)")
a = row("copy 1.5B, 1 client", copy15.get("1", {}), hard=False, bar=1.25)
b = row("chat 1.5B, 1 client", chat15.get("1", {}), hard=True, bar=0.97)
verdict["2 copy"], verdict["2 chat"] = a, b
print("== gate 3: under load (4 clients; must not give up batching; HARD on both)")
a = row("copy 1.5B, 4 clients", copy15.get("4", {}), hard=True, bar=0.97)
b = row("chat 1.5B, 4 clients", chat15.get("4", {}), hard=True, bar=0.97)
verdict["3 copy"], verdict["3 chat"] = a, b
print("== gate 4: staggered (clients join every 2 s, rounds 12/9/6/3)")
verdict["4 stag"] = row("staggered copy 1.5B", stag15.get("S", {}), hard=False, bar=1.00, tail_bar=1.10)
print("== reported only: the 7B, copy")
row("copy 7B, 1 client", copy7.get("1", {}))
row("copy 7B, 4 clients", copy7.get("4", {}))

print("\n== reading (the rule is in the pre-registration; this only applies it)")
print("  identity:", "FAIL " + "; ".join(f for f in fails if f.startswith("1")) if any(f.startswith("1") for f in fails) else "PASS")
for k, v in verdict.items():
    print(f"  gate {k}: {'PASS' if v else 'FAIL' if v is not None else 'NO DATA'}")
