#!/usr/bin/env python3
"""KV8a-Q gate 1: where each request's prompt ids part from the previous request's, and which held point could serve it.

Input: TestKV8aQ_gate1Scopes output (one line per request, in order: ids, pieces, keep, boundary). For each request it prints the
first id at which it differs from the previous request, the text either side, and the candidates a restore could start from:

  live / latest  the previous request's whole prompt (the slot's live state, or KV8a's end-of-prompt checkpoint)
  keep           an earlier request's KVScope.Keep (B35 keeps a recurrent snapshot there, rounded down to its pass grid)
  boundary       the system-and-tools boundary (KV7's prefix snapshot)

A candidate at position p serves a request only if p <= the first difference d: a recurrent state cannot roll back.

Usage: diverge.py scopes.jsonl
"""
import json, os, sys

rows = [json.loads(l) for l in open(sys.argv[1])]
name = lambda r: os.path.basename(r["file"])[:-5]


def first_diff(x, y):
    return next((i for i in range(min(len(x), len(y))) if x[i] != y[i]), min(len(x), len(y)))


def text(r, a, b):
    return json.dumps("".join(r["pieces"][max(0, a):b]))


for r in rows:
    print(f"{name(r)}: {r['n']} ids, keep {r['keep']} (tail past it {text(r, r['keep'], r['n'])}), boundary {r['boundary']}")
print()
for i in range(1, len(rows)):
    prev, cur = rows[i - 1], rows[i]
    d = first_diff(prev["ids"], cur["ids"])
    print(f"{name(cur)} vs {name(prev)}: first difference at id {d} of {prev['n']}"
          + (" (the whole previous prompt is a prefix)" if d == prev["n"] else f" ({prev['n'] - d} of {name(prev)}'s ids not shared)"))
    print(f"   shared up to: {text(prev, d - 14, d)}")
    print(f"   {name(prev)} goes on: {text(prev, d, d + 8)}")
    print(f"   {name(cur)} goes on: {text(cur, d, d + 8)}")
    cands = [("live/latest", name(prev), prev["n"])]
    cands += [("keep", name(r), r["keep"]) for r in rows[:i] if r["keep"] > 0]
    cands.append(("boundary", name(cur), cur["boundary"]))
    ok = [(p, kind, src) for kind, src, p in cands if 0 < p <= d and cur["ids"][:p] == rows[[name(r) for r in rows].index(src)]["ids"][:p]]
    best = max(ok) if ok else (0, "cold", "-")
    print("   candidates: " + ", ".join(f"{kind} of {src} at {p}{'' if 0 < p <= d else ' (past d)'}" for kind, src, p in cands))
    print(f"   deepest that serves: {best[1]} of {best[2]} at {best[0]}; re-prefill {cur['n'] - best[0]} of {cur['n']} ids")
