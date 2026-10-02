#!/usr/bin/env python3
"""The 100 rows B9's split runs on, chosen from D6a's 400 OOD noul rows by a rule fixed before any run (2026-10-01):
per environment (the id's first field), an allocation proportional to that environment's share of the 400 (largest
remainder), and within each environment the first k ids by sha256(id). Reads only the committed reference, whose rows
are the 400 in D6a's order, and writes noul-100-ids.txt beside this script.

    python3 select100.py
"""
import collections, hashlib, json, os

HERE = os.path.dirname(os.path.abspath(__file__))
REF = os.path.join(HERE, "..", "decisions-d6a-2026-09-28", "results", "b0-ood-noul-f32.jsonl")
N = 100
ids = [json.loads(l)["id"] for l in open(REF) if l.strip()]
assert len(ids) == 400, len(ids)
env = collections.OrderedDict()
for i in ids:
    env.setdefault(i.split(":")[0], []).append(i)
quota = {e: N * len(v) / len(ids) for e, v in env.items()}
alloc = {e: int(q) for e, q in quota.items()}
for e in sorted(env, key=lambda e: (-(quota[e] - alloc[e]), e))[: N - sum(alloc.values())]:
    alloc[e] += 1
chosen = set()
for e, v in env.items():
    chosen |= set(sorted(v, key=lambda i: hashlib.sha256(i.encode()).hexdigest())[: alloc[e]])
out = [i for i in ids if i in chosen]  # D6a's order
assert len(out) == N
open(os.path.join(HERE, "noul-100-ids.txt"), "w").write("\n".join(out) + "\n")
print("allocation:", {e: (len(env[e]), alloc[e]) for e in env})
