#!/usr/bin/env python3
"""Grade KV4's G2 MC3 cells (kv4-cuda-2026-10-21.md §2) from run-mc3.sh's w7.json: the new/old aggregate decode tok/s per pair of cells
(old4_1/new4_1, new4_2/old4_2, old4_3/new4_3), the median over the three pairs (PASS >= 0.98), and every turn's content_sha equal between
the arms.

  score_mc3.py <w7.json>
"""
import json, statistics, sys

res = json.load(open(sys.argv[1]))["results"]
ratios, shas_ok = [], True
for o, n in (("old4_1", "new4_1"), ("old4_2", "new4_2"), ("old4_3", "new4_3")):
    co, cn = res[o]["4"], res[n]["4"]
    ao, an = co["aggregate_tok_s"], cn["aggregate_tok_s"]
    ratios.append(an / ao)
    so = [t["content_sha"] for cl in co["per_client"] for t in cl]
    sn = [t["content_sha"] for cl in cn["per_client"] for t in cl]
    shas_ok &= so == sn and len(so) > 0
    print(f"{o} {ao:.1f} -> {n} {an:.1f} tok/s: {an / ao:.3f}; {len(so)} turns, replies equal: {so == sn}")
med = statistics.median(ratios)
print(f"median {med:.3f}: {'PASS' if med >= 0.98 and shas_ok else 'FAIL'}; content_sha equal in every turn: {shas_ok}")
