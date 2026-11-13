#!/usr/bin/env python3
"""Grade run-p4a.sh's w7.json by registration.md beside this file. Per (kv, clients) cell, each pair's B/A ratio of the aggregate decode
tok/s and of the median per-client tok/s (a client's completion tokens over its summed turn latency), every turn's content_sha equal between
the arms, and how many pairs B is at least as fast as A. Decisive cells (4 clients): PASS when every pair's replies are equal, the median
aggregate ratio is at least 1.00 and B is at least as fast on at least 9 of 10 pairs; else FAIL. The 2-client cells are reported.

  grade-p4a.py <w7.json>
"""
import json, statistics, sys

res = json.load(open(sys.argv[1]))["results"]


def cell(key):
    c = next(iter(res[key].values()))
    per = [sum(t["completion_tokens"] or 0 for t in cl) / (sum(t["latency_ms"] for t in cl) / 1e3) for cl in c["per_client"]]
    return c["aggregate_tok_s"], statistics.median(per), [t["content_sha"] for cl in c["per_client"] for t in cl]


verdicts = []
for kv in ("f16", "i8"):
    for n, pairs in ((4, 10), (2, 3)):
        agg, pc, faster, same = [], [], 0, True
        for p in range(1, pairs + 1):
            ka, kb = f"A_{kv}_{n}_{p}", f"B_{kv}_{n}_{p}"
            if ka not in res or kb not in res:
                print(f"kv {kv} clients {n} pair {p}: MISSING ({ka in res}, {kb in res})")
                same = False
                continue
            (aa, pa, sa), (ab, pb, sb) = cell(ka), cell(kb)
            agg.append(ab / aa)
            pc.append(pb / pa)
            faster += ab >= aa
            same &= sa == sb and len(sa) > 0
        if not agg:
            continue
        line = (f"kv {kv} clients {n}: aggregate B/A {' '.join(f'{r:.3f}' for r in agg)} median {statistics.median(agg):.3f}; "
                f"per-client median {statistics.median(pc):.3f}; B >= A on {faster}/{len(agg)}; replies equal {same}")
        if n == 4:
            ok = same and len(agg) == 10 and statistics.median(agg) >= 1.0 and faster >= 9
            verdicts.append(ok)
            line += f" -> {'PASS' if ok else 'FAIL'}"
        print(line)
print("VERDICT:", "PASS" if verdicts and all(verdicts) else "FAIL")
