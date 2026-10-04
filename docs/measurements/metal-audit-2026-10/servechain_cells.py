#!/usr/bin/env python3
"""Tabulate the serve-chain A/B (docs/tasks/task-metal-audit-2026-10.md, "The serve chain: served grade").

usage: servechain_cells.py <log dir>    (the run's LOG: pass1/ .. pass4/, each with peer.json and serve-logs/)

Per pass and cell: new = goinfer (the hold on), old = goinfer_old (the same rev, never hold), each bench_peer's mean
over its runs; the pass's ratio is new / old. Per cell: the median of the passes' ratios and how many are above 1.
The precondition is read from each cell's serve log (the shutdown lines, internal/serveapp/main.go): the share of the
cell's decode tokens that ran held, held / (held + solo + batched-step tokens), at least 0.95 on the new arm and 0 on
the old, and the decode path metal-resident (int4) on both.
"""
import json
import os
import re
import statistics
import sys

RB = re.compile(r'resident batch "bench": (\d+) runs, (\d+) batched steps \((\d+) tokens\), (\d+) solo tokens')
HELD = re.compile(r'resident batch "bench" held: (\d+) tokens in (\d+) holds')


def serve_share(path):
    """(held share, decode path ok) for one cell's serve log; None if the log lacks the shutdown lines."""
    try:
        txt = open(path, errors="replace").read()
    except OSError:
        return None
    rb, held = RB.findall(txt), HELD.findall(txt)
    if not rb or not held:
        return None
    step_tok, solo = int(rb[-1][2]), int(rb[-1][3])
    h = int(held[-1][0])
    total = h + solo + step_tok
    return (h / total if total else 0.0), ("decode path: metal-resident (int4)" in txt)


def main(log):
    passes = sorted(d for d in os.listdir(log) if d.startswith("pass"))
    cells, pre_fail = {}, []
    for p in passes:
        recs = json.load(open(os.path.join(log, p, "peer.json")))
        by = {}
        for r in recs:
            if r.get("kind") == "provenance" or r.get("error"):
                if r.get("error") and r.get("engine") in ("goinfer", "goinfer_old"):
                    pre_fail.append(f"{p} {r['engine']} {r['model']} {r['config']}: error {r['error']}")
                continue
            by[(r["model"], r["depth"], r["config"], r["engine"])] = r
        for (mk, depth, cfg, eng), r in by.items():
            if eng != "goinfer" or (mk, depth, cfg, "goinfer_old") not in by:
                continue
            old = by[(mk, depth, cfg, "goinfer_old")]
            cells.setdefault((mk, depth, cfg), []).append((p, r["mean"], old["mean"], r["mean"] / old["mean"]))
            for e, want_new in (("goinfer", True), ("goinfer_old", False)):
                sl = os.path.join(log, p, "serve-logs", f"{e}-metal-{mk}-{depth}-{cfg}.log")
                s = serve_share(sl)
                if s is None:
                    pre_fail.append(f"{p} {e} {mk} {cfg}: no shutdown counters in {sl}")
                    continue
                share, path_ok = s
                if not path_ok or (want_new and share < 0.95) or (not want_new and share != 0):
                    pre_fail.append(f"{p} {e} {mk} {cfg}: held share {share:.3f}, decode path ok {path_ok}")
    print(f"{'model':5} {'depth':>5} {'config':16} {'median new/old':>14}  above1  per pass (new / old tok/s = ratio)")
    for (mk, depth, cfg), rows in sorted(cells.items(), key=lambda kv: (kv[0][2], kv[0][0])):
        ratios = [x[3] for x in rows]
        per = "  ".join(f"{p}: {n:.1f}/{o:.1f}={q:.3f}" for p, n, o, q in rows)
        print(f"{mk:5} {depth:5} {cfg:16} {statistics.median(ratios):14.3f}  {sum(q > 1 for q in ratios)}/{len(ratios)}"
              f"     {per}")
    print("precondition:", "HOLDS" if not pre_fail else "FAILS")
    for f in pre_fail:
        print("  " + f)
    return 1 if pre_fail else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
