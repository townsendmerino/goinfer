#!/usr/bin/env python3
"""Grade G-M2-speed (docs/tasks/task-minimax-code-compat-2026-09.md, "M2") from bench_peer.py's results file.

  grade_m2_speed.py <g-m2-speed.json> <serve log dir>

Per model: each ABBA block holds one restarted cell an arm; a block's rate is the mean of its runs; the statistic is
the median over the blocks of new / old. SHIP when it is >= 0.97 on both models, PARKED in [0.95, 0.97) on either,
FAIL under 0.95 on either. VOID when an arm's serve log does not show the context the comparison is about (the new
build above 4096, the old build at 4096), a cell errored, or a block is missing an arm. Exit 0 only on SHIP.
"""
import json, os, re, statistics, sys

def contexts(path):
    try:
        return sorted({int(m.group(1)) for m in re.finditer(r"context: (\d+) tokens", open(path, errors="replace").read())})
    except OSError:
        return []

def main():
    cells = [r for r in json.load(open(sys.argv[1])) if r.get("phase") == "A"]
    logs = sys.argv[2]
    verdicts, void = [], []
    for mk in ("1.5B", "7B"):
        rate = {}
        for r in cells:
            if r["model"] != mk or r.get("backend") != "metal":
                continue
            if r.get("error") or not r.get("runs"):
                void.append(f"{mk}: {r['engine']} block {r.get('block')} errored: {r.get('error')}")
                continue
            rate[(r["engine"], r.get("block"))] = statistics.mean(r["runs"])
        blocks = sorted({b for _, b in rate})
        ratios = []
        for b in blocks:
            if ("goinfer", b) not in rate or ("goinfer_old", b) not in rate:
                void.append(f"{mk}: block {b} is missing an arm")
                continue
            ratios.append(rate[("goinfer", b)] / rate[("goinfer_old", b)])
        new_ctx = contexts(os.path.join(logs, f"goinfer-metal-{mk}-128-greedy.log"))
        old_ctx = contexts(os.path.join(logs, f"goinfer_old-metal-{mk}-128-greedy.log"))
        print(f"{mk}: new build's context {new_ctx}, old build's {old_ctx}")
        if not new_ctx or min(new_ctx) <= 4096:
            void.append(f"{mk}: the new build did not resolve a context above 4096 in every cell ({new_ctx}); nothing was compared")
        if old_ctx != [4096]:
            void.append(f"{mk}: the old build's context is {old_ctx}, not 4096")
        if not ratios:
            void.append(f"{mk}: no paired block")
            continue
        med = statistics.median(ratios)
        news = [rate[("goinfer", b)] for b in blocks if ("goinfer", b) in rate]
        olds = [rate[("goinfer_old", b)] for b in blocks if ("goinfer_old", b) in rate]
        print(f"{mk}: new {statistics.median(news):.2f} tok/s, old {statistics.median(olds):.2f} tok/s (medians over {len(blocks)} blocks); "
              f"new/old by block {' '.join(f'{x:.4f}' for x in ratios)}; median {med:.4f}, min {min(ratios):.4f}, max {max(ratios):.4f}")
        verdicts.append("SHIP" if med >= 0.97 else "PARKED" if med >= 0.95 else "FAIL")
    for v in void:
        print("VOID: " + v)
    if void:
        print("VERDICT VOID")
        return 2
    final = "FAIL" if "FAIL" in verdicts else "PARKED" if "PARKED" in verdicts else "SHIP"
    print(f"VERDICT {final} (per model: {', '.join(verdicts)}; bar: median new/old >= 0.97 on both)")
    return 0 if final == "SHIP" else 1

if __name__ == "__main__":
    sys.exit(main())
