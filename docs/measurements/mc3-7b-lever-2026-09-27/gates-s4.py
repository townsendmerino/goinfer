#!/usr/bin/env python3
"""MC3 S4's W7 gates (docs/tasks/task-concurrency-2026-09.md, registered in 915bf6fc) from bench_w7_plain.py's JSON.
Usage: gates-s4.py W7-7B.json W7-1.5B.json. Cells <arm><clients>_<pair>; p99 / p50 nearest-rank over a cell's turns."""
import json, math, statistics, sys


def pct(xs, p):
    s = sorted(xs)
    return s[max(0, math.ceil(p / 100 * len(s)) - 1)]


def load(path):
    cells = {}
    for key, by_n in json.load(open(path))["results"].items():
        for cell in by_n.values():
            cells[key] = cell
    return cells


def identity(cells, name):
    ok = True
    for n in ("1", "2", "4"):
        group = {k: v for k, v in cells.items() if k[3:].split("_")[0] == n}
        if not group:
            continue
        ref_key = sorted(group)[0]
        ref = group[ref_key]["per_client"]
        for k, cell in sorted(group.items()):
            for ci, (a, b) in enumerate(zip(ref, cell["per_client"])):
                for ta, tb in zip(a, b):
                    if ta["content_sha"] != tb["content_sha"]:
                        print(f"GATE 1 FAIL {name}: {k} client {ci} turn {tb['n']}: content differs from {ref_key}")
                        ok = False
                    if (ta["prompt_tokens"] - ta["prefill_reused_tokens"]) != (tb["prompt_tokens"] - tb["prefill_reused_tokens"]):
                        print(f"GATE 1 FAIL {name}: {k} client {ci} turn {tb['n']}: reuse differs from {ref_key}")
                        ok = False
        print(f"{name} clients={n}: {len(group)} cells compared turn by turn (content and reuse)")
    return ok


def ratios(cells, n):
    agg, p50, p99 = [], [], []
    for i in (1, 2, 3):
        o, w = cells.get(f"old{n}_{i}"), cells.get(f"new{n}_{i}")
        if not (o and w):
            continue
        lo = [t["latency_ms"] for c in o["per_client"] for t in c]
        lw = [t["latency_ms"] for c in w["per_client"] for t in c]
        agg.append(w["aggregate_tok_s"] / o["aggregate_tok_s"])
        p50.append(pct(lw, 50) / pct(lo, 50))
        p99.append(pct(lw, 99) / pct(lo, 99))
        print(f"  pair {i}, {n} client(s): aggregate old {o['aggregate_tok_s']:.2f} new {w['aggregate_tok_s']:.2f} = {agg[-1]:.3f}x; "
              f"p50 {p50[-1]:.3f}x; p99 {pct(lo, 99)/1000:.2f} -> {pct(lw, 99)/1000:.2f} s = {p99[-1]:.3f}x")
    return statistics.median(agg), statistics.median(p50), statistics.median(p99)


def main(p7, p15):
    c7, c15 = load(p7), load(p15)
    g1 = identity(c7, "7B") & identity(c15, "1.5B")
    print(f"GATE 1 identity and reuse: {'PASS' if g1 else 'FAIL'}")
    a2, _, _ = ratios(c7, 2)
    print(f"GATE 2 7B 2-client aggregate median {a2:.3f}x (bar >= 1.05; 1.03-1.05 owner; < 1.03 parks): "
          f"{'PASS' if a2 >= 1.05 else 'owner' if a2 >= 1.03 else 'PARK'}")
    a4, _, q4 = ratios(c7, 4)
    print(f"GATE 3 7B 4-client aggregate {a4:.3f}x (>= 0.98) and p99 {q4:.3f}x (<= 1.02): {'PASS' if a4 >= 0.98 and q4 <= 1.02 else 'FAIL'}")
    _, s50, s99 = ratios(c7, 1)
    print(f"GATE 4 7B lone request p50 {s50:.3f}x p99 {s99:.3f}x (<= 1.05 each): {'PASS' if s50 <= 1.05 and s99 <= 1.05 else 'FAIL'}")
    a15, _, _ = ratios(c15, 2)
    print(f"GATE 5 1.5B 2-client aggregate {a15:.3f}x (>= 0.98): {'PASS' if a15 >= 0.98 else 'FAIL'}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
