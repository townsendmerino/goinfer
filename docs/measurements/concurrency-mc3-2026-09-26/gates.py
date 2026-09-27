#!/usr/bin/env python3
"""MC3's W7 gates (docs/tasks/task-concurrency-2026-09.md, pre-registered in 505cc6fc) from bench_w7_plain.py's JSON.

Cells are keyed <arm><clients>_<pair>: old4_1 .. new1_3. p99 is nearest-rank over every turn of a cell (24 turns at 4
clients, 6 at 1 client, so it is the cell's worst turn); p50 likewise.
"""
import json, math, statistics, sys


def pct(xs, p):
    s = sorted(xs)
    return s[max(0, math.ceil(p / 100 * len(s)) - 1)]


def turns(cell):
    return [t for c in cell["per_client"] for t in c]


def main(path):
    R = json.load(open(path))["results"]
    cells = {}
    for key, by_n in R.items():
        for n, cell in by_n.items():
            cells[key] = cell
    ok, idfail = True, 0

    # 1. identity and 2. reuse: every cell at the same client count agrees turn by turn, client by client.
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
                        print(f"GATE 1 FAIL: {k} client {ci} turn {tb['n']}: content {tb['content_sha']} != {ref_key}'s {ta['content_sha']}")
                        ok, idfail = False, idfail + 1
                    pa = (ta["prompt_tokens"] or 0) - (ta["prefill_reused_tokens"] or 0)
                    pb = (tb["prompt_tokens"] or 0) - (tb["prefill_reused_tokens"] or 0)
                    if pa != pb:
                        print(f"GATE 2 FAIL: {k} client {ci} turn {tb['n']}: prefilled {pb} != {ref_key}'s {pa}")
                        ok, idfail = False, idfail + 1
        print(f"clients={n}: {len(group)} cells compared turn by turn against {ref_key} (content and prompt-reused)")

    def pairs(n):
        out = []
        for i in (1, 2, 3):
            o, w = cells.get(f"old{n}_{i}"), cells.get(f"new{n}_{i}")
            if o and w:
                out.append((i, o, w))
        return out

    # 3. aggregate and 4. p99 under load, 4 clients.
    agg, p99 = [], []
    for i, o, w in pairs(4):
        a = w["aggregate_tok_s"] / o["aggregate_tok_s"]
        lo = [t["latency_ms"] for t in turns(o)]
        lw = [t["latency_ms"] for t in turns(w)]
        r = pct(lw, 99) / pct(lo, 99)
        agg.append(a)
        p99.append(r)
        print(f"pair {i} 4 clients: aggregate old {o['aggregate_tok_s']:.1f} new {w['aggregate_tok_s']:.1f} tok/s = {a:.3f}x; "
              f"p99 turn old {pct(lo, 99)/1000:.2f} s new {pct(lw, 99)/1000:.2f} s = {r:.3f}x; "
              f"p50 old {pct(lo, 50)/1000:.2f} new {pct(lw, 50)/1000:.2f} s")
    if agg:
        ma, mp = statistics.median(agg), statistics.median(p99)
        print(f"GATE 3 aggregate median {ma:.3f}x (bar >= 1.2x): {'PASS' if ma >= 1.2 else 'owner band' if ma >= 1.03 else 'FAIL (park)'}")
        print(f"GATE 4 p99 under load median {mp:.3f}x (bar <= 1.0): {'PASS' if mp <= 1.0 else 'FAIL'}")
        ok = ok and mp <= 1.0

    # 5. solo guard, 1 client.
    s50, s99 = [], []
    for i, o, w in pairs(1):
        lo = [t["latency_ms"] for t in turns(o)]
        lw = [t["latency_ms"] for t in turns(w)]
        s50.append(pct(lw, 50) / pct(lo, 50))
        s99.append(pct(lw, 99) / pct(lo, 99))
        print(f"pair {i} 1 client: p50 old {pct(lo, 50)/1000:.3f} new {pct(lw, 50)/1000:.3f} s = {s50[-1]:.3f}x; "
              f"p99 old {pct(lo, 99)/1000:.3f} new {pct(lw, 99)/1000:.3f} s = {s99[-1]:.3f}x; "
              f"aggregate old {o['aggregate_tok_s']:.1f} new {w['aggregate_tok_s']:.1f}")
    if s50:
        m50, m99 = statistics.median(s50), statistics.median(s99)
        g5 = m50 <= 1.05 and m99 <= 1.05
        print(f"GATE 5 solo guard: p50 median {m50:.3f}x, p99 median {m99:.3f}x (bar <= 1.05 each): {'PASS' if g5 else 'FAIL'}")
        ok = ok and g5

    # reported: 2 clients, and 4-client p99 / a lone request's.
    for i, o, w in pairs(2):
        lo = [t["latency_ms"] for t in turns(o)]
        lw = [t["latency_ms"] for t in turns(w)]
        print(f"reported, 2 clients: aggregate old {o['aggregate_tok_s']:.1f} new {w['aggregate_tok_s']:.1f} = "
              f"{w['aggregate_tok_s']/o['aggregate_tok_s']:.3f}x; p99 old {pct(lo, 99)/1000:.2f} new {pct(lw, 99)/1000:.2f} s")
    solo = [pct([t["latency_ms"] for t in turns(w)], 99) for _, _, w in pairs(1)]
    load = [pct([t["latency_ms"] for t in turns(w)], 99) for _, _, w in pairs(4)]
    if solo and load:
        print(f"reported: new's 4-client p99 / new's lone-request p99 = {statistics.median(load)/statistics.median(solo):.2f}x")
    print(f"GATES 1-2 identity and reuse: {'PASS' if idfail == 0 else f'FAIL ({idfail} turns)'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
