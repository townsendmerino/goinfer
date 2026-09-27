#!/usr/bin/env python3
"""MC1 on CUDA: the W7 gates (docs/tasks/task-concurrency-2026-09.md MC1, pre-registered before any timing) from
bench_w7_plain.py's JSON.

Cells are keyed <arm><clients>_<pair>: old1_1 .. new4_3 (1.5B), and the reported 7B pair old7b4_1 / new7b4_1 in a second file. p50 and
p99 are nearest-rank over every turn of a cell (6 turns at 1 client, so p99 is the cell's worst turn).

Gates (hard unless marked):
  1. identity: at 1 client old == new content_sha, every turn, every pair; at 2 and 4 clients new's client 0 == new's
     1-client run, every turn. Old vs new at 2 and 4 clients is reported only.
  2. reuse: new's prompt - prefill_reused_tokens at 2 and 4 clients equals its 1-client run's, turn for turn, on every
     client (turn 1: the cold prefill, reused 0 on a fresh server; turns 2-6: only the new user turn).
  3. (expected band, not a gate) new's 2- and 4-client aggregate is 0.90-1.0x its 1-client aggregate.
  4. ship: 4-client aggregate new / old, median of 3 pairs >= 1.03x.
  5. solo guard: 1-client p50 and p99 turn, new / old, median of 3 pairs <= 1.05x each.
"""
import json, math, statistics, sys


def pct(xs, p):
    s = sorted(xs)
    return s[max(0, math.ceil(p / 100 * len(s)) - 1)]


def turns(cell):
    return [t for c in cell["per_client"] for t in c]


def prefilled(t):
    return (t["prompt_tokens"] or 0) - (t["prefill_reused_tokens"] or 0)


def main(path, path7b=None):
    R = json.load(open(path))["results"]
    cells = {k: cell for k, by_n in R.items() for cell in by_n.values()}
    ok = True

    def cell(arm, n, i):
        return cells.get(f"{arm}{n}_{i}")

    # 1. identity, 1 client: old == new, every pair, every turn; and all new 1-client cells agree.
    ref1 = cell("new", 1, 1)
    idfail = 0
    for i in (1, 2, 3):
        for arm in ("old", "new"):
            c = cell(arm, 1, i)
            if not c or not ref1:
                print(f"GATE 1 MISSING: {arm}1_{i}")
                ok = False
                continue
            for ta, tb in zip(ref1["per_client"][0], c["per_client"][0]):
                if ta["content_sha"] != tb["content_sha"]:
                    print(f"GATE 1 FAIL: {arm}1_{i} turn {tb['n']}: content {tb['content_sha']} != new1_1's {ta['content_sha']}")
                    idfail += 1
    # 1. identity at 2 and 4 clients: new's client 0 == new's 1-client run. 2. reuse: every client prefills what
    # the 1-client run's client 0 prefills, turn for turn.
    reusefail = 0
    ref_pre = [prefilled(t) for t in ref1["per_client"][0]] if ref1 else []
    for n in (2, 4):
        for i in (1, 2, 3):
            c = cell("new", n, i)
            if not c:
                print(f"GATE 1/2 MISSING: new{n}_{i}")
                ok = False
                continue
            for ta, tb in zip(ref1["per_client"][0], c["per_client"][0]):
                if ta["content_sha"] != tb["content_sha"]:
                    print(f"GATE 1 FAIL: new{n}_{i} client 0 turn {tb['n']}: content {tb['content_sha']} != new1_1's {ta['content_sha']}")
                    idfail += 1
            for ci, cl in enumerate(c["per_client"]):
                got = [prefilled(t) for t in cl]
                # turn 1 is the whole first prompt, whose nonce tokenizes to a client-specific length: compare its
                # reuse (0, cold) rather than its prefill count; turns 2-6 compare prefill counts exactly.
                t1_reused = cl[0]["prefill_reused_tokens"] or 0
                if t1_reused != (ref1["per_client"][0][0]["prefill_reused_tokens"] or 0) or got[1:] != ref_pre[1:]:
                    print(f"GATE 2 FAIL: new{n}_{i} client {ci}: prefilled {got} (turn-1 reused {t1_reused}) vs 1-client {ref_pre}")
                    reusefail += 1
    # reported: old vs new at 2 and 4 clients.
    for n in (2, 4):
        for i in (1, 2, 3):
            o, w = cell("old", n, i), cell("new", n, i)
            if o and w:
                diff = sum(ta["content_sha"] != tb["content_sha"] for a, b in zip(o["per_client"], w["per_client"]) for ta, tb in zip(a, b))
                print(f"reported: old vs new at {n} clients pair {i}: {diff} of {len(turns(w))} turns differ in content")
    print(f"GATE 1 identity: {'PASS' if idfail == 0 else f'FAIL ({idfail} turns)'}")
    print(f"GATE 2 reuse: {'PASS' if reusefail == 0 else f'FAIL ({reusefail} clients)'}")
    ok = ok and idfail == 0 and reusefail == 0

    # 3. expected band: new's 2/4-client aggregate / its 1-client aggregate (same pair index).
    for n in (2, 4):
        rs = [cell("new", n, i)["aggregate_tok_s"] / cell("new", 1, i)["aggregate_tok_s"]
              for i in (1, 2, 3) if cell("new", n, i) and cell("new", 1, i)]
        if rs:
            m = statistics.median(rs)
            note = "in band" if 0.90 <= m <= 1.0 else ("FINDING (< 0.85x at 4 clients)" if n == 4 and m < 0.85 else "outside band")
            print(f"BAND 3 new {n} clients / new 1 client: {', '.join(f'{r:.3f}' for r in rs)}; median {m:.3f}x — {note}")
        ro = [cell("old", n, i)["aggregate_tok_s"] / cell("old", 1, i)["aggregate_tok_s"]
              for i in (1, 2, 3) if cell("old", n, i) and cell("old", 1, i)]
        if ro:
            print(f"reported: old {n} clients / old 1 client: {', '.join(f'{r:.3f}' for r in ro)}; median {statistics.median(ro):.3f}x")

    # 4. ship: 4-client aggregate new / old; also 2 clients, reported.
    for n in (4, 2):
        rs = []
        for i in (1, 2, 3):
            o, w = cell("old", n, i), cell("new", n, i)
            if o and w:
                rs.append(w["aggregate_tok_s"] / o["aggregate_tok_s"])
                lo, lw = [t["latency_ms"] for t in turns(o)], [t["latency_ms"] for t in turns(w)]
                print(f"pair {i} {n} clients: aggregate old {o['aggregate_tok_s']:.2f} new {w['aggregate_tok_s']:.2f} tok/s = {rs[-1]:.3f}x; "
                      f"p50 turn old {pct(lo, 50)/1000:.2f} new {pct(lw, 50)/1000:.2f} s; p99 turn old {pct(lo, 99)/1000:.2f} new {pct(lw, 99)/1000:.2f} s")
        if rs and n == 4:
            m = statistics.median(rs)
            print(f"GATE 4 ship: 4-client aggregate new/old median {m:.3f}x (bar >= 1.03x): {'PASS' if m >= 1.03 else 'FAIL'}")
            ok = ok and m >= 1.03
        elif rs:
            print(f"reported: 2-client aggregate new/old median {statistics.median(rs):.3f}x")

    # 5. solo guard.
    s50, s99 = [], []
    for i in (1, 2, 3):
        o, w = cell("old", 1, i), cell("new", 1, i)
        if o and w:
            lo, lw = [t["latency_ms"] for t in turns(o)], [t["latency_ms"] for t in turns(w)]
            s50.append(pct(lw, 50) / pct(lo, 50))
            s99.append(pct(lw, 99) / pct(lo, 99))
            print(f"pair {i} 1 client: p50 old {pct(lo, 50)/1000:.3f} new {pct(lw, 50)/1000:.3f} s = {s50[-1]:.3f}x; "
                  f"p99 old {pct(lo, 99)/1000:.3f} new {pct(lw, 99)/1000:.3f} s = {s99[-1]:.3f}x; "
                  f"aggregate old {o['aggregate_tok_s']:.2f} new {w['aggregate_tok_s']:.2f}")
    if s50:
        m50, m99 = statistics.median(s50), statistics.median(s99)
        g5 = m50 <= 1.05 and m99 <= 1.05
        print(f"GATE 5 solo guard: p50 median {m50:.3f}x, p99 median {m99:.3f}x (bar <= 1.05 each): {'PASS' if g5 else 'FAIL'}")
        ok = ok and g5
    else:
        ok = False

    # reported: the 7B pair (clamp).
    c7 = {k: c for k, by_n in json.load(open(path7b))["results"].items() for c in by_n.values()} if path7b else {}
    o, w = c7.get("old7b4_1"), c7.get("new7b4_1")
    if o and w:
        print(f"reported, 7B 4 clients: aggregate old {o['aggregate_tok_s']:.2f} new {w['aggregate_tok_s']:.2f} tok/s = "
              f"{w['aggregate_tok_s']/o['aggregate_tok_s']:.3f}x; new prefilled per client {[[prefilled(t) for t in c] for c in w['per_client']]}")
    print(f"ALL HARD GATES: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None))
