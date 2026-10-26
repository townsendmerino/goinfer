#!/usr/bin/env python3
"""SP0's projection (README.md registers it): reads TestSP0_prize's JSONL and prints, per class and pooled, tokens per cycle, the
cost arms and the projected speedup per width, at each context length, with the band from the verify cost's quartiles.

  project.py validate <validate.jsonl>             the harness's own gate: simulated rounds against BlockSpec.Generate's
  project.py dflash   <a.jsonl> [<b.jsonl> ...]    DFlash: speedup(w) = tpc(w) * t_step / (t_verifyN(w) + t_draft(w))
  project.py ngram    <a.jsonl>                    n-gram: speedup(K) = tokens * t_step / sum over rounds of the round's cost

Pooled means the ratio of sums over the prompts (tokens over rounds), not a mean of ratios. Cost files: the first file that has
cost lines is used.
"""
import json, statistics, sys
from collections import defaultdict

RULE_GO, RULE_PARK = 1.30, 1.20


def load(paths):
    rows = []
    for p in paths:
        rows += [json.loads(l) for l in open(p)]
    return rows


def quart(xs):
    q = statistics.quantiles(xs, n=4)
    return q[0], statistics.median(xs), q[2]


def costs(rows):
    c = defaultdict(dict)  # ctx -> (arm, w) -> (q1, med, q3)
    for r in rows:
        if r["kind"] == "cost":
            c[r["ctx"]][(r["arm"], r["w"])] = quart(r["ms"])
    return c


def verdict(x):
    return "GO (>= 1.30)" if x >= RULE_GO else ("parked (1.20-1.30)" if x >= RULE_PARK else "no (< 1.20)")


def validate(rows):
    """The harness's own gate. A prompt on which BlockSpec's acceptance guard stopped the loop before the end of the continuation (its
    committed tokens fall more than one block short) is graded on the rounds the loop ran, and named; every other prompt on all of
    its rounds, with its round total counted toward the 2% bar."""
    loops = {(r["suite"], r["idx"]): r for r in rows if r["kind"] == "teacher"}
    sims = {(r["suite"], r["idx"]): r for r in rows if r["kind"] == "accept_dflash" and r["w"] == 8}
    same = total = rl = rs = 0
    tripped = []
    for k, lp in sorted(loops.items()):
        a, b = lp["loop_commits"], sims[k]["commits"]
        trip = sum(a) + 8 < sims[k]["tokens"]
        n = len(a) if trip else max(len(a), len(b))
        s = sum(1 for i in range(min(len(a), len(b), n)) if a[i] == b[i])
        same, total = same + s, total + n
        if trip:
            tripped.append(f"{k[0]}/{k[1]}")
        else:
            rl, rs = rl + lp["loop_rounds"], rs + len(b)
        print(f"{k[0]}/{k[1]}: loop {lp['loop_rounds']} rounds (lossless {lp['loop_equals_greedy']}), simulation {len(b)}; "
              f"{s}/{n} rounds identical" + (" (the guard stopped the loop: graded on the rounds it ran)" if trip else ""))
    frac = same / total
    drift = abs(rs - rl) / rl if rl else 0.0
    ok = frac >= 0.95 and drift <= 0.02 and rl > 0
    print(f"\nrounds identical {same}/{total} = {frac:.3f} (bar 0.95); untripped prompts' total rounds loop {rl} vs simulation {rs}, "
          f"{drift*100:.2f}% (bar 2%); guard tripped on {len(tripped)} of {len(loops)}: {', '.join(tripped) or 'none'}")
    print("HARNESS VALID" if ok else "HARNESS VOID: every DFlash reading is void")


def dflash(rows):
    c = costs(rows)
    acc = defaultdict(lambda: [0, 0])  # (class, w) -> [tokens, rounds]
    for r in rows:
        if r["kind"] == "accept_dflash":
            cls = r["suite"] + ("+think" if r.get("_think") else "")
            for key in ((cls, r["w"]), ("pooled", r["w"])):
                acc[key][0] += r["tokens"]
                acc[key][1] += r["rounds"]
    classes = sorted({k[0] for k in acc}, key=lambda s: (s == "pooled", s))
    widths = sorted({k[1] for k in acc})
    for L, cl in sorted(c.items()):
        step = cl[("step", 1)]
        print(f"\n=== context {L}: step {step[1]:.2f} ms (IQR {step[0]:.2f}-{step[2]:.2f})")
        print("  w   verifyN ms  forwardN ms  draft ms   " + "  ".join(f"{s:>16}" for s in classes))
        best = (0, 0)
        for w in widths:
            v, f, d = cl[("verifyN", w)], cl[("forwardN", w)], cl[("draft", w)]
            cells = []
            for s in classes:
                tok, rnd = acc[(s, w)]
                tpc = tok / rnd
                sp = tpc * step[1] / (v[1] + d[1])
                lo, hi = tpc * step[1] / (v[2] + d[2]), tpc * step[1] / (v[0] + d[0])
                cells.append(f"{tpc:4.2f}t {sp:4.2f}x [{lo:4.2f},{hi:4.2f}]")
                if s == "pooled" and sp > best[0]:
                    best = (sp, w, lo, hi)
            print(f"  {w:<3} {v[1]:9.2f}  {f[1]:10.2f}  {d[1]:8.2f}   " + "  ".join(f"{x:>16}" for x in cells))
        sp, w, lo, hi = best
        span = " (its band spans 1.30: parked)" if lo < RULE_GO <= hi else ""
        print(f"  best pooled: w={w}, {sp:.2f}x [{lo:.2f}, {hi:.2f}] -> {verdict(sp)}{span}")


def ngram(rows):
    c = costs(rows)
    per = defaultdict(lambda: [0, defaultdict(int)])  # (class, k) -> [tokens, hist]
    for r in rows:
        if r["kind"] == "accept_ngram":
            for key in ((r["suite"], r["k"]), ("pooled", r["k"])):
                per[key][0] += r["tokens"]
                for m, n in r["drafted_hist"].items():
                    per[key][1][int(m)] += n
    classes = sorted({k[0] for k in per}, key=lambda s: (s == "pooled", s))
    ks = sorted({k[1] for k in per})
    for L, cl in sorted(c.items()):
        step = cl[("step", 1)]
        vcost = {}
        for w in range(2, 10):
            f, v = cl[("forwardN", w)], cl[("verifyN", w)]
            vcost[w] = min((f, "forwardN"), (v, "verifyN"), key=lambda x: x[0][1])
        print(f"\n=== context {L}: step {step[1]:.2f} ms; verify per width (the cheaper arm): "
              + ", ".join(f"{w}:{vcost[w][0][1]:.1f}({vcost[w][1][0]})" for w in range(2, 10)))
        best = (0, 0)
        for k in ks:
            cells = []
            for s in classes:
                tok, hist = per[(s, k)]
                rounds = sum(hist.values())
                t = {i: sum(n * (step[i] if m == 0 else vcost[1 + m][0][i]) for m, n in hist.items()) for i in (0, 1, 2)}
                sp, lo, hi = tok * step[1] / t[1], tok * step[1] / t[2], tok * step[1] / t[0]
                hit = 1 - hist.get(0, 0) / rounds
                cells.append(f"{tok/rounds:4.2f}t hit{hit:4.2f} {sp:4.2f}x [{lo:4.2f},{hi:4.2f}]")
                if s == "pooled" and sp > best[0]:
                    best = (sp, k, lo, hi)
            print(f"  K={k:<2} " + "  ".join(f"{s}: {x}" for s, x in zip(classes, cells)))
        sp, k, lo, hi = best
        span = " (its band spans 1.30: parked)" if lo < RULE_GO <= hi else ""
        print(f"  best pooled: K={k}, {sp:.2f}x [{lo:.2f}, {hi:.2f}] -> {verdict(sp)}{span}")


if __name__ == "__main__":
    mode, paths = sys.argv[1], sys.argv[2:]
    rows = []
    for p in paths:
        think = "think" in p
        for r in load([p]):
            r["_think"] = think
            rows.append(r)
    {"validate": validate, "dflash": dflash, "ngram": ngram}[mode](rows)
