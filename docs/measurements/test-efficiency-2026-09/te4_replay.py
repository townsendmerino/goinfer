#!/usr/bin/env python3
"""TE4 replay: a pre-registered sequential stopping rule (TE4-SEQ-v1) replayed over past paired A/B gates.

Read-only over docs/measurements; stdlib only; runs and times nothing. Re-run from the repo root:

    python3 docs/measurements/test-efficiency-2026-09/te4_replay.py

The rule is fixed in te4-replay-2026-09-28.md §1, written before this script existed. Summary:
  * served unit = one completion; pair = completion k of run r, new cell vs old cell, same pass; d = ln(new/old);
    block = one run of each arm; a look after every block.
  * in-process unit = one logged ABBA pair; d = ln(logged speed-up); looks at n >= 3.
  * Lan-DeMets O'Brien-Fleming-type spending alpha*(t) = 2 - 2 Phi(z_0.975 / sqrt t), two-sided 0.05, t = pairs /
    pairs at the cap; z boundaries by recursive integration, carried to t_{n-1} at the same tail.
  * interval = mean -/+ (c* SE + delta), delta = 0.022 served, 0 in-process.
  * stop PASS if lower > ln(ship); stop non-pass if upper < ln(kill); at the cap the gate's own statistic and bands.
"""
import ast
import itertools
import json
import math
import os
import random
import statistics
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
M = os.path.join(REPO, "docs", "measurements")

ALPHA = 0.05
DELTA_SERVED = 0.022
DELTA_INPROC = 0.0
MIN_N = 3
PERM_MAX = 200
PERM_SEED = 20260928

# ----------------------------------------------------------------------------------------------------------------
# Statistics (stdlib only)
# ----------------------------------------------------------------------------------------------------------------
SQRT2 = math.sqrt(2.0)


def Phi(x):
    return 0.5 * (1.0 + math.erf(x / SQRT2))


def phi(x):
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def Phi_inv(p):
    lo, hi = -40.0, 40.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if Phi(mid) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _betacf(a, b, x):
    FPMIN = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > FPMIN else FPMIN)
    h = d
    for m in range(1, 500):
        m2 = 2 * m
        for aa in (m * (b - m) * x / ((qam + m2) * (a + m2)),
                   -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))):
            d = 1.0 + aa * d
            d = 1.0 / (d if abs(d) > FPMIN else FPMIN)
            c = 1.0 + aa / c
            c = c if abs(c) > FPMIN else FPMIN
            de = d * c
            h *= de
        if abs(de - 1.0) < 3e-16:
            break
    return h


def betainc(a, b, x):
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    bt = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log(1.0 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def t_cdf(t, nu):
    tail = 0.5 * betainc(nu / 2.0, 0.5, nu / (nu + t * t))
    return 1.0 - tail if t >= 0 else tail


_TCACHE = {}


def t_inv(p, nu):
    key = (p, nu)
    if key not in _TCACHE:
        _TCACHE[key] = _t_inv(p, nu)
    return _TCACHE[key]


def _t_inv(p, nu):
    lo, hi = -1e7, 1e7
    for _ in range(400):
        mid = 0.5 * (lo + hi)
        if t_cdf(mid, nu) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def spend(t, alpha=ALPHA, zq=None):
    """Lan-DeMets O'Brien-Fleming-type two-sided spending, alpha*(t) = 2 - 2 Phi(z_{1-alpha/2} / sqrt t)."""
    if t <= 0:
        return 0.0
    z = Phi_inv(1.0 - alpha / 2.0) if zq is None else zq
    return min(alpha, 2.0 - 2.0 * Phi(z / math.sqrt(t)))


_BCACHE = {}


def ld_boundaries(ts, alpha=ALPHA, ngrid=401, spend_fn=None):
    """Two-sided symmetric z boundaries for looks at information fractions ts (increasing, ts[-1] == 1).
    Recursive numerical integration (Armitage-McPherson-Rowe) of the continuation sub-density of W = Z sqrt(t),
    Simpson's rule on each continuation interval."""
    key = (tuple(round(t, 12) for t in ts), alpha, ngrid, spend_fn)
    if key in _BCACHE:
        return _BCACHE[key]
    sf = spend_fn or (lambda t: spend(t, alpha))
    cs, grid, prev, tprev = [], None, 0.0, 0.0
    for j, t in enumerate(ts):
        target = sf(t) - prev
        if j == 0:
            c = Phi_inv(1.0 - target / 2.0) if target > 0 else 40.0
        else:
            sd = math.sqrt(t - tprev)
            ws, gs, wts = grid
            st = math.sqrt(t)

            def exit_prob(cc):
                b = cc * st
                return sum(wt * g * (1.0 - (Phi((b - w) / sd) - Phi((-b - w) / sd))) for w, g, wt in zip(ws, gs, wts))
            if target <= 0:
                c = 40.0
            else:
                lo, hi = 0.0, 40.0
                for _ in range(80):
                    mid = 0.5 * (lo + hi)
                    if exit_prob(mid) > target:
                        lo = mid
                    else:
                        hi = mid
                c = 0.5 * (lo + hi)
        cs.append(c)
        b = c * math.sqrt(t)
        n = ngrid
        ys = [-b + 2.0 * b * i / (n - 1) for i in range(n)]
        h = 2.0 * b / (n - 1)
        wy = [h / 3.0 * (1 if i in (0, n - 1) else (4 if i % 2 else 2)) for i in range(n)]
        if j == 0:
            s0 = math.sqrt(t)
            gy = [phi(y / s0) / s0 for y in ys]
        else:
            sd = math.sqrt(t - tprev)
            ws, gs, wts = grid
            gy = [sum(wt * g * phi((y - w) / sd) / sd for w, g, wt in zip(ws, gs, wts)) for y in ys]
        grid, prev, tprev = (ys, gy, wy), sf(t), t
    _BCACHE[key] = cs
    return cs


def self_check():
    """The integrator must reproduce published values before anything else is trusted: gsDesign's sfLDOF
    (one-sided 0.025 per side, symmetric), K = 5 equally spaced: 4.877 3.357 2.680 2.290 2.031; K = 3: 3.710 2.511
    1.993. And t_{7} 0.975 = 2.3646."""
    z = Phi_inv(1.0 - 0.025 / 2.0)
    sf = (lambda t: 2.0 * (2.0 - 2.0 * Phi(z / math.sqrt(t))))
    k5 = [round(c, 3) for c in ld_boundaries([0.2, 0.4, 0.6, 0.8, 1.0], spend_fn=sf)]
    k3 = [round(c, 3) for c in ld_boundaries([1 / 3, 2 / 3, 1.0], spend_fn=sf)]
    ok = k5 == [4.877, 3.357, 2.68, 2.29, 2.031] and k3 == [3.71, 2.511, 1.993] and abs(t_inv(0.975, 7) - 2.3646) < 1e-3
    print(f"self-check: gsDesign sfLDOF K=5 {k5}, K=3 {k3}; t_7(0.975) = {t_inv(0.975, 7):.4f} -> {'OK' if ok else 'FAILED'}")
    if not ok:
        sys.exit("self-check failed: the boundary integrator does not reproduce published values")
    reg = [round(c, 3) for c in ld_boundaries([1 / 3, 2 / 3, 1.0])]
    print(f"TE4-SEQ-v1 spending (alpha*(t) = 2 - 2 Phi(1.96/sqrt t)), K=3 equally spaced: z boundaries {reg}")




# ----------------------------------------------------------------------------------------------------------------
# Data: bench_peer JSONs (served), in-process logs
# ----------------------------------------------------------------------------------------------------------------
_JCACHE = {}


def counts_of(c):
    v = c.get("counts") or {}
    return v if isinstance(v, dict) else ast.literal_eval(v)  # some harness versions stored a repr'd dict


def load_bench(path):
    if path not in _JCACHE:
        d = json.load(open(os.path.join(M, path)))
        if not (isinstance(d, list) and d and isinstance(d[0], dict) and d[0].get("kind") == "provenance"):
            raise ValueError(f"{path}: not bench_peer-shaped")
        _JCACHE[path] = [c for c in d[1:] if isinstance(c, dict) and "runs" in c]
    return _JCACHE[path]


def find_cell(path, engine, model, backend, depth=128, config="greedy", phase=None):
    hits = [c for c in load_bench(path)
            if c.get("engine") == engine and c.get("model") == model and c.get("backend") == backend
            and c.get("depth") == depth and c.get("config") == config and c.get("runs")
            and (phase is None or c.get("phase") == phase)]
    if len(hits) != 1:
        raise KeyError(f"{path}: {len(hits)} cells for {engine}/{model}/{backend}/{depth}/{config}/{phase}")
    return hits[0]


def cell_id(path, c):
    return "|".join(str(x) for x in (path, c.get("phase"), c["engine"], c["model"], c["backend"], c["depth"], c["config"]))


def split_runs(cell):
    """Split counts.completion_rates into runs, verified against runs[] (each run's mean must match its entry)."""
    runs = cell["runs"]
    cr = counts_of(cell).get("completion_rates") or []
    if not cr:
        raise ValueError("no completion_rates")
    nr = len(runs)
    tol = lambda r: 1e-9 * max(1.0, abs(r))
    if len(cr) % nr == 0:
        k = len(cr) // nr
        chunks = [cr[i * k:(i + 1) * k] for i in range(nr)]
        if all(abs(statistics.mean(ch) - r) <= tol(r) for ch, r in zip(chunks, runs)):
            return chunks

    def rec(start, ri):  # unequal run sizes: find the split whose means match runs[]
        if ri == nr:
            return [] if start == len(cr) else None
        for end in range(start + 1, len(cr) + 1):
            if abs(statistics.mean(cr[start:end]) - runs[ri]) <= tol(runs[ri]):
                rest = rec(end, ri + 1)
                if rest is not None:
                    return [cr[start:end]] + rest
        return None
    chunks = rec(0, 0)
    if chunks is None:
        raise ValueError("completion_rates cannot be split to match runs[]")
    return chunks


def cell_stat(cell, how):
    """A gate's own per-cell statistic, as its record computed it."""
    if how == "mean_runs":
        return statistics.mean(cell["runs"])
    if how == "median_runs":
        return statistics.median(cell["runs"])
    if how == "mean_field":  # bench_peer.py's rec["mean"], rounded to 0.1 tok/s
        return cell["mean"]
    raise ValueError(how)


def pair_blocks(path, ca, cb):
    """Completion-level pairs, run by run: block r = [ln(a_k) - ln(b_k)] over completion k of run r."""
    ra, rb = split_runs(ca), split_runs(cb)
    if len(ra) != len(rb):
        raise ValueError(f"{path}: run counts differ")
    ia, ib = cell_id(path, ca), cell_id(path, cb)
    blocks, cells = [], []
    for x, y in zip(ra, rb):
        k = min(len(x), len(y))
        blocks.append([math.log(a) - math.log(b) for a, b in zip(x[:k], y[:k])])
        cells.append((ia, ib))
    return blocks, cells


def ab_sub(passes, model, backend, depth=128, config="greedy", phase=None, stat="mean_runs"):
    """Served new-vs-old. passes = [(path, new_label, old_label), ...] in recorded order."""
    def load():
        blocks, cells, ratios, shown = [], [], [], []
        for path, nl, ol in passes:
            cn = find_cell(path, nl, model, backend, depth, config, phase)
            co = find_cell(path, ol, model, backend, depth, config, phase)
            b, c = pair_blocks(path, cn, co)
            blocks += b
            cells += c
            ratios.append(cell_stat(cn, stat) / cell_stat(co, stat))
        s = math.exp(statistics.mean(math.log(r) for r in ratios))
        return {"blocks": blocks, "cells": cells, "stat": s, "pass_ratios": ratios}
    return load


PEER_LO, PEER_HI = 0.97, 1.03


def peer_outcome(rs, capped):
    """peer-claim-2026-09-25/grade.py's outcome(), verbatim in logic."""
    if capped:
        return "AMBIGUOUS-LOW" if any(r < PEER_LO for r in rs) else "AMBIGUOUS-HIGH"
    if all(r > PEER_HI for r in rs):
        return "AHEAD"
    if all(PEER_LO <= r <= PEER_HI for r in rs):
        return "LEVEL"
    if all(r < PEER_LO for r in rs):
        return "BEHIND"
    if any(r < PEER_LO for r in rs):
        return "AMBIGUOUS-LOW"
    return "AMBIGUOUS-HIGH"


def peer_sub(path, model, backend, depth, config, peer):
    def load():
        cg = find_cell(path, "goinfer", model, backend, depth, config)
        cp = find_cell(path, peer, model, backend, depth, config)
        blocks, cells = pair_blocks(path, cg, cp)
        rs = [a / b for a, b in zip(cg["runs"], cp["runs"])]
        spread = lambda xs: (max(xs) - min(xs)) / statistics.mean(xs)
        capped = spread(cg["runs"]) > 0.05 or spread(cp["runs"]) > 0.05
        return {"blocks": blocks, "cells": cells, "stat": statistics.median(rs), "pairs": rs,
                "cap_label": peer_outcome(rs, capped), "runpairs": list(zip(cg["runs"], cp["runs"]))}
    return load


def abba_sub(logpath, label):
    """cpu_roofline_ab_test.go lines: '... (<label>) pair i: ON a | OFF b ms/tok | ON is Rx faster'."""
    def load():
        rs = [float(l.split("ON is")[1].split("x")[0]) for l in open(os.path.join(M, logpath))
              if f"({label}) pair " in l and "ON is" in l]
        if not rs:
            raise ValueError(f"{logpath}: no pairs for {label}")
        return {"blocks": [[math.log(r)] for r in rs],
                "cells": [(f"{logpath}|{label}|ON", f"{logpath}|{label}|OFF")] * len(rs),
                "stat": statistics.median(rs), "pairs": rs}
    return load


def rounds_sub(e2e, num, den, scale=1.0):
    """BenchmarkResidentDecode rounds: '<label> <round> BenchmarkResidentDecode-16 ... <x> tok/s', -count 2 per
    process; the first sample of each process is discarded (run.sh), so each process contributes its second."""
    def load():
        d = {}
        for l in open(os.path.join(M, e2e)):
            p = l.split()
            if len(p) > 3 and l.rstrip().endswith("tok/s"):
                d.setdefault(p[0], {}).setdefault(int(p[1]), []).append(float(p[-2]))
        rounds = sorted(d[num])
        rs = [scale * d[num][r][1] / d[den][r][1] for r in rounds]
        return {"blocks": [[math.log(r)] for r in rs],
                "cells": [(f"{e2e}|{num}", f"{e2e}|{den}")] * len(rs),
                "stat": statistics.median(rs), "pairs": rs}
    return load


# ----------------------------------------------------------------------------------------------------------------
# The rule (TE4-SEQ-v1)
# ----------------------------------------------------------------------------------------------------------------
def sequential(blocks, S, K, delta, min_blocks=1, veto=None):
    """Replay blocks in the given order. Returns ("PASS"|"FAIL"|None, blocks_used, trace); None = reached the cap.
    min_blocks and veto exist only for the post-hoc variant (§6 of the record); their defaults are TE4-SEQ-v1."""
    ncap = sum(len(b) for b in blocks)
    looks, n = [], 0
    for i, b in enumerate(blocks):
        n += len(b)
        if n >= MIN_N:
            looks.append((i, n))
    cz = ld_boundaries([nn / ncap for _, nn in looks]) if looks else []
    data, trace, li = [], [], 0
    for i, b in enumerate(blocks):
        data.extend(b)
        if li < len(looks) and looks[li][0] == i:
            n = len(data)
            m = statistics.mean(data)
            se = statistics.stdev(data) / math.sqrt(n)
            ct = t_inv(Phi(cz[li]), n - 1)
            lo, hi = m - ct * se - delta, m + ct * se + delta
            trace.append({"block": i + 1, "n": n, "mean": m, "c": ct, "lo": lo, "hi": hi})
            li += 1
            blocked = (i + 1 < min_blocks) or (veto is not None and veto(i + 1))
            if i + 1 < len(blocks) and not blocked:  # before the cap the interval decides; at the cap, the gate's bands
                if lo > math.log(S):
                    return "PASS", i + 1, trace
                if hi < math.log(K):
                    return "FAIL", i + 1, trace
    return None, len(blocks), trace


def bands(stat, S, K):
    if stat >= S:
        return "PASS"
    if stat < K:
        return "FAIL"
    return "AMBIG"


RANK = {"PASS": 2, "AMBIG": 1, "FAIL": 0,
        "AHEAD": 4, "LEVEL": 3, "AMBIGUOUS-HIGH": 3, "AMBIGUOUS-LOW": 1, "BEHIND": 0,
        "NOT SLOWER": 2, "UNRESOLVED": 1, "FLAGGED": 0,
        "q4k DEFAULT": 2, "PARKED": 1, "int4 STAYS": 0}
TOP = {"PASS", "AHEAD", "NOT SLOWER", "q4k DEFAULT"}


def flip_kind(recorded, replay):
    if recorded == replay:
        return ""
    if replay in TOP and recorded not in TOP:
        return "DISQUALIFYING (non-pass -> pass)"
    if RANK.get(replay, 0) > RANK.get(recorded, 0):
        return "toward pass"
    if RANK.get(replay, 0) < RANK.get(recorded, 0):
        return "toward fail"
    return "sideways"


# ----------------------------------------------------------------------------------------------------------------
# The gates (inventory: te4-replay-2026-09-28.md §2). Quotes are verbatim; paths are relative to the repo root.
# ----------------------------------------------------------------------------------------------------------------
L1 = "cpu-decode-peer-gap-2026-09-27"
TL1 = "docs/tasks/task-cpu-decode-peer-gap-2026-09.md"
TQ = "docs/tasks/task-int4-weight-quality-2026-09.md"
PC = "docs/measurements/peer-claim-2026-09-25.md"


def ab(id_, load, S, K, recorded, rec_stat=None, graded=True):
    return {"id": id_, "load": load, "S": S, "K": K, "recorded": recorded, "rec_stat": rec_stat, "graded": graded,
            "pass_label": "PASS", "fail_label": "FAIL"}


def peer(id_, load, recorded, rec_stat=None, graded=True):
    return {"id": id_, "load": load, "S": PEER_HI, "K": PEER_LO, "recorded": recorded, "rec_stat": rec_stat,
            "graded": graded, "pass_label": "AHEAD", "fail_label": "BEHIND", "peer": True}


def combine_default(labels):
    if all(v == "PASS" for v in labels):
        return "PASS"
    if any(v == "FAIL" for v in labels):
        return "FAIL"
    return "AMBIG"


def combine_tuple(labels):
    c = {}
    for v in labels:
        c[v] = c.get(v, 0) + 1
    return " · ".join(f"{c[k]} {k}" for k in sorted(c, key=lambda k: -RANK.get(k, 0)))


def combine_q4k(labels_by_id):
    loa = {"AHEAD", "LEVEL", "AMBIGUOUS-HIGH"}
    q = [k.replace("S2 ", "") for k, v in labels_by_id.items() if k.startswith("S2 ") and v in loa]
    s1 = {k.replace("S1 ", ""): v for k, v in labels_by_id.items() if k.startswith("S1 ")}
    if all(s1[k] in loa for k in q):
        return "q4k DEFAULT"
    if any(s1[k] == "BEHIND" for k in q):
        return "int4 STAYS"
    return "PARKED"


def gate6_cap(sub):
    r1, r2 = sub["pass_ratios"]
    if (r1 >= 1.03 and r2 <= 0.97) or (r1 <= 0.97 and r2 >= 1.03):
        return "UNRESOLVED"
    if sub["stat"] < 1.00 and r1 < 1.00 and r2 < 1.00:
        return "FLAGGED"
    return "NOT SLOWER"


def s6_subs(files, rec):
    out = []
    for (model, phase, depth), r in zip([(m, p, d) for m in ("1.5B", "7B") for p, d in (("A", 128), ("B", 128), ("B", 2048))], rec):
        path = files[model]
        out.append(ab(f"{model} @{depth} ({phase})", ab_sub([(path, "goinfer", "goinfer_old")], model, "metal", depth,
                                                          "greedy", phase), 0.97, 0.95, "PASS", 1 + r / 100))
    return out


GATES = [
    {"id": "L1-gate5", "class": "served-AB", "machine": "nobara CPU (amd64)",
     "data": [f"{L1}/gate5-served.json"], "N": "3 runs × 8 completions, 1 pass",
     "bar": f"{TL1}:284-286 — \"SHIP (default) if the new ÷ old ratio is ≥ 1.03× on the 1.5B and the 7B, and ≥ 0.95× on "
            "the 0.5B … Anything else goes to the owner\"; statistic: ratio of run medians",
     "verdict_q": f"{TL1}:319 — \"5. **Served speed: SHIP.**\" (1.016× / 1.047× / 1.089×)",
     "recorded": "PASS", "names": {"PASS": "SHIP", "FAIL": "to the owner"},
     "subs": [ab("0.5B", ab_sub([(f"{L1}/gate5-served.json", "goinfer", "goinfer_old")], "0.5B", "cpu", stat="median_runs"), 0.95, 0.95, "PASS", 1.016),
              ab("1.5B", ab_sub([(f"{L1}/gate5-served.json", "goinfer", "goinfer_old")], "1.5B", "cpu", stat="median_runs"), 1.03, 1.03, "PASS", 1.047),
              ab("7B", ab_sub([(f"{L1}/gate5-served.json", "goinfer", "goinfer_old")], "7B", "cpu", stat="median_runs"), 1.03, 1.03, "PASS", 1.089)]},
    {"id": "L1-arm64-fix", "class": "served-AB", "machine": "MacBook CPU (arm64)",
     "data": [f"{L1}/arm64-fix-served.json"], "N": "3 runs × 8, pass 1 only (pass 2 registered, skipped by the 09:05 amendment)",
     "bar": f"{TL1}:486-488 — \"Closed: combined fix ÷ old ≥ 0.97 on all three models. Partial: 0.90–0.97 … Fallback: "
            "< 0.90\"; statistic: ratio of rounded cell means",
     "verdict_q": f"{TL1}:527 — \"**Served gate: CLOSED, and a gain.**\" (1.041× / 1.053× / 1.080×)",
     "recorded": "PASS", "names": {"PASS": "CLOSED", "AMBIG": "partial", "FAIL": "fallback"},
     "subs": [ab(m, ab_sub([(f"{L1}/arm64-fix-served.json", "goinfer", "goinfer_old")], m, "cpu", stat="mean_field"),
                 0.97, 0.90, "PASS", r) for m, r in (("0.5B", 1.041), ("1.5B", 1.053), ("7B", 1.080))]},
    {"id": "S6-gates-2+4", "class": "served-AB", "machine": "MacBook Metal",
     "data": ["s6-alias-2026-09-24/gates-decode/results.json (1.5B)", "s6-alias-2026-09-24/gates-decode/results-7b.json (7B re-run)"],
     "N": "3 runs × 8, 1 pass",
     "bar": "docs/measurements/s6-alias-2026-09-24.md:115-116 — \"pre-registered: pass ≤ 3% slower, 3–5% ambiguous/parked, "
            "> 5% fail\" (docs/tasks/task-never-swap-2026-09.md:1101-1105: \"within 3% of the copied path\")",
     "verdict_q": "docs/measurements/s6-alias-2026-09-24.md:114 — \"**Gates 2 and 4 — decode within 3% at depth 128 and 2048: "
                  "PASS, worst −0.35%**\"",
     "recorded": "PASS", "names": {},
     "subs": s6_subs({"1.5B": "s6-alias-2026-09-24/gates-decode/results.json",
                      "7B": "s6-alias-2026-09-24/gates-decode/results-7b.json"}, [-0.07, -0.13, 0.34, -0.31, -0.35, -0.27])},
    {"id": "S6-v14", "class": "served-AB", "machine": "MacBook Metal",
     "data": ["s6-alias-2026-09-24/step2-v14/results-v14.json"], "N": "3 runs × 8, 1 pass",
     "bar": "as S6-gates-2+4 (the same gates re-run on .giw v14)",
     "verdict_q": "docs/measurements/s6-alias-2026-09-24.md:196-197 — \"Gates 2/4 (decode within 3%), re-run on v14, "
                  "interleaved, 3 runs per cell: **PASS, worst −1.49%**\"",
     "recorded": "PASS", "names": {},
     "subs": s6_subs({"1.5B": "s6-alias-2026-09-24/step2-v14/results-v14.json",
                      "7B": "s6-alias-2026-09-24/step2-v14/results-v14.json"}, [-0.95, -0.18, -0.20, -1.40, -1.49, -1.19])},
    {"id": "q4k-peer", "class": "served-peer", "machine": "nobara CUDA",
     "data": ["q4k-peer-2026-09-26/s2-int4.json (S2)", "q4k-peer-2026-09-26/s1-q4k.json (S1)"], "N": "3 runs × 8 per cell",
     "bar": f"{TQ}:375-384 — goinfer ÷ llama.cpp by peer-claim's all-pairs bands; \"If any such cell turns BEHIND in S1, "
            "int4 stays the default\"",
     "verdict_q": f"{TQ}:405-406 — \"every cell that is level or ahead under int4 turns BEHIND under q4k … **int4 stays the "
                  "CUDA default for Q4_K GGUFs; q4k stays opt-in.**\"",
     "recorded": "int4 STAYS", "combine": "q4k", "names": {},
     "subs": [peer(f"{s} {m}@{d}", peer_sub(f"q4k-peer-2026-09-26/{f}", m, "cuda", d, "greedy", "llamacpp"), lab, st_)
              for s, f, labs in (("S2", "s2-int4.json", [("AHEAD", 1.108), ("AHEAD", 1.042), ("LEVEL", 1.019),
                                                          ("AHEAD", 1.040), ("LEVEL", 1.003), ("LEVEL", 0.985)]),
                                  ("S1", "s1-q4k.json", [("BEHIND", 0.851), ("BEHIND", 0.814), ("BEHIND", 0.829),
                                                         ("BEHIND", 0.928), ("BEHIND", 0.901), ("BEHIND", 0.887)]))
              for (m, d), (lab, st_) in zip([(m, d) for m in ("1.5B", "7B") for d in (128, 2048, 3900)], labs)]},
]

_NG = lambda key: False
PEER_FAMILIES = [
    ("peer-claim-a", "nobara CUDA", "CUDA greedy 0.5B/1.5B/7B × 128/2048/3900/8000",
     [("peer-claim-2026-09-25/a-e-dense.json", "peer-claim-2026-09-25/grade.jsonl", "cuda", _NG),
      ("peer-claim-2026-09-25/a-depth8000.json", "peer-claim-2026-09-25/grade.jsonl", "cuda", lambda k: k[2] == 128)],
     f"{PC}:173"),
    ("peer-claim-b", "nobara CUDA", "controls gemma3-1b / phi3-mini × 128/3900",
     [("peer-claim-2026-09-25/b-controls.json", "peer-claim-2026-09-25/grade.jsonl", "cuda", _NG)], f"{PC}:174"),
    ("peer-claim-c", "nobara CUDA", "Gemma 4 26B-A4B (architecture comparison)",
     [("peer-claim-2026-09-25/c-26b.json", "peer-claim-2026-09-25/grade.jsonl", "cuda", _NG)], f"{PC}:175"),
    ("peer-claim-e", "nobara CPU (amd64)", "CPU 0.5B/1.5B/7B @128",
     [("peer-claim-2026-09-25/a-e-dense.json", "peer-claim-2026-09-25/grade.jsonl", "cpu", _NG)],
     f"{PC}:177"),
    ("peer-claim-f", "nobara CUDA", "sampled @128, 0.5B / phi3-mini",
     [("peer-claim-2026-09-25/f-sampled.json", "peer-claim-2026-09-25/grade.jsonl", "cuda", lambda k: k[3] == "greedy")],
     f"{PC}:178"),
    ("peer-claim-g", "MacBook Metal", "Metal 0.5B/1.5B/7B × 128/2048/3900",
     [("peer-claim-2026-09-25-mac/g-metal-decode.json", "peer-claim-2026-09-25-mac/grade.jsonl", "metal", _NG)], f"{PC}:179"),
    ("peer-claim-i", "MacBook CPU (arm64)", "CPU arm64 0.5B/1.5B @128",
     [("peer-claim-2026-09-25-mac/i-cpu-decode.json", "peer-claim-2026-09-25-mac/grade.jsonl", "cpu", _NG)], f"{PC}:181"),
]


def grader_rows(path, jsonl):
    """Rows of a sectioned grade.jsonl that belong to `path` (sections are '## <basename>')."""
    rows, cur = [], None
    for line in open(os.path.join(M, jsonl)):
        line = line.strip()
        if line.startswith("##"):
            cur = line[2:].strip()
        elif line.startswith("{") and cur == os.path.basename(path):
            rows.append(json.loads(line))
    return rows


def peer_family_subs(files_rows):
    subs = []
    for path, jsonl, be_default, ungraded in files_rows:
        for r in grader_rows(path, jsonl):
            if "K" in r:
                continue
            be = r.get("backend", be_default)
            cfg = r.get("config", "greedy")
            key = (be, r["model"], r["depth"], cfg)
            if be != be_default:
                continue  # a-e-dense holds both families a (cuda) and e (cpu)
            graded = r["outcome"] != "VOID" and not ungraded(key)
            sid = f"{be} {r['model']}@{r['depth']}{'' if cfg == 'greedy' else ' ' + cfg} vs {r['peer']}"
            subs.append(peer(sid, peer_sub(path, r["model"], be, r["depth"], cfg, r["peer"]), r["outcome"],
                             r["median"], graded=graded))
    return subs


for gid, mach, what, rows, src in PEER_FAMILIES:
    GATES.append({"id": gid, "class": "served-peer", "machine": mach, "data": sorted({r[0] for r in rows}),
                  "N": "3 runs × 8 per cell",
                  "bar": f"{PC}:24-42 — r = goinfer ÷ peer per run pair; AHEAD every pair > 1.03, LEVEL every pair in "
                         "[0.97, 1.03], BEHIND every pair < 0.97, AMBIGUOUS-HIGH/LOW otherwise; > 5% run spread caps at "
                         "AMBIGUOUS; pre-registered at :8 (\"Committed and pushed before any timed run\")",
                  "verdict_q": f"{src} (cell {gid[-1]}: {what}); per-comparison labels from the committed grader output",
                  "recorded": None, "combine": "tuple", "names": {}, "subs": peer_family_subs(rows)})

INPROC = [
    ("R06-gate3", "inproc-ABBA", "nobara CPU (amd64)", f"{L1}/r06-gate3-ab.log", "5 ABBA pairs",
     f"{TL1}:77-78 — \"The flip stands unless any size's paired median speed-up … reads < 0.98\"",
     f"{TL1}:102 — \"No size's median is below 0.98, so the flip stands.\"", "PASS",
     [("1.5B", 0.98, 0.98, "PASS", 1.030, True), ("0.5B", 0.98, 0.98, "PASS", 1.016, True), ("7B", 0.98, 0.98, "PASS", 1.018, True)]),
    ("L1-S0-gate2", "inproc-ABBA", "nobara CPU (amd64)", f"{L1}/s0-gate2-ab.log", "5 ABBA pairs",
     f"{TL1}:187-190 — \"BUILD … the paired median is ≥ 1.04× on both the 1.5B and the 7B. PARK: < 1.02× on either … "
     "Between: the owner decides. The 0.5B is reported, not gated.\"",
     f"{TL1}:214 — \"**Verdict: BUILD** (both gated sizes ≥ 1.04×)\"", "PASS",
     [("1.5B", 1.04, 1.02, "PASS", 1.059, True), ("0.5B", 1.04, 1.02, "reported", 0.957, False), ("7B", 1.04, 1.02, "PASS", 1.084, True)]),
    ("L1-S0b-gate2", "inproc-ABBA", "nobara CPU (amd64)", f"{L1}/s0b-gate2-ab.log", "5 ABBA pairs",
     "as L1-S0-gate2 (S0's bands, re-run on the row kernel)",
     f"{TL1}:236,246 — \"**The S0 gates re-run on it**\" … medians 1.059× / 1.090× … \"The row kernel is the one the build "
     "uses\" (BUILD by S0's bands; the word is not restated)", "PASS",
     [("1.5B", 1.04, 1.02, "PASS", 1.059, True), ("0.5B", 1.04, 1.02, "reported", 0.968, False), ("7B", 1.04, 1.02, "PASS", 1.090, True)]),
]
for gid, cls, mach, log, n, bar, vq, rec, subs in INPROC:
    GATES.append({"id": gid, "class": cls, "machine": mach, "data": [log], "N": n, "bar": bar, "verdict_q": vq,
                  "recorded": rec, "names": {"PASS": "PASS/BUILD", "AMBIG": "owner", "FAIL": "PARK/FAIL"},
                  "subs": [ab(lab, abba_sub(log, lab), S, K, r, st_, graded=g) for lab, S, K, r, st_, g in subs]})

BW = 1.057 / 0.970  # q4k ÷ int4 weight bytes per token for the 1.5B, the records' effective-bandwidth factor
LEVERS = [
    ("q4k-lever1", "q4k-lever1-2026-09-26/e2e.txt", ("1.5B-q4k-new", "1.5B-int4"), ("7B-q4k-new", "7B-q4k-old"), None,
     f"{TQ}:438-443 — \"## Lever 1 result (2026-09-26): FAIL\"; 1.5B 0.811 FAIL, 7B guard 0.926 FAIL",
     "FAIL", ("FAIL", 0.811), ("FAIL", 0.926)),
    ("q4k-lever3-run1", "q4k-lever3-2026-09-26/run1-ungated/e2e.txt", ("1.5B-q4k-fused", "1.5B-int4"),
     ("7B-q4k-fused", "7B-q4k-unfused"), ("phi3-q4k-fused", "phi3-q4k-unfused"),
     f"{TQ}:506-509 — \"1.5B q4k effective bandwidth ÷ int4 **0.914** … **PASS**; **7B guard 0.951** … **FAIL**\" "
     "(so the change is reworked)", "FAIL", ("PASS", 0.914), ("FAIL", 0.951)),
    ("q4k-lever3-run2", "q4k-lever3-2026-09-26/run2-sizegated/e2e.txt", ("1.5B-q4k-fused", "1.5B-int4"),
     ("7B-q4k-fused", "7B-q4k-unfused"), ("phi3-q4k-fused", "phi3-q4k-unfused"),
     f"{TQ}:520-525 — Run 2, size-gated: 1.5B **0.911** PASS, 7B guard **1.001** PASS", "PASS", ("PASS", 0.911), ("PASS", 1.001)),
]
for gid, e2e, bar15, guard, info, vq, rec, r15, r7 in LEVERS:
    subs = [ab("1.5B bar", rounds_sub(e2e, *bar15, scale=BW), 0.90, 0.85, r15[0], r15[1]),
            ab("7B guard", rounds_sub(e2e, *guard), 0.98, 0.98, r7[0], r7[1])]
    if info:
        subs.append(ab("phi3 info", rounds_sub(e2e, *info), 1.0, 1.0, "reported", None, graded=False))
    GATES.append({"id": gid, "class": "inproc-rounds", "machine": "nobara CUDA", "data": [e2e],
                  "N": "6 paired rounds (alternating order, fresh process per arm)",
                  "bar": f"{TQ}:429-436 (lever 1) / :496-498 (lever 3) — 1.5B q4k effective bandwidth ÷ int4 \"≥ 0.90 PASS; "
                         "[0.85, 0.90) AMBIGUOUS; < 0.85 FAIL\"; guard: 7B \"≥ 0.98×\" or the change is reworked",
                  "verdict_q": vq, "recorded": rec, "names": {}, "subs": subs})

SANITY = {"id": "L1-gate6 (sanity, no ship bar)", "class": "served-AB", "machine": "MacBook CPU (arm64)",
          "data": [f"{L1}/arm64-speed-served.json (pass 1: goinfer = new)",
                   f"{L1}/arm64-speed-served-pass2.json (pass 2: labels swapped, goinfer_old = new)"],
          "N": "3 runs × 8 × 2 order-reversed passes",
          "bar": f"{L1}/run-arm64-speed-pass2.sh:17-23 — geometric mean of the passes; \"UNRESOLVED BY DRIFT\" if they "
                 "disagree on direction (±0.03); \"the NEON-widen … follow-up is flagged when the combined ratio is < 1.00 "
                 "and both passes are below 1.00\"; \"Reported, not a ship band\"",
          "verdict_q": f"{TL1}:416 — \"**By the rule, every model is slower in both passes, so the NEON-widen follow-up is "
                       "flagged.**\" (0.544× / 0.554× / 0.435×)",
          "recorded": None, "combine": "tuple", "names": {},
          "subs": [dict(ab(m, ab_sub([(f"{L1}/arm64-speed-served.json", "goinfer", "goinfer_old"),
                                      (f"{L1}/arm64-speed-served-pass2.json", "goinfer_old", "goinfer")], m, "cpu",
                                     stat="mean_field"), 1.0, 1.0, "FLAGGED", r),
                        pass_label="NOT SLOWER", fail_label="FLAGGED", cap_fn=gate6_cap)
                   for m, r in (("0.5B", 0.544), ("1.5B", 0.554), ("7B", 0.435))]}


# ----------------------------------------------------------------------------------------------------------------
# Replay machinery
# ----------------------------------------------------------------------------------------------------------------
_DATA = {}


def data_of(sub):
    k = id(sub)
    if k not in _DATA:
        _DATA[k] = sub["load"]()
    return _DATA[k]


def delta_for(g, override=None):
    if override is not None and g["class"].startswith("served"):
        return override
    return DELTA_SERVED if g["class"].startswith("served") else DELTA_INPROC


def cap_label(sub, d):
    if sub.get("cap_fn"):
        return sub["cap_fn"](d)
    if sub.get("peer"):
        return d["cap_label"]
    return bands(d["stat"], sub["S"], sub["K"])


def gate_verdict(g, res):
    graded = [(s["id"], r["label"]) for s, r in zip(g["subs"], res) if s["graded"]]
    if g.get("combine") == "q4k":
        return combine_q4k(dict(graded))
    if g.get("combine") == "tuple":
        return combine_tuple([v for _, v in graded])
    return combine_default([v for _, v in graded])


def recorded_verdict(g):
    if g.get("recorded") is not None:
        return g["recorded"]
    return combine_tuple([s["recorded"] for s in g["subs"] if s["graded"]])


def nblocks(g):
    ns = {len(data_of(s)["blocks"]) for s in g["subs"]}
    if len(ns) != 1:
        raise ValueError(f"{g['id']}: sub-gates differ in block count {ns}")
    return ns.pop()


def replay_gate(g, order=None, delta=None, posthoc=False):
    """Replay every sub-gate of g in one block order. Returns (per-sub results, gate verdict, accounting).
    posthoc=True is the variant of §6 (designed after v1's replay): on served gates no stop before block 2, and on
    peer gates the record's own > 5% run-spread cap, over the runs seen so far, blocks a stop."""
    dl = delta_for(g, delta)
    n = nblocks(g)
    order = list(range(n)) if order is None else order
    res = []
    used_all, rec_all, used_gr, rec_gr = {}, {}, {}, {}
    for s in g["subs"]:
        d = data_of(s)
        blocks = [d["blocks"][i] for i in order]
        if s["graded"]:
            mb, veto = 1, None
            if posthoc and g["class"].startswith("served"):
                mb = 2
                if s.get("peer"):
                    rp = [d["runpairs"][i] for i in order]
                    spr = lambda xs: (max(xs) - min(xs)) / statistics.mean(xs)
                    veto = (lambda k, rp=rp: k >= 2 and (spr([a for a, _ in rp[:k]]) > 0.05 or spr([b for _, b in rp[:k]]) > 0.05))
            v, used, trace = sequential(blocks, s["S"], s["K"], dl, mb, veto)
            label = s["pass_label"] if v == "PASS" else s["fail_label"] if v == "FAIL" else cap_label(s, d)
            how = "early" if v else "cap"
        else:
            used, trace, how = len(blocks), [], "reported"
            label = s["recorded"]
        res.append({"label": label, "used": used, "how": how, "trace": trace, "cap": cap_label(s, d)})
        consumed = order[:used]
        for tgt_u, tgt_r, ok in ((used_all, rec_all, True), (used_gr, rec_gr, s["graded"])):
            if not ok:
                continue
            cu, cr = {}, {}
            for i in range(n):
                for cid in d["cells"][i]:
                    cr[cid] = cr.get(cid, 0) + 1
                    if i in consumed:
                        cu[cid] = cu.get(cid, 0) + 1
            for cid in cr:
                tgt_r[cid] = max(tgt_r.get(cid, 0), cr[cid])
                tgt_u[cid] = max(tgt_u.get(cid, 0), cu.get(cid, 0))
    acct = {"used": sum(used_all.values()), "rec": sum(rec_all.values()),
            "used_g": sum(used_gr.values()), "rec_g": sum(rec_gr.values())}
    return res, gate_verdict(g, res), acct


def orders_for(n):
    """Every permutation of n blocks if n! <= PERM_MAX, otherwise PERM_MAX uniform random ones (seeded)."""
    if math.factorial(n) <= PERM_MAX:
        return [list(p) for p in itertools.permutations(range(n))]
    rng = random.Random(PERM_SEED)
    out = []
    for _ in range(PERM_MAX):
        p = list(range(n))
        rng.shuffle(p)
        out.append(p)
    return out


def pct(a, b):
    return 100.0 * a / b if b else float("nan")


def fmt_ratio(x):
    return "—" if x is None else f"{x:.3f}"


# ----------------------------------------------------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------------------------------------------------
def main():
    print("# TE4 replay (TE4-SEQ-v1), te4_replay.py\n")
    self_check()
    print(f"parameters: alpha {ALPHA} two-sided, delta served {DELTA_SERVED}, delta in-process {DELTA_INPROC}, "
          f"looks at n >= {MIN_N}, permutations: all if N! <= {PERM_MAX} else {PERM_MAX} (seed {PERM_SEED})\n")

    # --- inventory
    print("## Gate inventory (eligible: pre-registered numeric bar + per-run data)\n")
    print("| # | gate | class | machine | data | N recorded | graded sub-gates | recorded verdict |")
    print("|---|---|---|---|---|---|---|---|")
    for i, g in enumerate(GATES, 1):
        ng = sum(1 for s in g["subs"] if s["graded"])
        print(f"| {i} | {g['id']} | {g['class']} | {g['machine']} | {'; '.join(g['data'])} | {g['N']} | {ng} of "
              f"{len(g['subs'])} | {recorded_verdict(g)} |")
    print("\n### Bars and recorded verdicts (quoted)\n")
    for g in GATES + [SANITY]:
        print(f"- **{g['id']}.** Bar: {g['bar']}. Verdict: {g['verdict_q']}.")

    # --- per sub-gate, recorded order
    print("\n## Per sub-gate, recorded order\n")
    print("| gate | sub-gate | recorded | recomputed at cap (stat) | replay | blocks used / recorded | how | interval at the "
          "deciding look (ratio scale) | flip |")
    print("|---|---|---|---|---|---|---|---|---|")
    sub_rows, gate_rows = [], []
    for g in GATES + [SANITY]:
        res, gv, acct = replay_gate(g)
        gate_rows.append((g, res, gv, acct))
        for s, r in zip(g["subs"], res):
            d = data_of(s)
            last = r["trace"][-1] if r["trace"] else None
            iv = f"[{math.exp(last['lo']):.3f}, {math.exp(last['hi']):.3f}] @ block {last['block']} (n={last['n']})" if last else "—"
            chk = "" if (not s["graded"] or r["cap"] == s["recorded"]) else " **≠ record**"
            fk = flip_kind(s["recorded"], r["label"]) if s["graded"] else "(reported)"
            print(f"| {g['id']} | {s['id']} | {s['recorded']} ({fmt_ratio(s['rec_stat'])}) | {r['cap']} "
                  f"({d['stat']:.3f}){chk} | {r['label']} | {r['used']} / {len(d['blocks'])} | {r['how']} | {iv} | {fk} |")
            if g is not SANITY:
                sub_rows.append((g, s, r))

    # --- per gate
    print("\n## Per gate, recorded order\n")
    print("| gate | class | recorded verdict | replay verdict | agree | cell-runs used / recorded | saved | graded-only "
          "saved | disqualifying flip |")
    print("|---|---|---|---|---|---|---|---|---|")
    for g, res, gv, acct in gate_rows:
        rv = recorded_verdict(g)
        dq = any(flip_kind(s["recorded"], r["label"]).startswith("DISQ") for s, r in zip(g["subs"], res) if s["graded"])
        dq = dq or flip_kind(rv, gv).startswith("DISQ")
        print(f"| {g['id']} | {g['class']} | {rv} | {gv} | {'yes' if gv == rv else '**NO**'} | {acct['used']} / "
              f"{acct['rec']} | {pct(acct['rec'] - acct['used'], acct['rec']):.0f}% | "
              f"{pct(acct['rec_g'] - acct['used_g'], acct['rec_g']):.0f}% | {'**YES**' if dq else 'no'} |")

    # --- summary
    head = [(g, res, gv, acct) for g, res, gv, acct in gate_rows if g is not SANITY]

    def summarize(rows, label):
        n = len(rows)
        agree = sum(1 for g, res, gv, acct in rows if gv == recorded_verdict(g))
        subs = [(s, r) for g, res, gv, acct in rows for s, r in zip(g["subs"], res) if s["graded"]]
        sagree = sum(1 for s, r in subs if r["label"] == s["recorded"])
        dq = [(g["id"], s["id"], s["recorded"], r["label"]) for g, res, gv, acct in rows for s, r in zip(g["subs"], res)
              if s["graded"] and flip_kind(s["recorded"], r["label"]).startswith("DISQ")]
        dq += [(g["id"], "(gate)", recorded_verdict(g), gv) for g, res, gv, acct in rows
               if flip_kind(recorded_verdict(g), gv).startswith("DISQ")]
        other = [(g["id"], s["id"], s["recorded"], r["label"], flip_kind(s["recorded"], r["label"]))
                 for g, res, gv, acct in rows for s, r in zip(g["subs"], res)
                 if s["graded"] and r["label"] != s["recorded"] and not flip_kind(s["recorded"], r["label"]).startswith("DISQ")]
        u = sum(a["used"] for *_, a in rows)
        rr = sum(a["rec"] for *_, a in rows)
        ug = sum(a["used_g"] for *_, a in rows)
        rg = sum(a["rec_g"] for *_, a in rows)
        early = sum(1 for s, r in subs if r["how"] == "early")
        print(f"| {label} | {n} | {agree} / {n} ({pct(agree, n):.1f}%) | {sagree} / {len(subs)} ({pct(sagree, len(subs)):.1f}%) "
              f"| {early} / {len(subs)} | {len(dq)} | {len(other)} | {u} / {rr} → **{pct(rr - u, rr):.1f}%** | "
              f"{pct(rg - ug, rg):.1f}% |")
        return {"n": n, "agree": agree, "dq": dq, "other": other, "saved": pct(rr - u, rr), "saved_g": pct(rg - ug, rg),
                "sagree": sagree, "nsubs": len(subs)}

    print("\n## Summary, recorded order (the sanity case is excluded)\n")
    hdr = ("| set | gates | gate agreement | sub-gate agreement | sub-gates stopped early | disqualifying flips | other "
           "disagreements | cell-runs used / recorded → saved | saved, graded cells only |")
    print(hdr)
    print("|---|---|---|---|---|---|---|---|---|")
    tot = summarize(head, "**all eligible**")
    for cls in ("served-AB", "served-peer", "inproc-ABBA", "inproc-rounds"):
        summarize([x for x in head if x[0]["class"] == cls], cls)
    summarize([x for x in head if x[0]["class"].startswith("served")], "served (both)")
    print("\nDisqualifying flips (recorded non-pass → replay pass):", tot["dq"] or "none")
    print("Other disagreements:", tot["other"] or "none")

    # --- permutations
    def perm_table(kw, title, with_sanity=False):
        print(title)
        print("| gate | orders | agree with record | orders with a disqualifying flip | cell-runs saved: mean (min–max) |")
        print("|---|---|---|---|---|")
        rows, details = [], []
        for g in GATES + ([SANITY] if with_sanity else []):
            orders = orders_for(nblocks(g))
            ag, dqn, sv, u_sum, r_sum = 0, 0, [], 0, 0
            for o in orders:
                res, gv, acct = replay_gate(g, o, **kw)
                rv = recorded_verdict(g)
                ag += gv == rv
                bad = [(s["id"], s["recorded"], r["label"], r["used"]) for s, r in zip(g["subs"], res)
                       if s["graded"] and r["label"] != s["recorded"]]
                if any(flip_kind(a, b).startswith("DISQ") for _, a, b, _ in bad) or flip_kind(rv, gv).startswith("DISQ"):
                    dqn += 1
                if bad:
                    details.append((g["id"], [x + 1 for x in o], bad))
                sv.append(pct(acct["rec"] - acct["used"], acct["rec"]))
                u_sum += acct["used"]
                r_sum += acct["rec"]
            print(f"| {g['id']} | {len(orders)} | {ag} ({pct(ag, len(orders)):.0f}%) | {dqn} | "
                  f"{statistics.mean(sv):.0f}% ({min(sv):.0f}–{max(sv):.0f}%) |")
            if g is not SANITY:
                rows.append((pct(ag, len(orders)), dqn, u_sum / len(orders), r_sum / len(orders)))
        eu, er = sum(r[2] for r in rows), sum(r[3] for r in rows)
        print(f"\nEligible gates (sanity row excluded): mean per-gate agreement over orders {statistics.mean(r[0] for r in rows):.1f}%; gates with any "
              f"disqualifying order: {sum(1 for r in rows if r[1])}; orders with a disqualifying flip, summed: "
              f"{sum(r[1] for r in rows)}; expected pooled savings over a random order: {pct(er - eu, er):.1f}%.")
        if details:
            print("\nOrders that disagree (block order as recorded run numbers; sub-gate: recorded → replay @ blocks used):")
            for gid, o, bad in details:
                print(f"- {gid} order {o}: " + "; ".join(f"{sid}: {a} → {b} @ {u} ({flip_kind(a, b)})" for sid, a, b, u in bad))
        print()

    perm_table({}, "\n## Order sensitivity: permutations of each gate's blocks (TE4-SEQ-v1)\n", with_sanity=True)

    # --- sensitivity arm V0 (delta = 0 on served)
    print("\n## Sensitivity arm V0 (pre-registered, not the verdict): δ = 0 on served gates\n")
    print("| gate | recorded | V0 replay | agree | cell-runs saved | disqualifying flip |")
    print("|---|---|---|---|---|---|")
    v0 = []
    for g in GATES:
        res, gv, acct = replay_gate(g, delta=0.0)
        rv = recorded_verdict(g)
        dq = any(flip_kind(s["recorded"], r["label"]).startswith("DISQ") for s, r in zip(g["subs"], res) if s["graded"]) \
            or flip_kind(rv, gv).startswith("DISQ")
        v0.append((g, res, gv, acct))
        print(f"| {g['id']} | {rv} | {gv} | {'yes' if gv == rv else '**NO**'} | "
              f"{pct(acct['rec'] - acct['used'], acct['rec']):.0f}% | {'**YES**' if dq else 'no'} |")
    print()
    print(hdr)
    print("|---|---|---|---|---|---|---|---|---|")
    v0s = summarize(v0, "V0, all eligible")
    print("V0 disqualifying flips:", v0s["dq"] or "none")
    print("V0 other disagreements:", v0s["other"] or "none")
    print()
    perm_table({"delta": 0.0}, "### V0 over permuted orders (reported; added after v1's replay, no rule change)\n")

    # --- post-hoc variant (designed after seeing v1's replay; NOT a validation)
    print("## Post-hoc variant v1+ (designed after v1's replay on these same gates: NOT a validation)\n")
    print("v1 plus: served gates never stop before block 2; peer gates may not stop while either arm's run spread over "
          "the runs seen so far exceeds 5% (the record's own cap).\n")
    print("| gate | recorded | v1+ replay | agree | cell-runs saved | disqualifying flip |")
    print("|---|---|---|---|---|---|")
    ph = []
    for g in GATES:
        res, gv, acct = replay_gate(g, posthoc=True)
        rv = recorded_verdict(g)
        dq = any(flip_kind(s["recorded"], r["label"]).startswith("DISQ") for s, r in zip(g["subs"], res) if s["graded"]) \
            or flip_kind(rv, gv).startswith("DISQ")
        ph.append((g, res, gv, acct))
        print(f"| {g['id']} | {rv} | {gv} | {'yes' if gv == rv else '**NO**'} | "
              f"{pct(acct['rec'] - acct['used'], acct['rec']):.0f}% | {'**YES**' if dq else 'no'} |")
    print()
    print(hdr)
    print("|---|---|---|---|---|---|---|---|---|")
    phs = summarize(ph, "v1+, all eligible")
    print("v1+ disqualifying flips:", phs["dq"] or "none")
    print("v1+ other disagreements:", phs["other"] or "none")
    print()
    perm_table({"posthoc": True}, "### v1+ over permuted orders\n")

    # --- verdict
    print("\n## Verdict against TE4's acceptance criteria (TE4-SEQ-v1, recorded order)\n")
    n = tot["n"]
    agree_ok = pct(tot["agree"], n) >= 95.0
    flips_ok = not tot["dq"]
    save_ok = tot["saved"] >= 25.0
    print(f"- agreement {tot['agree']} / {n} gates = {pct(tot['agree'], n):.1f}% (needs ≥ 95%): {'MET' if agree_ok else 'NOT MET'}")
    print(f"- disqualifying flips: {len(tot['dq'])} (needs 0): {'MET' if flips_ok else 'NOT MET'}")
    print(f"- runs saved: {tot['saved']:.1f}% (needs ≥ 25%): {'MET' if save_ok else 'NOT MET'}")
    killed = (not agree_ok) or (not flips_ok) or (not save_ok)
    print(f"- kill line (disagree > 5%, any FAIL flip, or saves < 25%): {'HIT' if killed else 'not hit'}")
    if n < 20:
        print(f"- only {n} eligible gates (< 20): one disagreement is {100.0 / n:.1f}% > 5%, so the agreement criterion "
              "reads as zero disagreements, and the verdict is provisional on the gate count.")


if __name__ == "__main__":
    main()
