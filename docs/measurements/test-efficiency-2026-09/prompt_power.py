#!/usr/bin/env python3
"""prompt_power.py — how many prompts each fidelity-gate criterion needs, from the gates' own per-prompt log lines (TE12).

Read-only. Parses every per-prompt line the prefill/decode fidelity gates print, e.g.

  [ref-gate] S set "a" K=256 prompt  3/10 exact(agree=85.9% HF=0/64 KL=0.0513) fast(agree=93.8% HF=0/64 KL=0.0463) ...

groups them into cells (log, gate tag, model, prompt set, K, candidate arm), and reports, for cells with 10 prompts:

  1. prompts needed at 80% power (one-sided 5%) when the arms are truly equal, per criterion, at the gates' own margins:
       - KL:        mean(candidate KL) <= 1.10 x mean(exact KL). Tested as a paired mean, d_i = cand_i - 1.1 * exact_i,
                    because per-prompt RATIOS explode where the exact arm's KL is near zero (one logged prompt reads
                    0.0021 -> 0.0143, a 6.9x "ratio" on a 0.012 difference) — the gates themselves compare means.
       - agreement: mean(candidate) >= mean(exact) - 1.0 pt (a paired mean difference already).
  2. a replay of a conservative group-sequential rule on the KL criterion — looks after 4, 7 and 10 prompts, one-sided
     alpha 0.05/3 per look (Bonferroni) — in the recorded prompt order AND over 2,000 random prompt orders per cell,
     against the fixed-10 verdict. TE4's SEQ-v1 was killed on order sensitivity; this is the same test.
  3. the crit A/B/C flags the gates printed in their verdict lines, tallied.
  4. what n prompts can say about a worst-case ("no prompt regresses") criterion: 0 bad of n bounds the bad-prompt
     rate at 1 - 0.05^(1/n) (95%).

Caveats: each sd is estimated from 10 prompts (itself uncertain by roughly +/-25%); the power figures assume equal arms,
the case a non-inferiority gate must pass; positions enter only through the per-prompt aggregate, which is the right
unit for an interval (positions share their prompt).

usage: python3 docs/measurements/test-efficiency-2026-09/prompt_power.py [log dirs ...]
       (default: docs/measurements and ~/goinfer-logs, whichever exist)
"""
import collections
import glob
import math
import os
import random
import re
import statistics as st
import sys
from pathlib import Path
from statistics import NormalDist

REPO = Path(__file__).resolve().parents[3]
LINE = re.compile(
    r'^\[(?P<tag>[\w-]+)\]\s+(?P<model>\S+)(?:\s+set\s+"(?P<set>\w)")?\s+K=(?P<K>\d+)\s+prompt\s+(?P<i>\d+)/(?P<n>\d+)\s+'
    r'exact\(agree=(?P<ea>[\d.]+)%\s+HF=(?P<eh>\d+)/(?P<pos>\d+)\s+KL=(?P<ek>[\d.]+)\)\s+'
    r'(?P<arm>\w+)\(agree=(?P<fa>[\d.]+)%\s+HF=(?P<fh>\d+)/\d+\s+KL=(?P<fk>[\d.]+)\)', re.M)
CRIT_CUDA = re.compile(r'critA\([^)]*\)=(true|false) critB\([^)]*\)=(true|false) critC\([^)]*\)=(true|false)')
CRIT_METAL = re.compile(r'critA=(true|false) critB=(true|false) critC=(true|false)')
Z = NormalDist().inv_cdf(0.95) + NormalDist().inv_cdf(0.80)
LOOKS, ALPHA = (4, 7, 10), 0.05 / 3
ORDERS = 2000


def tq(p, df):
    """Student-t quantile, Cornish-Fisher expansion of the normal quantile (within ~1% for df >= 3)."""
    z = NormalDist().inv_cdf(p)
    return (z + (z ** 3 + z) / 4 / df + (5 * z ** 5 + 16 * z ** 3 + 3 * z) / 96 / df ** 2
            + (3 * z ** 7 + 19 * z ** 5 + 17 * z ** 3 - 15 * z) / 384 / df ** 3)


def pct(xs, f):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(f * len(xs)))]


def sequential(d):
    """Pass when the upper bound of mean(d) is below 0, fail when the lower bound is above 0, at looks 4/7/10."""
    for n in LOOKS:
        m, h = st.mean(d[:n]), tq(1 - ALPHA, n - 1) * st.stdev(d[:n]) / math.sqrt(n)
        if m + h < 0:
            return "pass", n
        if m - h > 0:
            return "fail", n
    return "open", LOOKS[-1]


def fixed(d):
    m, h = st.mean(d), tq(0.95, len(d) - 1) * st.stdev(d) / math.sqrt(len(d))
    return "pass" if m + h < 0 else ("fail" if m - h > 0 else "open")


def main():
    roots = sys.argv[1:] or [str(REPO / "docs" / "measurements")] + [
        d for d in (os.path.expanduser("~/goinfer-logs"),) if os.path.isdir(d)]
    cells, crit_cuda, crit_metal, nfiles = collections.defaultdict(dict), collections.Counter(), collections.Counter(), 0
    for root in roots:
        for p in glob.glob(os.path.join(root, "**", "*.log"), recursive=True):
            try:
                t = open(p, errors="replace").read()
            except OSError:
                continue
            hit = False
            for m in LINE.finditer(t):
                key = (os.path.relpath(p, os.path.dirname(root)), m["tag"], m["model"], m["set"] or "", int(m["K"]), m["arm"])
                cells[key][int(m["i"])] = m.groupdict()   # last occurrence of a prompt index wins
                hit = True
            for m in CRIT_CUDA.finditer(t):
                crit_cuda[m.groups()] += 1
            for m in CRIT_METAL.finditer(t):
                crit_metal[m.groups()] += 1
            nfiles += hit
    rows = []
    for key, byi in cells.items():
        seq = [byi[i] for i in sorted(byi)]
        if len(seq) < 10:
            continue
        e = [float(x["ek"]) for x in seq]
        d = [float(x["fk"]) - 1.1 * float(x["ek"]) for x in seq]
        da = [float(x["fa"]) - float(x["ea"]) for x in seq]
        rows.append(dict(key=key, d=d,
                         n_kl=math.ceil((Z * st.stdev(d) / (0.1 * st.mean(e))) ** 2) if st.mean(e) > 0 else 10 ** 6,
                         n_ag=math.ceil((Z * st.stdev(da) / 1.0) ** 2)))
    print(f"# fidelity-gate prompt budgets — {nfiles} logs, {len(rows)} cells with 10 prompts\n")
    nk, na = [r["n_kl"] for r in rows], [r["n_ag"] for r in rows]
    print("prompts needed at 80% power when the arms are truly equal (the gates' own margins):")
    print(f"  KL, mean(cand) <= 1.10 x mean(exact):  median {pct(nk, .5)}, p90 {pct(nk, .9)}, max {max(nk)}; "
          f"{sum(n > 10 for n in nk)} of {len(nk)} cells need more than 10")
    print(f"  agreement, >= exact - 1 pt:           median {pct(na, .5)}, p90 {pct(na, .9)}, max {max(na)}; "
          f"{sum(n > 10 for n in na)} of {len(na)} cells need more than 10")
    print(f"  worst-case ('no prompt regresses'): 0 bad of 10 bounds the bad-prompt rate at "
          f"{100 * (1 - 0.05 ** (1 / 10)):.0f}% (95%); 0 of 30 at {100 * (1 - 0.05 ** (1 / 30)):.0f}%\n")

    rec, used_rec = collections.Counter(), 0
    random.seed(20260928)
    tot = bad = false_pass = used_perm = 0
    touchy = []
    for r in rows:
        v = sequential(r["d"])
        rec[v] += 1
        used_rec += v[1]
        fx, nb = fixed(r["d"]), 0
        for _ in range(ORDERS):
            xs = r["d"][:]
            random.shuffle(xs)
            vv, n = sequential(xs)
            used_perm += n
            tot += 1
            if vv != "open" and vv != fx:
                nb += 1
                false_pass += vv == "pass"
        bad += nb
        if nb:
            touchy.append((r["key"], fx, nb / ORDERS))
    print("group-sequential rule on the KL criterion (looks after 4 / 7 / 10 prompts, one-sided alpha 0.05/3 each):")
    for (v, n), c in sorted(rec.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        print(f"  recorded order: {v:<5} at {n:>2} prompts: {c} cells")
    print(f"  recorded order: prompts used {used_rec} of {10 * len(rows)} ({100 * used_rec / (10 * len(rows)):.0f}%)")
    print(f"  {ORDERS} random orders per cell ({tot} in all): mean prompts {used_perm / tot:.2f}; the verdict differs from "
          f"the fixed-10 verdict in {bad} ({100 * bad / tot:.2f}%), {false_pass} of them a PASS the fixed-10 reading "
          f"does not give")
    for key, fx, frac in touchy:
        print(f"    {key[0][:60]} {key[2]} K={key[4]} (fixed-10 {fx}): {100 * frac:.1f}% of orders")
    print()
    print("verdict flags the gates printed (A = hard flips, B = agreement, C = KL):")
    for name, c in (("CUDA form: C is KL <= 1.1x exact", crit_cuda), ("Metal pooled form: C is KL <= exact, no margin", crit_metal)):
        fails = {k: v for k, v in c.items() if "false" in k}
        print(f"  {name}: {sum(c.values())} verdicts, {sum(fails.values())} not passing")
        for (a_, b_, c_), v in sorted(fails.items(), key=lambda kv: -kv[1]):
            print(f"    A={a_:<5} B={b_:<5} C={c_:<5} x{v}")


if __name__ == "__main__":
    main()
