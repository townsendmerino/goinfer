#!/usr/bin/env python3
"""D6a's pre-registered metrics and decision (docs/tasks/task-constrained-confidence.md, D6a pre-registration),
from `goinfer-chat decide` output run WITHOUT a calibration (raw, T = 1) on eval-sample.jsonl, plus the
calibration.json `decisions-calibrate` fitted on calib-sample.jsonl for the same arm.

A calibrated distribution is computed here from the raw one: decide's p_i = softmax(z)_i over the options, so
softmax(z / T)_i = p_i^(1/T) / sum_j p_j^(1/T) — the same thing decide would print with the calibration loaded.

Metrics (per kind, then combined with the OOD split's own kind proportions — noul 9,767 / choice 3,219 / score 72 of
13,058 — so the overall figures compare with JEV-9B's published full-split OOD numbers):
  top-1: argmax(p) == argmax(target)   (openjev_v2 targets are gold; the few non-one-hot choice rows use their argmax)
  ECE:   15 equal-width bins on max(p), |accuracy - mean max(p)| weighted by bin mass (top-label ECE)
  noul AUROC: P(true) against the gold label (reported)

Decision (arm A = Qwen3.5-9B chat-v1, calibrated), against JEV-9B's published OOD top-1 0.9181 / ECE 0.0396:
  top-1 >= 0.8881 and ECE <= sqrt(0.05^2 + floor^2)  -> Route A is enough: ship D1 + D5, park D2-D4
    (floor = ece_floor(): the ECE a perfectly calibrated model reads at this sample; owner amendment 2026-09-28)
  top-1 <  0.8181                  -> build D2-D4
  otherwise                        -> the owner's call, with these numbers
Control (arm B = Qwen3.5-9B bare-v1, raw): the authors' B0 (the same computation, bf16) read OOD top-1 0.5180. Arm B
within 0.05 of it -> goinfer's Route A reproduces theirs; outside -> flagged, and arm A is not trusted until explained.

usage: analyze.py <arm-name>=<eval-out.jsonl>:<calibration.json> ...
"""
import json
import random, math, sys

PROP = {"noul": 9767, "choice": 3219, "score": 72}
TOTAL = sum(PROP.values())
JEV9B_TOP1, JEV9B_ECE, B0_TOP1 = 0.9181, 0.0396, 0.5180


def argmax(xs):
    return max(range(len(xs)), key=lambda i: xs[i])


def temper(p, t):
    if t == 1:
        return p
    q = [max(x, 1e-300) ** (1 / t) for x in p]
    s = sum(q)
    return [x / s for x in q]


def auroc(pos, neg):
    if not pos or not neg:
        return None
    s = 0.0
    for a in pos:
        for b in neg:
            s += 1 if a > b else 0.5 if a == b else 0
    return s / (len(pos) * len(neg))


def metrics(rows, temps):
    """rows: decide output lines; temps: kind -> T (1 when uncalibrated)."""
    per = {}
    for k in PROP:
        rs = [r for r in rows if r["kind"] == k and r.get("distribution") and r.get("target")]
        if not rs:
            continue
        ps = [temper(r["distribution"], temps.get(k, 1)) for r in rs]
        acc = [argmax(p) == argmax(r["target"]) for p, r in zip(ps, rs)]
        conf = [max(p) for p in ps]
        per[k] = {"n": len(rs), "top1": sum(acc) / len(rs), "acc": acc, "conf": conf}
        if k == "noul":
            pos = [p[1] for p, r in zip(ps, rs) if argmax(r["target"]) == 1]
            neg = [p[1] for p, r in zip(ps, rs) if argmax(r["target"]) == 0]
            per[k]["auroc"] = auroc(pos, neg)
    # combined with the split's proportions: each row weighs PROP[k]/TOTAL/n_k
    bins = [[0.0, 0.0, 0.0] for _ in range(15)]  # weight, weighted acc, weighted conf
    top1 = 0.0
    wsum = 0.0
    for k, m in per.items():
        w = PROP[k] / TOTAL / m["n"]
        top1 += PROP[k] / TOTAL * m["top1"]
        wsum += PROP[k] / TOTAL
        for a, c in zip(m["acc"], m["conf"]):
            b = min(14, int(c * 15))
            bins[b][0] += w
            bins[b][1] += w * a
            bins[b][2] += w * c
    ece = sum(abs(b[1] - b[2]) for b in bins if b[0] > 0) / wsum
    for k, m in per.items():
        kb = [[0.0, 0.0, 0.0] for _ in range(15)]
        for a, c in zip(m["acc"], m["conf"]):
            b = min(14, int(c * 15))
            kb[b][0] += 1
            kb[b][1] += a
            kb[b][2] += c
        m["ece"] = sum(abs(b[1] - b[2]) for b in kb if b[0] > 0) / m["n"]
    return top1 / wsum, ece, per


def combined_ece(per, outcome):
    """The combined, kind-weighted top-label ECE of metrics(), with accuracies taken from outcome(k, i)."""
    bins = [[0.0, 0.0, 0.0] for _ in range(15)]
    wsum = 0.0
    for k, m in per.items():
        w = PROP[k] / TOTAL / m["n"]
        wsum += PROP[k] / TOTAL
        for i, c in enumerate(m["conf"]):
            b = min(14, int(c * 15))
            bins[b][0] += w
            bins[b][1] += w * outcome(k, i)
            bins[b][2] += w * c
    return sum(abs(b[1] - b[2]) for b in bins if b[0] > 0) / wsum


def ece_floor(per, draws=2000, seed=0):
    """The ECE a PERFECTLY CALIBRATED model would read at this sample: each row is correct with probability equal to
    its own confidence, scored with the same bins and kind weights, averaged over draws (fixed seed, so the bar is
    reproducible). Owner amendment 2026-09-28 (TE3's finding): at 400 / 400 / 72 rows this floor is ~0.02-0.05 against
    a 0.05 bar. The bar adds the floor in quadrature, sqrt(0.05^2 + floor^2): sampling noise and a true calibration gap
    combine roughly that way inside |accuracy - confidence|, so this bar sits at a true gap of ~0.05. Adding the floor
    linearly (0.05 + floor) was checked by simulation and rejected: it passed a model 5 points overconfident 37-39
    times in 40, which moves the bar to a true gap of ~0.07."""
    rng = random.Random(seed)
    tot = 0.0
    for _ in range(draws):
        draw = {k: [rng.random() < c for c in m["conf"]] for k, m in per.items()}
        tot += combined_ece(per, lambda k, i: draw[k][i])
    return tot / draws


def main(args):
    results = {}
    for a in args:
        name, rest = a.split("=", 1)
        out, cal = rest.split(":", 1)
        rows = [json.loads(l) for l in open(out)]
        failed = sum(1 for r in rows if r.get("error"))
        c = json.load(open(cal))
        temps = c.get("per_kind", {})
        raw = metrics(rows, {})
        calib = metrics(rows, temps)
        results[name] = (raw, calib)
        print(f"## {name}  ({len(rows)} rows, {failed} failed validation; template {rows[0].get('template')}; T {temps})")
        for label, (top1, ece, per) in (("raw", raw), ("calibrated", calib)):
            kinds = ", ".join(f"{k} top-1 {m['top1']:.4f} ece {m['ece']:.4f} (n {m['n']})" +
                              (f" auroc {m['auroc']:.4f}" if m.get("auroc") is not None else "") for k, m in per.items())
            print(f"  {label:10s} overall top-1 {top1:.4f}  ECE {ece:.4f}   | {kinds}")
        fits = c.get("fit", {})
        for k, f in fits.items():
            flag = "  (AT THE SEARCH BOUND: not a calibration)" if f["T"] >= 49.9 or f["T"] <= 0.0201 else ""
            print(f"  fit {k}: n {f['n']} T {f['T']:.4f} KL {f['kl_before']:.4f} -> {f['kl_after']:.4f}{flag}")
        print()
    if "A" in results:
        top1, ece, per_a = results["A"][1]
        floor = ece_floor(per_a)
        bar = math.hypot(0.05, floor)
        print(f"ECE floor (a perfectly calibrated model at this sample, arm A's own confidences, 2000 draws): {floor:.4f}; "
              f"bar sqrt(0.05^2 + floor^2) = {bar:.4f}")
        if top1 >= JEV9B_TOP1 - 0.03 and ece <= bar:
            verdict = "ROUTE A IS ENOUGH: ship D1 + D5, park D2-D4"
        elif top1 < JEV9B_TOP1 - 0.10:
            verdict = "BUILD D2-D4 (more than 10 points behind JEV-9B)"
        else:
            verdict = "IN BETWEEN: the owner's call"
        print(f"DECISION (arm A calibrated): top-1 {top1:.4f} vs JEV-9B {JEV9B_TOP1} (gap {JEV9B_TOP1 - top1:+.4f} points x100 = "
              f"{100 * (JEV9B_TOP1 - top1):.1f}), ECE {ece:.4f} vs bar {bar:.4f} (0.05 and the floor in quadrature) -> {verdict}")
    if "B" in results:
        top1 = results["B"][0][0]
        ok = abs(top1 - B0_TOP1) <= 0.05
        print(f"CONTROL (arm B raw): top-1 {top1:.4f} vs B0 {B0_TOP1} -> {'reproduces' if ok else 'DOES NOT REPRODUCE: arm A not trusted until explained'}")


if __name__ == "__main__":
    main(sys.argv[1:])
