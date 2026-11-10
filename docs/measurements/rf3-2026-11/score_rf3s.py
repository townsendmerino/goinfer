#!/usr/bin/env python3
"""Scores G-RF3-S. Usage: score_rf3s.py <out dir> [--aa]. Pairs in a block b: (A0,B1) and (A3,B2); a ratio is A/B for times
(TTFT, decoder gap max) so >1 means B is better. --aa scores the A/A cell instead: both pairs' TTFT ratios must be within 0.03
of 1.00 (exit 1 if not)."""
import json, statistics as st, sys, os

out, aa = sys.argv[1], "--aa" in sys.argv


def med(x):
    x = [v for v in x if v is not None]
    return st.median(x) if x else None


def metrics(r):
    inside = 0
    for t in r["decoder_times"]:
        inside += sum(1 for x in t if any(w[1] and w[0] <= x <= w[1] for w in r["newcomer_windows"]))
    return {"ttft": med(r["newcomer_ttft_s"]), "gap": r["decoder_gap_ms"]["max"], "inside": inside,
            "sha": (r["decoder_sha"], r["newcomer_sha"])}


def pairs(res, blocks):
    for b in blocks:
        yield metrics(res[f"{b}-A0"]), metrics(res[f"{b}-B1"])
        yield metrics(res[f"{b}-A3"]), metrics(res[f"{b}-B2"])


if aa:
    res = json.load(open(os.path.join(out, "15-stall-aa.json")))["results"]
    bad = False
    for a, b in pairs(res, ["aa0"]):
        r = a["ttft"] / b["ttft"]
        print(f"A/A pair TTFT ratio {r:.3f}")
        bad = bad or abs(r - 1) > 0.03
    sys.exit(1 if bad else 0)
fail = False
for cell in ["15-stall", "15-solo", "7-stall", "7-solo"]:
    f = os.path.join(out, cell + ".json")
    if not os.path.exists(f):
        print(cell, "missing"); fail = True; continue
    res = json.load(open(f))["results"]
    ps = list(pairs(res, ["b0", "b1", "b2", "b3"]))
    ttft = st.median(a["ttft"] / b["ttft"] for a, b in ps)
    gaps = [a["gap"] / b["gap"] for a, b in ps if a["gap"] and b["gap"]]
    gap = st.median(gaps) if gaps else None
    inside = st.median(b["inside"] / a["inside"] for a, b in ps if a["inside"])  if any(a["inside"] for a, _ in ps) else None
    same = all(a["sha"] == b["sha"] for a, b in ps)
    if cell.endswith("stall"):
        v = "PASS" if ttft >= 1.03 else ("AMBIGUOUS" if ttft >= 0.98 else "FAIL")
        gv = "PASS" if gap >= 0.95 else ("AMBIGUOUS" if gap >= 0.90 else "FAIL")
        print(f"{cell}: TTFT A/B {ttft:.3f} {v}; gap-max A/B {gap:.3f} {gv}; decoder tokens in windows B/A {inside}; shas identical {same}")
        fail = fail or v != "PASS" or gv != "PASS" or not same
    else:
        v = "PASS" if ttft >= 0.95 else "FAIL"
        print(f"{cell}: solo TTFT A/B {ttft:.3f} {v}; shas identical {same}")
        fail = fail or v != "PASS" or not same
print("VERDICT:", "RF3 ships on CUDA" if not fail else "NOT a clean pass: read per the registered outcomes (ambiguous = parked; a failed guard = does not ship)")
