#!/usr/bin/env python3
"""te1_analyze.py — grades TE1's night run (run-te1-aa-mutation.sh) by the rule pre-registered in
docs/tasks/task-test-efficiency-2026-09.md, TE1 "Pre-registration". Written, and committed, before the run exists.

  usage: te1_analyze.py [DIR]    (default: te1-2026-09-28 beside this script)

Reads aa-{load,instant}-{1,2}.json, mut-{instant,load}.{json,log} and timeline.txt. Prints each metric, then the verdict.
"""
import json
import math
import os
import re
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
D = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "te1-2026-09-28")
PASS_RATIO, KILL_RATIO = 1.25, 2.07   # spread ratio bands; 2.07 = sqrt(F(6,6) one-sided 5% point, 4.28)
SHARE_BAND = 0.10                     # instant gate: idle-gate share of served wall at night
HOLD_SLACK_S, RELEASE_S = 5.0, 10.0   # mutation: hold for the hog's life (-5 s), instant releases within 10 s


def cells(path):
    try:
        return [r for r in json.load(open(path)) if isinstance(r, dict) and "engine" in r]
    except (OSError, ValueError):
        return None


def timeline():
    spans = {}
    try:
        for line in open(os.path.join(D, "timeline.txt")):
            m = re.match(r"=== .* (\d{9,}) (START|END) (aa gate=(\w+) tag=(\d)|mutation gate=(\w+))", line)
            if m:
                key = ("aa", m[4], m[5]) if m[4] else ("mut", m[6], "")
                spans.setdefault(key, {})[m[2]] = int(m[1])
    except OSError:
        pass
    return spans


def main():
    spans = timeline()
    print(f"# TE1 night run — {D}\n")
    ok_all = True

    # 1. idle-gate share of served wall, and sweep wall
    print("## 1. idle-gate share and sweep wall\n")
    print("| sweep | cells | wall | gate wait | share |")
    print("|---|---:|---:|---:|---:|")
    share, wall = {"load": [], "instant": []}, {"load": [], "instant": []}
    for g in ("load", "instant"):
        for t in ("1", "2"):
            cs = cells(os.path.join(D, f"aa-{g}-{t}.json"))
            sp = spans.get(("aa", g, t), {})
            if not cs or "START" not in sp or "END" not in sp:
                print(f"| {g} {t} | MISSING | | | |")
                ok_all = False
                continue
            w = sp["END"] - sp["START"]
            gw = sum((c.get("machine", {}).get("gate") or {}).get("wait_s", 0) for c in cs)
            share[g].append(gw / w)
            wall[g].append(w)
            print(f"| {g} {t} | {len(cs)} | {w / 60:.1f} min | {gw / 60:.1f} min | {gw / w:.1%} |")
    if all(share.values()):
        s_i = statistics.mean(share["instant"])
        red = 1 - statistics.mean(wall["instant"]) / statistics.mean(wall["load"])
        print(f"\ninstant-gate share {s_i:.1%} (band <= {SHARE_BAND:.0%}): {'IN BAND' if s_i <= SHARE_BAND else 'OUT OF BAND'}")
        print(f"sweep wall, instant vs load: {red:+.1%} (band -30..-45% reduction; reported)\n")

    # 2. A/A spread
    print("## 2. A/A spread (same binary in both arms)\n")
    print("| gate | sweep | model | goinfer | goinfer_old | ln ratio |")
    print("|---|---|---|---:|---:|---:|")
    lr = {"load": [], "instant": []}
    for g in ("load", "instant"):
        for t in ("1", "2"):
            cs = cells(os.path.join(D, f"aa-{g}-{t}.json")) or []
            by = {(c["engine"], c["model"]): c for c in cs if c.get("runs")}
            for m in ("0.5B", "1.5B", "7B"):
                a, b = by.get(("goinfer", m)), by.get(("goinfer_old", m))
                if not a or not b:
                    continue
                r = math.log(statistics.mean(a["runs"]) / statistics.mean(b["runs"]))
                lr[g].append(r)
                print(f"| {g} | {t} | {m} | {statistics.mean(a['runs']):.2f} | {statistics.mean(b['runs']):.2f} | {r:+.4f} |")
    verdict_aa = "INCOMPLETE"
    if len(lr["load"]) == 6 and len(lr["instant"]) == 6:
        rms = {g: math.sqrt(sum(x * x for x in v) / len(v)) for g, v in lr.items()}
        ratio = rms["instant"] / rms["load"] if rms["load"] > 0 else float("inf")
        verdict_aa = "PASS" if ratio <= PASS_RATIO else ("KILL" if ratio > KILL_RATIO else "PARKED (owner; more pairs)")
        print(f"\nRMS ln ratio: load {rms['load']:.4f}, instant {rms['instant']:.4f}; ratio {ratio:.2f} "
              f"(pass <= {PASS_RATIO}, kill > {KILL_RATIO}): **{verdict_aa}**")
        print(f"mean ln ratio (a systematic arm offset would mean carry-over): load {statistics.mean(lr['load']):+.4f}, "
              f"instant {statistics.mean(lr['instant']):+.4f}\n")
    else:
        ok_all = False

    # 3. mutation
    print("## 3. mutation: a CPU hog on every core, 90 s, from the moment cell 1's record lands\n")
    mut_ok = {}
    for g in ("instant", "load"):
        cs = cells(os.path.join(D, f"mut-{g}.json")) or []
        log = open(os.path.join(D, f"mut-{g}.log")).read() if os.path.exists(os.path.join(D, f"mut-{g}.log")) else ""
        hs = re.search(r"=== ([\d.]+) HOG START", log)
        he = re.search(r"=== ([\d.]+) HOG STOP", log)
        if len(cs) < 2 or not hs or not he:
            print(f"- {g}: INCOMPLETE ({len(cs)} cells, hog start {bool(hs)}, stop {bool(he)})")
            mut_ok[g] = False
            continue
        need = float(he[1]) - float(hs[1])
        w = (cs[1].get("machine", {}).get("gate") or {}).get("wait_s", 0)
        held = w >= need - HOLD_SLACK_S
        released = w <= need + RELEASE_S
        mut_ok[g] = held and (released if g == "instant" else True)
        print(f"- {g}: hog ran {need:.1f} s; cell 2's gate waited {w:.1f} s -> held {'YES' if held else 'NO'}"
              + (f", released within {RELEASE_S:g} s {'YES' if released else 'NO'}" if g == "instant"
                 else f", release lag {w - need:+.1f} s (recorded)"))
    print()

    # 4. thermal
    th = []
    for f in sorted(os.listdir(D)) if os.path.isdir(D) else []:
        if f.endswith(".json"):
            for c in cells(os.path.join(D, f)) or []:
                t = c.get("machine", {}).get("therm")
                if t and "No thermal warning level has been recorded" not in t:
                    th.append((f, c["engine"], c["model"], t))
    print(f"## 4. thermal: {len(th)} cell(s) recorded a thermal or performance warning")
    for x in th[:10]:
        print(f"- {x}")

    # verdict
    mut_pass = all(mut_ok.get(g) for g in ("instant", "load"))
    share_ok = all(share.values()) and statistics.mean(share["instant"]) <= SHARE_BAND
    print("\n## verdict (pre-registered)\n")
    if not ok_all:
        print("INCOMPLETE — a sweep or its timeline is missing; nothing is decided.")
    elif not mut_pass:
        print("KILL — the instant gate failed the mutation (it must hold while the box is loaded, and release promptly).")
    elif verdict_aa == "KILL":
        print("KILL — A/A spread widened past the kill band: the long wait bought real recovery. Fallback per TE1.")
    elif verdict_aa.startswith("PARKED"):
        print("PARKED — A/A spread ratio between the pass and kill bands; to the owner, with more pairs.")
    else:
        print(f"PASS — mutation held and released; A/A spread within band; idle-gate share "
              f"{'in' if share_ok else 'OUT OF'} band. Make BENCH_IDLE_GATE=instant the default"
              f"{'' if share_ok else ' only after the share miss is explained'}.")


if __name__ == "__main__":
    main()
