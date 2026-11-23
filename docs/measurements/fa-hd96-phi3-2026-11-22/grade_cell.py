#!/usr/bin/env python3
"""Grade gate 3's served cell for the hd-96 flash-decode lane (README beside this file) from one output directory; OP1-o2's grader with
the model and the decision depth changed.

  grade_cell.py <out dir>     # decode-cuda-*.json (bench_peer.py): engines goinfer (the hd-96 lane) and goinfer_old (the exact path)

Per depth, per block: goinfer's mean rate over its runs divided by goinfer_old's. The statistic is the median of the blocks' ratios.
The registered rule: SHIP when it is >= 1.02 at depth 3,900 and >= 0.98 at depth 128; REGRESSION when it is under 0.98 at depth 3,900;
PARKED otherwise. A block with an errored or missing arm is VOID, and so is a depth with fewer than 2 good blocks. Exit 0 on SHIP.
"""
import glob, json, os, statistics, sys

SHIP = {3900: 1.02, 128: 0.98}
REGRESSION_AT_3900 = 0.98


def main():
    out = sys.argv[1]
    cells, void = [], []
    for path in sorted(glob.glob(os.path.join(out, "decode-cuda-*.json"))):
        cells += [r for r in json.load(open(path)) if r.get("phase") in ("A", "B") and r.get("model") == "phi3-mini"]
    med = {}
    for depth in sorted({r["depth"] for r in cells}):
        rate = {}
        for r in cells:
            if r["depth"] != depth:
                continue
            if r.get("error") or not r.get("runs"):
                void.append(f"depth {depth}: {r['engine']} block {r.get('block')} errored: {r.get('error')}")
                continue
            rate[(r["engine"], r.get("block"))] = statistics.mean(r["runs"])
        ratios = []
        for b in sorted({b for _, b in rate}, key=str):
            if ("goinfer", b) not in rate or ("goinfer_old", b) not in rate:
                void.append(f"depth {depth}: block {b} is missing an arm")
                continue
            ratios.append(rate[("goinfer", b)] / rate[("goinfer_old", b)])
        if len(ratios) < 2:
            void.append(f"depth {depth}: {len(ratios)} good block(s), the registration asks for 2")
            continue
        med[depth] = statistics.median(ratios)
        lane = statistics.mean(v for (e, _), v in rate.items() if e == "goinfer")
        exact = statistics.mean(v for (e, _), v in rate.items() if e == "goinfer_old")
        print(f"depth {depth}: lane/exact median {med[depth]:.4f} over {len(ratios)} blocks ({lane:.2f} vs {exact:.2f} tok/s); by block "
              + " ".join(f"{r:.4f}" for r in ratios))
    for v in void:
        print("VOID: " + v)
    if void or set(med) != set(SHIP):
        print("VERDICT VOID")
        return 2
    if med[3900] < REGRESSION_AT_3900:
        print(f"VERDICT REGRESSION (depth 3900 median {med[3900]:.4f} < {REGRESSION_AT_3900})")
        return 1
    if med[3900] >= SHIP[3900] and med[128] >= SHIP[128]:
        print(f"VERDICT SHIP (>= {SHIP[3900]} at 3900, >= {SHIP[128]} at 128): the evidence for Francis's default decision, not the decision")
        return 0
    print("VERDICT PARKED (between the registered lines)")
    return 1


if __name__ == "__main__":
    sys.exit(main())
