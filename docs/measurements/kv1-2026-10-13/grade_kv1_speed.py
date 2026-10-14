#!/usr/bin/env python3
"""Grade KV1's G5b (docs/measurements/kv1-2026-10-13.md) from one box's output directory.

  grade_kv1_speed.py <out dir>     # decode-*.json (bench_peer.py, phase A) and ttft-*.json (kv1_ttft.py)

Per cell the statistic is the median over blocks of new / old, as a rate (for time to first token, old's time over
new's). PASS: the median is in [0.99, 1.01]. FAIL: the median is under 0.99 and every block is under 1. PARKED:
anything else outside the band. VOID: a cell errored, or a block is missing an arm. Exit 0 only when every cell passes.
"""
import glob, json, os, statistics, sys


def verdict(ratios):
    med = statistics.median(ratios)
    if 0.99 <= med <= 1.01:
        return "PASS", med
    if med < 0.99 and all(r < 1 for r in ratios):
        return "FAIL", med
    return "PARKED", med


def main():
    out, results, void = sys.argv[1], [], []
    for path in sorted(glob.glob(os.path.join(out, "decode-*.json"))):
        cells = [r for r in json.load(open(path)) if r.get("phase") == "A"]
        for mk in sorted({r["model"] for r in cells}):
            rate = {}
            for r in cells:
                if r["model"] != mk:
                    continue
                if r.get("error") or not r.get("runs"):
                    void.append(f"decode {mk}: {r['engine']} block {r.get('block')} errored: {r.get('error')}")
                    continue
                rate[(r["engine"], r.get("block"))] = statistics.mean(r["runs"])
            ratios = []
            for b in sorted({b for _, b in rate}):
                if ("goinfer", b) not in rate or ("goinfer_old", b) not in rate:
                    void.append(f"decode {mk}: block {b} is missing an arm")
                    continue
                ratios.append(rate[("goinfer", b)] / rate[("goinfer_old", b)])
            if ratios:
                new = statistics.median(v for (e, _), v in rate.items() if e == "goinfer")
                old = statistics.median(v for (e, _), v in rate.items() if e == "goinfer_old")
                results.append((f"decode {mk} depth 128", ratios, f"new {new:.2f} tok/s, main {old:.2f} tok/s"))
    for path in sorted(glob.glob(os.path.join(out, "ttft-*.json"))):
        d = json.load(open(path))
        mk, by = d["args"]["name"], {}
        for c in d["cells"]:
            if c.get("error"):
                void.append(f"ttft {mk}: {c['arm']} block {c['block']} errored: {c['error']}")
                continue
            by[(c["arm"], c["block"])] = (statistics.mean(c["ttft_s"]), c["prompt_tokens"])
        ratios, toks = [], set()
        for b in range(d["args"]["blocks"]):
            if ("new", b) not in by or ("old", b) not in by:
                void.append(f"ttft {mk}: block {b} is missing an arm")
                continue
            ratios.append(by[("old", b)][0] / by[("new", b)][0])
            toks.update(t for t in by[("new", b)][1] + by[("old", b)][1] if t)
        if ratios:
            new = statistics.median(v[0] for (a, _), v in by.items() if a == "new")
            old = statistics.median(v[0] for (a, _), v in by.items() if a == "old")
            results.append((f"time to first token {mk}, prompts of {sorted(toks)} tokens", ratios, f"new {new:.3f} s, main {old:.3f} s"))
    verdicts = []
    for name, ratios, note in results:
        v, med = verdict(ratios)
        verdicts.append(v)
        print(f"{v:6s} {name}: median new/main {med:.4f} over {len(ratios)} blocks (min {min(ratios):.4f}, max {max(ratios):.4f}; {note})")
        print("       by block: " + " ".join(f"{r:.4f}" for r in ratios))
    for v in void:
        print("VOID: " + v)
    if void or not results:
        print("VERDICT VOID" if void else "VERDICT VOID: no results found")
        return 2
    final = "FAIL" if "FAIL" in verdicts else "PARKED" if "PARKED" in verdicts else "PASS"
    print("VERDICT " + final)
    return 0 if final == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
