#!/usr/bin/env python3
"""Grade KV2's G6 (docs/measurements/kv2-2026-10-14.md §2, G6) from one box's output directory: KV1's cells and
instruments, graded by G6's interval rule.

  grade_kv2_speed.py <out dir>     # decode-*.json (bench_peer.py, phase A) and ttft-*.json (kv1_ttft.py)

Per block d = ln(new / old) of the rate (for time to first token, ln(old / new) of the time). With the interval
mean(d) +/- t(0.975, N-1) * sd(d) / sqrt(N): PASS when the lower end is at or above ln(0.99) and the median ratio is at
most 1.01; FAIL when the upper end is below ln(0.99); PARKED otherwise. KV1's median-in-band reading is printed beside
it. VOID: a cell errored, or a block is missing an arm. Exit 0 only when every cell passes.
"""
import glob, json, math, os, statistics, sys


T975 = [12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228, 2.201, 2.179, 2.160, 2.145, 2.131, 2.120,
        2.110, 2.101, 2.093, 2.086, 2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045, 2.042]


def verdict(ratios):
    med = statistics.median(ratios)
    d = [math.log(r) for r in ratios]
    n = len(d)
    if n < 2:
        return "PARKED", med, (float("nan"), float("nan"))
    half = T975[min(n - 2, len(T975) - 1)] * statistics.stdev(d) / math.sqrt(n)
    lo, hi = math.exp(statistics.mean(d) - half), math.exp(statistics.mean(d) + half)
    if lo >= 0.99 and med <= 1.01:
        return "PASS", med, (lo, hi)
    if hi < 0.99:
        return "FAIL", med, (lo, hi)
    return "PARKED", med, (lo, hi)


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
        v, med, (lo, hi) = verdict(ratios)
        verdicts.append(v)
        kv1 = "in" if 0.99 <= med <= 1.01 else "outside"
        print(f"{v:6s} {name}: median new/main {med:.4f} over {len(ratios)} blocks, 95% interval [{lo:.4f}, {hi:.4f}] "
              f"(KV1's reading: median {kv1} [0.99, 1.01]; min {min(ratios):.4f}, max {max(ratios):.4f}; {note})")
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
