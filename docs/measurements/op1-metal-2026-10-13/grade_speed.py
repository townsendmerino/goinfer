#!/usr/bin/env python3
"""Grade OP1's Metal speed guard, G2-S (docs/tasks/task-forward-architecture-2026-10.md §4.6), from one output directory.

  grade_speed.py <out dir>     # decode-metal-*.json (bench_peer.py) and ttft-metal-*.json (bench_prefill_stall.py)

Per cell the statistic is new / old as a rate, per paired block (for time to first token, old's time over new's), and
its 95% two-sided t interval on the log ratio. The registered rule: PASS when the interval clears 0.97, FAIL when it lies
under 0.97, PARKED (for the owner) when it straddles 0.97. A cell with fewer than the registered 4 blocks, an errored
arm or a block missing an arm is VOID. Cells that were added after the registration are reported and do not gate.
Exit 0 only when every registered cell passes.
"""
import glob, json, math, os, statistics, sys

BAR, BLOCKS = 0.97, 4
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201}
REGISTERED = {("1.5B", 128), ("1.5B", 3900), ("gemma3-4b", 128)}  # decode cells; the time-to-first-token cell is registered too


def interval(ratios):
    logs = [math.log(r) for r in ratios]
    m, n = statistics.mean(logs), len(logs)
    if n < 2:
        return math.exp(m), float("nan"), float("nan")
    half = T975[n - 1] * statistics.stdev(logs) / math.sqrt(n)
    return math.exp(m), math.exp(m - half), math.exp(m + half)


def verdict(lo, hi):
    if lo >= BAR:
        return "PASS"
    if hi < BAR:
        return "FAIL"
    return "PARKED"


def main():
    out, results, void = sys.argv[1], [], []
    for path in sorted(glob.glob(os.path.join(out, "decode-metal-*.json"))):
        cells = [r for r in json.load(open(path)) if r.get("phase") in ("A", "B")]
        for mk, depth in sorted({(r["model"], r["depth"]) for r in cells}):
            rate = {}
            for r in cells:
                if (r["model"], r["depth"]) != (mk, depth):
                    continue
                if r.get("error") or not r.get("runs"):
                    void.append(f"decode {mk} depth {depth}: {r['engine']} block {r.get('block')} errored: {r.get('error')}")
                    continue
                rate[(r["engine"], r.get("block"))] = statistics.mean(r["runs"])
            ratios = []
            for b in sorted({b for _, b in rate}, key=str):
                if ("goinfer", b) not in rate or ("goinfer_old", b) not in rate:
                    void.append(f"decode {mk} depth {depth}: block {b} is missing an arm")
                    continue
                ratios.append(rate[("goinfer", b)] / rate[("goinfer_old", b)])
            if ratios:
                new = statistics.median(v for (e, _), v in rate.items() if e == "goinfer")
                old = statistics.median(v for (e, _), v in rate.items() if e == "goinfer_old")
                results.append((f"decode {mk} depth {depth}", (mk, depth) in REGISTERED, ratios, f"new {new:.2f} tok/s, base {old:.2f} tok/s"))
    for path in sorted(glob.glob(os.path.join(out, "ttft-metal-*.json"))):
        res = json.load(open(path)).get("results", {})
        by = {}
        for key, r in res.items():
            arm, _, block = key.partition("_")
            t = [x for x in r.get("newcomer_ttft_s", []) if x]
            if len(t) != len(r.get("newcomer_ttft_s", [])) or not t:
                void.append(f"time to first token: cell {key} has a newcomer with no first token")
                continue
            by[(arm, block)] = statistics.mean(t)
        ratios = []
        for b in sorted({b for _, b in by}):
            if ("new", b) not in by or ("old", b) not in by:
                void.append(f"time to first token: block {b} is missing an arm")
                continue
            ratios.append(by[("old", b)] / by[("new", b)])
        if ratios:
            new = statistics.median(v for (a, _), v in by.items() if a == "new")
            old = statistics.median(v for (a, _), v in by.items() if a == "old")
            results.append((f"time to first token, {os.path.basename(path)[len('ttft-metal-'):-len('.json')]}", True, ratios, f"new {new:.3f} s, base {old:.3f} s"))
    verdicts = []
    for name, registered, ratios, note in results:
        mean, lo, hi = interval(ratios)
        v = verdict(lo, hi) if len(ratios) >= 2 else "VOID"
        if registered and len(ratios) < BLOCKS:
            void.append(f"{name}: {len(ratios)} paired block(s), the registration asks for {BLOCKS}")
        tag = v if registered else f"({v}, reported, not gating)"
        if registered:
            verdicts.append(v)
        print(f"{tag:8s} {name}: new/base {mean:.4f}, 95% interval [{lo:.4f}, {hi:.4f}] over {len(ratios)} blocks ({note})")
        print("         by block: " + " ".join(f"{r:.4f}" for r in ratios))
    for v in void:
        print("VOID: " + v)
    want = len(REGISTERED) + 1
    if len(verdicts) != want:
        void.append(f"{len(verdicts)} registered cell(s) have results, want {want}")
        print("VOID: " + void[-1])
    if void or not results:
        print("VERDICT VOID")
        return 2
    final = "FAIL" if "FAIL" in verdicts else "PARKED" if "PARKED" in verdicts else "PASS"
    print(f"VERDICT {final} (bar {BAR}: every registered cell's interval must clear it)")
    return 0 if final == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
