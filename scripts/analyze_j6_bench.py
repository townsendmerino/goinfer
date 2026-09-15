#!/usr/bin/env python3
"""analyze_j6_bench.py — paired-differencing analysis for scripts/bench_j6_scheduling.py's raw
output, per docs/measurements/j6-prefix-scheduling-PREREGISTERED.md's decision rule.

Computes, per rep, the ratio agg_tok_s(new_prefix_aware) / agg_tok_s(old_fifo) — a genuine PAIRED
ratio, not a ratio of means — then reports the mean/median/spread of those per-rep ratios and
applies the pre-registered pass/kill/void thresholds. Void checks (either arm's spread > 10%,
any rep with errors) are flagged, not silently dropped.
"""
import json, statistics, sys

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "docs/measurements/j6-prefix-scheduling-raw-2026-09-15.json"
    with open(path) as f:
        data = json.load(f)

    by_rep = {}
    for c in data["cells"]:
        by_rep.setdefault(c["rep"], {})[c["arm"]] = c

    ratios = []
    old_tps, new_tps = [], []
    voided = []
    for rep in sorted(by_rep):
        pair = by_rep[rep]
        if "old_fifo" not in pair or "new_prefix_aware" not in pair:
            voided.append((rep, "incomplete pair"))
            continue
        old, new = pair["old_fifo"], pair["new_prefix_aware"]
        if old["errors"] or new["errors"]:
            voided.append((rep, f"errors: old={old['errors']} new={new['errors']}"))
            continue
        if old["agg_tok_s"] <= 0:
            voided.append((rep, "old arm agg_tok_s <= 0"))
            continue
        ratio = new["agg_tok_s"] / old["agg_tok_s"]
        ratios.append(ratio)
        old_tps.append(old["agg_tok_s"])
        new_tps.append(new["agg_tok_s"])
        print(f"rep {rep:>3}: old={old['agg_tok_s']:.2f} tok/s  new={new['agg_tok_s']:.2f} tok/s  ratio={ratio:.4f}")

    if voided:
        print("\nVOIDED reps (not counted):")
        for rep, reason in voided:
            print(f"  rep {rep}: {reason}")

    if not ratios:
        print("\nNo valid paired reps — cannot compute a verdict.")
        return

    n = len(ratios)
    mean_ratio = statistics.mean(ratios)
    median_ratio = statistics.median(ratios)
    stdev_ratio = statistics.stdev(ratios) if n > 1 else 0.0

    def spread(xs):
        return (max(xs) - min(xs)) / statistics.mean(xs) if xs else 0.0

    old_spread, new_spread = spread(old_tps), spread(new_tps)

    print(f"\n=== {n} valid paired reps ===")
    print(f"old_fifo:         mean={statistics.mean(old_tps):.2f} tok/s  spread={old_spread*100:.1f}%")
    print(f"new_prefix_aware: mean={statistics.mean(new_tps):.2f} tok/s  spread={new_spread*100:.1f}%")
    print(f"paired ratio (new/old): mean={mean_ratio:.4f}  median={median_ratio:.4f}  stdev={stdev_ratio:.4f}")

    void_reasons = []
    if old_spread > 0.10:
        void_reasons.append(f"old_fifo spread {old_spread*100:.1f}% exceeds the pre-registered 10% void bound")
    if new_spread > 0.10:
        void_reasons.append(f"new_prefix_aware spread {new_spread*100:.1f}% exceeds the pre-registered 10% void bound")

    print()
    if void_reasons:
        print("VOID — pre-registered void rule triggered:")
        for r in void_reasons:
            print(f"  - {r}")
        print("Per the pre-registered rule: re-run, do not average through a void condition.")
        return

    if mean_ratio >= 1.3:
        verdict = "PASS"
    elif mean_ratio < 1.15:
        verdict = "KILL"
    else:
        verdict = "AMBIGUOUS — parked, not decided either way"
    print(f"VERDICT (against 1.3x pass / 1.15x kill): {verdict}")


if __name__ == "__main__":
    main()
