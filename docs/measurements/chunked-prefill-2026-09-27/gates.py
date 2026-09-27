#!/usr/bin/env python3
"""Chunked prefill's gates (docs/tasks/task-concurrency-2026-09.md, registered in d8ae520a) from bench_prefill_stall.py's
JSON. Stall cells old_1..new_3 (3 decoders + 3 newcomers), solo cells oldsolo_1..newsolo_3 (the newcomers alone)."""
import json, statistics, sys


def main(path):
    R = json.load(open(path))["results"]
    ok = True
    # 1. identity: every decoder reply and every newcomer reply equal across the stall cells; the newcomers' replies
    #    also equal across the solo cells, and between the two arms (the same prompts, greedy).
    stall = {k: v for k, v in R.items() if "solo" not in k}
    solo = {k: v for k, v in R.items() if "solo" in k}
    ref = next(iter(stall.values()))
    for k, v in sorted(stall.items()):
        if v["decoder_sha"] != ref["decoder_sha"] or v["newcomer_sha"] != ref["newcomer_sha"]:
            print(f"GATE 1 FAIL: {k}: replies differ ({v['decoder_sha']} {v['newcomer_sha']} vs {ref['decoder_sha']} {ref['newcomer_sha']})")
            ok = False
    for k, v in sorted(solo.items()):
        if v["newcomer_sha"] != ref["newcomer_sha"]:
            print(f"GATE 1 FAIL: {k}: newcomer replies differ from the stall cells'")
            ok = False
    print(f"GATE 1 identity: {'PASS' if ok else 'FAIL'} ({len(stall)} stall cells, {len(solo)} solo cells)")

    gap, ttft, wall = [], [], []
    for i in (1, 2, 3):
        o, n = R.get(f"old_{i}"), R.get(f"new_{i}")
        if not (o and n):
            continue
        gap.append(n["decoder_gap_ms"]["max"] / o["decoder_gap_ms"]["max"])
        ttft += [b / a for a, b in zip(o["newcomer_ttft_s"], n["newcomer_ttft_s"])]
        wall.append(n["wall_s"] / o["wall_s"])
        print(f"pair {i}: decoder gap max old {o['decoder_gap_ms']['max']:.0f} new {n['decoder_gap_ms']['max']:.0f} ms "
              f"(p99 {o['decoder_gap_ms']['p99']:.0f} -> {n['decoder_gap_ms']['p99']:.0f}, p50 {o['decoder_gap_ms']['p50']:.1f} -> "
              f"{n['decoder_gap_ms']['p50']:.1f}); newcomer TTFT old {o['newcomer_ttft_s']} new {n['newcomer_ttft_s']} s; "
              f"wall {o['wall_s']:.2f} -> {n['wall_s']:.2f} s")
    if gap:
        g2, g3, g4 = statistics.median(gap), statistics.median(ttft), statistics.median(wall)
        print(f"GATE 2 the stall (decoders' max gap) median {g2:.3f}x (bar <= 0.5): "
              f"{'PASS' if g2 <= 0.5 else 'owner band' if g2 <= 0.8 else 'FAIL'}")
        print(f"GATE 3 newcomer TTFT median {g3:.3f}x (bar <= 1.5): {'PASS' if g3 <= 1.5 else 'FAIL'}")
        print(f"GATE 4 wall median {g4:.3f}x (bar <= 1.05): {'PASS' if g4 <= 1.05 else 'FAIL'}")
    st = []
    for i in (1, 2, 3):
        o, n = R.get(f"oldsolo_{i}"), R.get(f"newsolo_{i}")
        if o and n:
            st += [b / a for a, b in zip(o["newcomer_ttft_s"], n["newcomer_ttft_s"])]
            print(f"solo pair {i}: TTFT old {o['newcomer_ttft_s']} new {n['newcomer_ttft_s']} s")
    if st:
        g5 = statistics.median(st)
        print(f"GATE 5 solo guard (newcomer TTFT alone) median {g5:.3f}x (bar <= 1.05): {'PASS' if g5 <= 1.05 else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
