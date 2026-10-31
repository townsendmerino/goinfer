#!/usr/bin/env python3
"""M1''s held-out grading (SB2-vk, task doc "SB2-vk", M1'): usage: score_m1p.py <dir> <pass>. M1' is m1p_predict in score_vk1.py,
its device constant fitted on G-VK1 pass 1 (m1p_fit.txt) and fixed before this pass. Loads that did not go resident have no
probe file and are not scored."""
import os, re, sys
import score_vk1 as s

M = s.MiB


def main():
    d, p = sys.argv[1], sys.argv[2]
    vk = open(os.path.join(d, f"{p}-vulkaninfo-memory.txt")).read()
    bar = any("MEMORY_PROPERTY_DEVICE_LOCAL_BIT" in t and "MEMORY_PROPERTY_HOST_VISIBLE_BIT" in t and "MEMORY_PROPERTY_HOST_COHERENT_BIT" in t
              for t in re.split(r"memoryTypes\[\d+\]", vk)[1:])
    Ls = s.loads(d, p)
    print(f"{'load':24} {'rise MiB':>8} {'M1p err':>8} {'M1 err':>8} {'M0 err':>8}")
    worst = 0.0
    for L in Ls:
        e = (s.m1p_predict(L["bufs"], bar) - L["rise"]) / M
        worst = max(worst, abs(e))
        print(f"{L['tag']:24} {L['rise'] / M:8.0f} {e:+8.1f} {(s.m1_predict(L['bufs'], bar, 64 * M) - L['rise']) / M:+8.1f} {(L['ledger'] - L['rise']) / M:+8.1f}")
    n = len(Ls)
    if n < 3:
        print(f"VOID: {n} scored loads, fewer than 3")
    elif worst <= 64:
        print(f"M1' HOLDS: every one of {n} held-out loads within 64 MiB (worst {worst:.1f})")
    elif worst > 96:
        print(f"M1' FAILS: a held-out load misses by {worst:.1f} MiB, over 96")
    else:
        print(f"M1' AMBIGUOUS: worst {worst:.1f} MiB, between 64 and 96")


if __name__ == "__main__":
    main()
