#!/usr/bin/env python3
"""Held-out set 5's grading (SB2-vk; README, "Held-out set 5"), and set 6's under the same rule: usage: score_heldout5.py <dir> [pass],
pass heldout5 (the default) or heldout6. For each scored load (one with a
<pass>-*-probe.txt; a load that did not go resident has none and is not scored): the planner's list from the model against the
ledger, M1' (m1p_predict in score_vk1.py, its constant fixed on G-VK1 pass 1) over the recorded list against nvidia-smi's rise, and
the webgpu plan's priced bytes against the rise. The rule is the README's, registered before any set-5 load."""
import os, re, sys
import score_vk1 as s

M = s.MiB


def main():
    d = sys.argv[1]
    p = sys.argv[2] if len(sys.argv) > 2 else "heldout5"
    vk = open(os.path.join(d, f"{p}-vulkaninfo-memory.txt")).read()
    bar = any("MEMORY_PROPERTY_DEVICE_LOCAL_BIT" in t and "MEMORY_PROPERTY_HOST_VISIBLE_BIT" in t and "MEMORY_PROPERTY_HOST_COHERENT_BIT" in t
              for t in re.split(r"memoryTypes\[\d+\]", vk)[1:])
    Ls = s.loads(d, p)
    print(f"{'load':26} {'rise MiB':>8} {'list-ledger':>11} {'M1p-rise':>9} {'plan-rise':>9}")
    wl = wu = wp = 0.0
    for L in Ls:
        kv = dict(re.findall(r"(\w+)=(\S+)", open(os.path.join(d, f"{p}-{L['tag']}-probe.txt")).read()))
        dl = (int(kv["plan_list"]) - L["ledger"]) / M
        du = (s.m1p_predict(L["bufs"], bar) - L["rise"]) / M
        dp = (int(kv["plan"]) - L["rise"]) / M
        wl, wu, wp = max(wl, abs(dl)), max(wu, abs(du)), max(wp, abs(dp))
        print(f"{L['tag']:26} {L['rise'] / M:8.0f} {dl:+11.1f} {du:+9.1f} {dp:+9.1f}")
    n = len(Ls)
    if n < 2:
        v = f"VOID: {n} scored loads, fewer than 2"
    elif wl > 96 or wu > 96 or wp > 96:
        v = "FAILS: a scored load is beyond 96 MiB"
    elif wl <= 32 and wu <= 64 and wp <= 64:
        v = f"HOLDS: {n} scored loads"
    else:
        v = "AMBIGUOUS"
    print(f"{v} (worst list-vs-ledger {wl:.1f} MiB, bar 32; M1' {wu:.1f} and plan {wp:.1f} MiB, bar 64; beyond 96 fails)")


if __name__ == "__main__":
    main()
