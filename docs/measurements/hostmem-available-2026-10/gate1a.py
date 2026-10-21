#!/usr/bin/env python3
"""B41 gate 1a under amendments A2, A3 and A5 (README.md here): every 20 s for 10 minutes, hostmem's reading against the kernel's jetsam
definition (vm.page_pageable_external_count + vm.page_free_count), both read by `hostmem -oracle` in one process, the jetsam figure
before and after ours; a sample whose jetsam figure moved by more than 0.02 GiB is retaken, up to three attempts. Pass (amendment A5):
ours less its purgeable pages within 0.05 GiB of the mean jetsam figure at every sample. Logs the raw difference, the speculative and
purgeable pages and the pressure level as context, and counts the samples whose speculative pages exceed the bar (where a speculative
page counted twice would show).

  python3 gate1a.py --hostmem <cmd/hostmem> --out DIR [--samples 30] [--every 20]
Exit 0 PASS, 1 FAIL.
"""
import argparse, json, os, re, subprocess, sys, time

GIB = 1 << 30
BAR = 0.05 * GIB  # amendment A5


def sysctl(k):
    return int(subprocess.run(["sysctl", "-n", k], capture_output=True, text=True, check=True).stdout.strip())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hostmem", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--samples", type=int, default=30)
    ap.add_argument("--every", type=float, default=20)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    phys, ps = sysctl("hw.memsize"), sysctl("hw.pagesize")
    rows = []
    for i in range(a.samples):
        tries = []
        for attempt in range(3):  # amendment A3: one process, bracketed; retake a sample taken while memory moved
            out = subprocess.run([a.hostmem, "-oracle"], capture_output=True, text=True, check=True).stdout.split()
            o = {out[k]: int(out[k + 1]) for k in range(0, len(out) - 1, 2)}
            moved = abs(o["jetsam_before"] - o["jetsam_after"])
            tries.append(moved)
            if moved <= 0.02 * GIB:
                break
            print(f"  retake: the jetsam figure moved {moved / GIB:.3f} GiB during attempt {attempt + 1}", flush=True)
        oracle = (o["jetsam_before"] + o["jetsam_after"]) / 2
        ours = o["ours"]
        graded = ours - o["purgeable"] - oracle  # amendment A5: the jetsam figure plus purgeable pages is the reading
        r = {"t": time.strftime("%H:%M:%S"), "ours": ours, "oracle": oracle, "jetsam_before": o["jetsam_before"],
             "jetsam_after": o["jetsam_after"], "attempts": len(tries), "moved": tries, "raw_diff": ours - oracle, "diff": graded,
             "speculative": o["speculative"], "purgeable": o["purgeable"], "pressure": sysctl("kern.memorystatus_vm_pressure_level"),
             "within_bar": abs(graded) <= BAR, "visible": o["speculative"] > BAR}
        rows.append(r)
        print(f"{r['t']} ours {ours / GIB:.3f} GiB  oracle {oracle / GIB:.3f} GiB  raw {r['raw_diff'] / GIB:+.3f}  graded (less purg) "
              f"{graded / GIB:+.4f} GiB  spec {o['speculative'] / GIB:.3f}  purg {o['purgeable'] / GIB:.3f}  pressure {r['pressure']}  "
              f"attempts {len(tries)}", flush=True)
        json.dump({"physical": phys, "rows": rows}, open(os.path.join(a.out, "gate1a.json"), "w"), indent=1)
        if i < a.samples - 1:
            time.sleep(a.every)
    worst = max(abs(r["diff"]) for r in rows)
    n, vis = sum(r["within_bar"] for r in rows), sum(r["visible"] for r in rows)
    print(f"{len(rows)} samples: within {BAR / GIB:.2f} GiB {n}/{len(rows)}, worst |diff| {worst / GIB:.4f} GiB; speculative above the bar at "
          f"{vis}/{len(rows)} (where a double count would show); pressure max {max(r['pressure'] for r in rows)}")
    ok = n == len(rows)
    print(f"VERDICT {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
