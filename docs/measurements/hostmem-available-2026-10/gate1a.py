#!/usr/bin/env python3
"""B41 gate 1a under amendments A2 and A3 (README.md here): every 20 s for 10 minutes, hostmem's reading against the kernel's jetsam
definition (vm.page_pageable_external_count + vm.page_free_count), both read by `hostmem -oracle` in one process, the jetsam figure
before and after ours; a sample whose jetsam figure moved by more than 0.02 GiB is retaken, up to three attempts. Pass (amendment A4): ours less
its speculative and purgeable pages within 2% of physical of the mean jetsam figure at every sample. Logs the Activity Monitor equivalent (free + speculative + file-backed pages) and the pressure level as context.

  python3 gate1a.py --hostmem <cmd/hostmem> --out DIR [--samples 30] [--every 20]
Exit 0 PASS, 1 FAIL.
"""
import argparse, json, os, re, subprocess, sys, time

GIB = 1 << 30


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
        am = o["free"] + o["speculative"] + o["file_backed"]
        graded = ours - o["speculative"] - o["purgeable"] - oracle  # amendment A4: like with like
        r = {"t": time.strftime("%H:%M:%S"), "ours": ours, "oracle": oracle, "jetsam_before": o["jetsam_before"],
             "jetsam_after": o["jetsam_after"], "attempts": len(tries), "moved": tries, "raw_diff": ours - oracle, "diff": graded,
             "am_equiv": am, "speculative": o["speculative"], "purgeable": o["purgeable"],
             "pressure": sysctl("kern.memorystatus_vm_pressure_level"),
             "within_2pct": abs(graded) <= 0.02 * phys, "within_1pt": abs(graded) <= 0.01 * phys}
        rows.append(r)
        print(f"{r['t']} ours {ours / GIB:.3f} GiB  oracle {oracle / GIB:.3f} GiB  raw {r['raw_diff'] / GIB:+.3f}  graded (less spec+purg) "
              f"{graded / GIB:+.3f} GiB ({100 * graded / phys:+.2f} pt)  "
              f"spec+purg {(o['speculative'] + o['purgeable']) / GIB:.3f}  AM-equiv {am / GIB:.3f}  pressure {r['pressure']}  "
              f"attempts {len(tries)}", flush=True)
        json.dump({"physical": phys, "rows": rows}, open(os.path.join(a.out, "gate1a.json"), "w"), indent=1)
        if i < a.samples - 1:
            time.sleep(a.every)
    worst = max(abs(r["diff"]) for r in rows)
    n2, n1 = sum(r["within_2pct"] for r in rows), sum(r["within_1pt"] for r in rows)
    print(f"{len(rows)} samples: within 2% of physical {n2}/{len(rows)}, within 1 point {n1}/{len(rows)}, worst |diff| {worst / GIB:.3f} GiB "
          f"({100 * worst / phys:.2f} pt); pressure max {max(r['pressure'] for r in rows)}")
    ok = n2 == len(rows)
    print(f"VERDICT {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
