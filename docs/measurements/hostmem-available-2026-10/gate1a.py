#!/usr/bin/env python3
"""B41 gate 1a under amendment A2 (README.md here): every 20 s for 10 minutes, hostmem's reading against the kernel's jetsam
definition read from sysctl (vm.page_pageable_external_count + vm.page_free_count, times hw.pagesize). Pass: within 2% of physical
at every sample. Logs the Activity Monitor equivalent (free + speculative + file-backed pages) and the pressure level as context.

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
        ours = int(subprocess.run([a.hostmem], capture_output=True, text=True, check=True).stdout.strip())
        ext, free = sysctl("vm.page_pageable_external_count"), sysctl("vm.page_free_count")
        spec, purg = sysctl("vm.page_speculative_count"), sysctl("vm.page_purgeable_count")
        oracle = (ext + free) * ps
        v = subprocess.run([a.hostmem, "-v"], capture_output=True, text=True).stdout
        m = {k: float(x) for k, x in re.findall(r"(free|speculative|file-backed) ([\d.]+)", v)}
        am = (m.get("free", 0) + m.get("speculative", 0) + m.get("file-backed", 0)) * GIB
        r = {"t": time.strftime("%H:%M:%S"), "ours": ours, "oracle": oracle, "diff": ours - oracle, "am_equiv": int(am),
             "speculative": spec * ps, "purgeable": purg * ps, "pressure": sysctl("kern.memorystatus_vm_pressure_level"),
             "within_2pct": abs(ours - oracle) <= 0.02 * phys, "within_1pt": abs(ours - oracle) <= 0.01 * phys}
        rows.append(r)
        print(f"{r['t']} ours {ours / GIB:.3f} GiB  oracle {oracle / GIB:.3f} GiB  diff {r['diff'] / GIB:+.3f} GiB ({100 * r['diff'] / phys:+.2f} pt)  "
              f"spec+purg {(spec + purg) * ps / GIB:.3f}  AM-equiv {am / GIB:.3f}  pressure {r['pressure']}", flush=True)
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
