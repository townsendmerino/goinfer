#!/usr/bin/env python3
"""sampler.py --stop-file F --interval S file...: one line per interval with each file's size and resident bytes
(mincore) and the kernel's dirty, writeback and file-cache counters. It stops when the stop file disappears."""
import argparse, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cachestat import resident

ap = argparse.ArgumentParser()
ap.add_argument("--stop-file", required=True)
ap.add_argument("--interval", type=float, default=2.0)
ap.add_argument("files", nargs="+")
a = ap.parse_args()
KEYS = ("MemAvailable", "Cached", "Dirty", "Writeback", "Active(file)", "Inactive(file)")
t0 = time.time()
while os.path.exists(a.stop_file):
    mi = {}
    for line in open("/proc/meminfo"):
        k, v = line.split(":", 1)
        if k in KEYS:
            mi[k] = int(v.split()[0]) // 1024
    parts = [f"t={time.time() - t0:7.1f}"]
    for f in a.files:
        try:
            r, s = resident(f)
            parts.append(f"{os.path.basename(f)}: size {s / 1e6:.0f} MB, resident {r / 1e6:.0f} MB")
        except OSError:
            parts.append(f"{os.path.basename(f)}: absent")
    parts.append(" ".join(f"{k}={mi.get(k, 0)}MB" for k in KEYS))
    try:
        parts.append("io-psi " + open("/proc/pressure/io").readline().split()[1])
    except OSError:
        pass
    print(" | ".join(parts), flush=True)
    time.sleep(a.interval)
