#!/usr/bin/env python3
"""TE1's mutation hog: every core busy for N seconds (default 90). A file with a __main__ guard, because macOS's
multiprocessing spawns its workers by re-importing __main__, which `python3 -c` cannot provide (attempt 1, 2026-09-28:
"Can't get attribute 'spin' on <module '__main__'>", so no hog ever ran).

    python3 te1_hog.py 90
"""
import multiprocessing as mp
import sys
import time


def spin(t):
    end = time.time() + t
    while time.time() < end:
        pass


if __name__ == "__main__":
    secs = float(sys.argv[1]) if len(sys.argv) > 1 else 90.0
    ps = [mp.Process(target=spin, args=(secs,)) for _ in range(mp.cpu_count())]
    for p in ps:
        p.start()
    for p in ps:
        p.join()
    bad = [p.exitcode for p in ps if p.exitcode != 0]
    sys.exit(1 if bad else 0)
