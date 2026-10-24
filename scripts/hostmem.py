#!/usr/bin/env python3
"""goinfer's memory readings for the scripts (B41, docs/measurements/hostmem-available-2026-10/README.md): the numbers `cmd/hostmem`
prints (internal/hostmem). available_mb is the figure the fit guard decides loads by (on macOS memory_pressure's free percentage of
physical memory, less 500 MiB); reclaimable_mb is the figure a timed run's validity is decided by (on macOS free + file-backed +
purgeable pages, what can be taken without compressing or swapping). On Linux both are /proc/meminfo MemAvailable. No formula lives
here. A script that cannot run the command gets None, which its callers already treat as unknown.

  python3 scripts/hostmem.py                 prints available MiB
  python3 scripts/hostmem.py --reclaimable   prints reclaimable MiB
"""
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _mb(flags, repo, timeout):
    try:
        r = subprocess.run(["go", "run", "./cmd/hostmem", "-mb"] + flags, cwd=repo, env=dict(os.environ, GOWORK="off"),
                           capture_output=True, text=True, timeout=timeout)
        return int(r.stdout.strip()) if r.returncode == 0 else None
    except Exception:
        return None


def available_mb(repo=REPO, timeout=120):
    """Available MiB from cmd/hostmem in repo (load decisions), or None."""
    return _mb([], repo, timeout)


def reclaimable_mb(repo=REPO, timeout=120):
    """Reclaimable MiB from cmd/hostmem -reclaimable in repo (a timed run's validity), or None."""
    return _mb(["-reclaimable"], repo, timeout)


def main(argv):
    """Prints available MiB, or reclaimable MiB with --reclaimable (or cmd/hostmem's -reclaimable); any other argument is an error."""
    flags = set(argv)
    if flags - {"--reclaimable", "-reclaimable"}:
        sys.exit("usage: hostmem.py [--reclaimable]; unknown: " + " ".join(sorted(flags - {"--reclaimable", "-reclaimable"})))
    mb = reclaimable_mb() if flags else available_mb()
    if mb is None:
        sys.exit("hostmem: no reading")
    print(mb)


if __name__ == "__main__":
    main(sys.argv[1:])
