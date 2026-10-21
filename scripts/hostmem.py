#!/usr/bin/env python3
"""goinfer's available-memory reading for the scripts (B41, docs/measurements/hostmem-available-2026-10/README.md): the number
`cmd/hostmem` prints, which is the one the fit guard reads (internal/hostmem: on macOS physical - (app + wired + compressed),
Activity Monitor's split; on Linux /proc/meminfo MemAvailable). No formula lives here. A script that cannot run the command gets
None, which its callers already treat as unknown.

  python3 scripts/hostmem.py      prints available MiB
"""
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def available_mb(repo=REPO, timeout=120):
    """Available MiB from cmd/hostmem in repo, or None."""
    try:
        r = subprocess.run(["go", "run", "./cmd/hostmem", "-mb"], cwd=repo, env=dict(os.environ, GOWORK="off"),
                           capture_output=True, text=True, timeout=timeout)
        return int(r.stdout.strip()) if r.returncode == 0 else None
    except Exception:
        return None


if __name__ == "__main__":
    mb = available_mb()
    if mb is None:
        sys.exit("hostmem: no reading")
    print(mb)
