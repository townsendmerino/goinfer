#!/usr/bin/env python3
"""The heavy-load void rule for a timed cell: ONE definition that bench_peer.py, night.py and every shell runner call.

A cell may START only if both hold:
  (a) this box's available memory (the reading the fit guard decides loads by: on macOS memory_pressure's free percentage of
      physical memory less 500 MiB, on Linux MemAvailable; scripts/hostmem.py) is at least the cell's fit need, with no margin;
  (b) the 1-minute load average is not above half the core count (the cap is a parameter, 4.0 on the 8-core Mac).
A cell is VOID if system swap grows at all between its start and its end. The reclaimable reading is no longer a void term; it is
recorded, with load, available memory and swap, at the start and the end of every cell, so a record taken under this rule can be
told from one taken under the old +2 GB rule.

  python3 scripts/void_rule.py start --need-mb N [--load-cap X] [--json-out F]   exit 0 clean, 75 void (the reason on stderr)
  python3 scripts/void_rule.py end --start-json F [--json-out G]                  exit 0, or 76 when swap grew (the reason on stderr)
  python3 scripts/void_rule.py snapshot                                            prints the four readings as JSON

Both print the snapshot as one JSON line on stdout. 75 matches night.py's RETRY_RC, so a runner that exits with it is re-queued.
"""
import json
import os
import re
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hostmem  # noqa: E402

RETRY_RC = 75
SWAP_VOID_RC = 76


def swap_used_mb():
    """System swap in use, MB: darwin's sysctl vm.swapusage, Linux's /proc/meminfo. None where neither reads."""
    try:
        if sys.platform == "darwin":
            out = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True, timeout=5).stdout
            m = re.search(r"used = ([0-9.]+)M", out)
            return float(m.group(1)) if m else None
        mi = dict(line.split(":", 1) for line in open("/proc/meminfo"))
        return (int(mi["SwapTotal"].split()[0]) - int(mi["SwapFree"].split()[0])) / 1024
    except Exception:
        return None


def snapshot():
    """The four readings a cell records at its start and its end; None for one that cannot be read."""
    return {"t": time.strftime("%F %T"), "load1": os.getloadavg()[0], "ncores": os.cpu_count() or 1,
            "available_mb": hostmem.available_mb(), "reclaimable_mb": hostmem.reclaimable_mb(), "swap_mb": swap_used_mb()}


def load_cap(ncores):
    """Half the cores: the cap a cell's start may not exceed."""
    return ncores / 2


def start_void(snap, need_mb, cap=None):
    """The reason a cell starting in state snap is void, or None. A reading that is None is unknown and does not void, as the fit
    guard treats an unknown. need_mb None or 0 means the cell loads no model, so the memory clause does not apply."""
    cap = load_cap(snap["ncores"]) if cap is None else cap
    if snap.get("load1") is not None and snap["load1"] > cap:
        return (f"load {snap['load1']:.2f} above half of {snap['ncores']} cores" if cap == snap["ncores"] / 2
                else f"load {snap['load1']:.2f} above the cap {cap:g}")
    avail = snap.get("available_mb")
    if need_mb and avail is not None and avail < need_mb:
        return f"available memory {avail} MB below the fit need {need_mb:.0f} MB"
    return None


def swap_void(start, end):
    """The reason a cell whose swap read `start` at its start and `end` at its end is void, or None: any growth voids. Either reading
    None is unknown and does not void."""
    s, e = start.get("swap_mb"), end.get("swap_mb")
    if s is not None and e is not None and e > s:
        return f"system swap grew {e - s:.1f} MB during the cell ({s:.1f} to {e:.1f} MB)"
    return None


def main(argv):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("start")
    s.add_argument("--need-mb", type=float, default=0)
    s.add_argument("--load-cap", type=float)
    s.add_argument("--json-out")
    e = sub.add_parser("end")
    e.add_argument("--start-json", required=True)
    e.add_argument("--json-out")
    sub.add_parser("snapshot")
    a = ap.parse_args(argv)
    snap = snapshot()
    if a.cmd == "snapshot":
        print(json.dumps(snap))
        return 0
    if a.cmd == "start":
        why = start_void(snap, a.need_mb, a.load_cap)
        snap["need_mb"], snap["void"] = a.need_mb, why
    else:
        why = swap_void(json.load(open(a.start_json)), snap)
        snap["void"] = why
    if a.json_out:
        json.dump(snap, open(a.json_out, "w"))
    print(json.dumps(snap))
    if why:
        print("VOID: " + why, file=sys.stderr)
        return RETRY_RC if a.cmd == "start" else SWAP_VOID_RC
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
