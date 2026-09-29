#!/usr/bin/env python3
"""A memory ballast that holds this Mac's AVAILABLE memory in a band, for the 413 prefill-share end-to-end check
(fix413-e2e-2026-09-28.md). "Available" is exactly what decoder.HostRAMAvailableBytes reads on darwin: vm_stat's free
+ inactive + speculative + purgeable pages. That is the margin serve's prefill admission prices a request against.

The ballast holds ANONYMOUS memory and re-touches every page each second. Memory touched once and left idle turns
"inactive", which the probe counts as available, so an idle ballast would not move the probe.

SAFETY. The Mac is 16 GB with a nearly full disk, and this runs unattended at night:
  - it releases everything and exits (status ABORT) if swap grows more than --max-swap-mb over its own start, or the
    disk's free space falls below --min-disk-gb;
  - it never holds more than --max-gb;
  - it exits (status EXPIRED) after --max-s seconds, so it cannot outlive its round;
  - it exits if its parent dies (the driver is the only thing that should be holding it).

Status is written once a second as JSON to --status: state (growing / settled / ABORT / EXPIRED / stopped),
available bytes, ballast bytes, swap growth, and settled_for_s (consecutive seconds inside the band).
"""
import argparse, json, os, re, shutil, signal, subprocess, sys, time

CHUNK = 128 << 20


def vm_available():
    out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    page = int(re.search(r"page size of (\d+) bytes", out).group(1))
    def pages(name):
        m = re.search(rf"^{name}:\s+(\d+)\.", out, re.M)
        return int(m.group(1)) if m else 0
    return page * (pages("Pages free") + pages("Pages inactive") + pages("Pages speculative") + pages("Pages purgeable")), page


def swap_used():
    out = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout
    m = re.search(r"used = ([\d.]+)M", out)
    return int(float(m.group(1)) * (1 << 20)) if m else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lo-gb", type=float, required=True)
    ap.add_argument("--hi-gb", type=float, required=True)
    ap.add_argument("--status", required=True)
    ap.add_argument("--max-gb", type=float, default=12.0)
    ap.add_argument("--max-swap-mb", type=float, default=256)
    ap.add_argument("--min-disk-gb", type=float, default=2.5)
    ap.add_argument("--max-s", type=float, default=900)
    a = ap.parse_args()
    lo, hi = int(a.lo_gb * (1 << 30)), int(a.hi_gb * (1 << 30))
    target = (lo + hi) // 2
    parent = os.getppid()
    swap0 = swap_used()
    t0 = time.time()
    chunks, settled_since, state = [], None, "growing"
    stop = [False]
    signal.signal(signal.SIGTERM, lambda *_: stop.__setitem__(0, True))

    def write(extra=None):
        st = {"t": round(time.time() - t0, 1), "state": state, "available": avail, "ballast": len(chunks) * CHUNK,
              "swap_growth": swap_used() - swap0,
              "settled_for_s": round(time.time() - settled_since, 1) if settled_since else 0.0}
        if extra:
            st.update(extra)
        tmp = a.status + ".tmp"
        with open(tmp, "w") as f:
            json.dump(st, f)
        os.replace(tmp, a.status)
        print(json.dumps(st), flush=True)

    while True:
        avail, page = vm_available()
        growth = swap_used() - swap0
        why = None
        if growth > a.max_swap_mb * (1 << 20):
            why = ("ABORT", f"swap grew {growth / (1 << 20):.0f} MB")
        elif shutil.disk_usage(os.path.expanduser("~")).free < a.min_disk_gb * (1 << 30):
            why = ("ABORT", "disk free below the floor")
        elif time.time() - t0 > a.max_s:
            why = ("EXPIRED", f"max {a.max_s:.0f}s")
        elif os.getppid() != parent:
            why = ("ABORT", "parent gone")
        elif stop[0]:
            why = ("stopped", "SIGTERM")
        if why:
            chunks.clear()
            state = why[0]
            avail, _ = vm_available()
            write({"reason": why[1]})
            sys.exit(0 if state == "stopped" else 3)
        if avail > hi and len(chunks) * CHUNK < a.max_gb * (1 << 30):
            n = max(1, min(4, (avail - target) // CHUNK))  # up to 512 MB a second while far above the band
            for _ in range(n):
                b = bytearray(CHUNK)
                for i in range(0, CHUNK, page):
                    b[i] = 1
                chunks.append(b)
        elif avail < lo and chunks:
            chunks.pop()
        for b in chunks:  # keep every page active, or the probe counts it as available again
            for i in range(0, CHUNK, page):
                b[i] = (b[i] + 1) & 0xFF
        inside = lo <= avail <= hi
        if inside and settled_since is None:
            settled_since = time.time()
        elif not inside:
            settled_since = None
        state = "settled" if settled_since and time.time() - settled_since >= 10 else "growing"
        write()
        time.sleep(1.0)


if __name__ == "__main__":
    main()
