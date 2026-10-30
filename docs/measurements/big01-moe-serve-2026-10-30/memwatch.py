#!/usr/bin/env python3
"""Sample a goinfer-serve process's memory once a second until it exits, and stop it if swap grows.

Usage: memwatch.py <pid> <log>

Each line: wall time, the process's VmRSS / RssAnon / RssFile / RssShmem / VmPin / VmLck / VmSwap / VmHWM (GB, 10^9
bytes; /proc reports KiB), the system's swap in use and MemAvailable (GB), and VRAM used (MiB, from nvidia-smi every
10 s). When the process's own VmSwap passes KILL_PROC_MB, or system swap in use rises more than KILL_SYS_MB (10^6
bytes) over its value at the first sample, it writes a VOID line and sends SIGTERM, then SIGKILL if the process is
still there after KILL_GRACE_S: the run is void either way, and stopping it at once keeps the box from paging further.
"""
import os
import signal
import subprocess
import sys
import time

KILL_PROC_MB = 64
KILL_SYS_MB = 512
KILL_GRACE_S = 15
KIB = 1024 / 1e9  # one /proc KiB in GB


def proc_status(pid):
    out = {}
    with open(f"/proc/{pid}/status") as f:
        for line in f:
            k, _, v = line.partition(":")
            if k in ("VmRSS", "RssAnon", "RssFile", "RssShmem", "VmPin", "VmLck", "VmSwap", "VmHWM"):
                out[k] = int(v.split()[0]) * KIB
    return out


def meminfo():
    m = {}
    with open("/proc/meminfo") as f:
        for line in f:
            k, _, v = line.partition(":")
            m[k] = int(v.split()[0]) * KIB
    return m["SwapTotal"] - m["SwapFree"], m["MemAvailable"]


def vram():
    try:
        return subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                              capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception as e:  # noqa: BLE001
        return f"? ({e})"


def stop(pid, log):
    """SIGTERM, then SIGKILL if pid is still there after KILL_GRACE_S."""
    try:
        os.kill(pid, signal.SIGTERM)
        for _ in range(KILL_GRACE_S * 10):
            time.sleep(0.1)
            os.kill(pid, 0)
        log.write(f"still running after {KILL_GRACE_S} s; sending SIGKILL to {pid}\n")
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def main():
    pid, path = int(sys.argv[1]), sys.argv[2]
    base_sys = None
    peak = {}
    n = 0
    v = vram()
    with open(path, "a", buffering=1) as log:
        log.write("time VmRSS RssAnon RssFile RssShmem VmPin VmLck VmSwap VmHWM | sys_swap MemAvail | vram_MiB\n")
        while True:
            try:
                s = proc_status(pid)
            except (FileNotFoundError, ProcessLookupError):
                break
            sw, avail = meminfo()
            if base_sys is None:
                base_sys = sw
            if n % 10 == 0:
                v = vram()
            for k, x in s.items():
                peak[k] = max(peak.get(k, 0), x)
            peak["sys_swap"] = max(peak.get("sys_swap", 0), sw)
            log.write(time.strftime("%H:%M:%S ") + " ".join(f"{s.get(k, 0):.2f}" for k in
                      ("VmRSS", "RssAnon", "RssFile", "RssShmem", "VmPin", "VmLck", "VmSwap", "VmHWM"))
                      + f" | {sw:.2f} {avail:.2f} | {v}\n")
            why = None
            if s.get("VmSwap", 0) * 1000 > KILL_PROC_MB:
                why = f"server VmSwap {s['VmSwap']:.2f} GB > {KILL_PROC_MB} MB"
            elif (sw - base_sys) * 1000 > KILL_SYS_MB:
                why = f"system swap +{(sw - base_sys):.2f} GB over the first sample's {base_sys:.2f} GB"
            if why:
                log.write(f"VOID: {why}; sending SIGTERM to {pid}\n")
                stop(pid, log)
                break
            n += 1
            time.sleep(1)
        log.write("peak " + " ".join(f"{k}={x:.2f}" for k, x in sorted(peak.items())) + "\n")


if __name__ == "__main__":
    main()
