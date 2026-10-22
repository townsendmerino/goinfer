#!/usr/bin/env python3
"""G4's never-swap reading (kv4-cuda-2026-10-21.md §2): every 0.5 s, for every live process whose name is one of the given names, the
peak VmHWM and the peak VmSwap from /proc/<pid>/status; one JSON line per process when it exits (or when the sampler is stopped), with
its first and last sample times, so a cell is matched to its server by time. System-wide swap is recorded beside it as context only.

  procmem_sampler.py <out.jsonl> <name> [<name> ...]     (stop with SIGTERM)
"""
import json, os, signal, sys, time

out, names = sys.argv[1], set(sys.argv[2:])
seen, stop = {}, False


def handler(*_):
    global stop
    stop = True


signal.signal(signal.SIGTERM, handler)


def status(pid):
    try:
        with open(f"/proc/{pid}/status") as f:
            d = dict(l.split(":", 1) for l in f if ":" in l)
        return d["Name"].strip(), int(d.get("VmHWM", "0 kB").split()[0]), int(d.get("VmSwap", "0 kB").split()[0])
    except Exception:
        return None


def sys_swap():
    with open("/proc/meminfo") as f:
        d = dict(l.split(":", 1) for l in f)
    return (int(d["SwapTotal"].split()[0]) - int(d["SwapFree"].split()[0])) // 1024


def flush(pid):
    r = seen.pop(pid)
    with open(out, "a") as f:
        f.write(json.dumps(r) + "\n")


while not stop:
    now = time.time()
    live = set()
    for p in os.listdir("/proc"):
        if not p.isdigit():
            continue
        s = status(p)
        if not s or s[0] not in names:
            continue
        live.add(p)
        r = seen.setdefault(p, {"pid": int(p), "name": s[0], "first": now, "hwm_kb": 0, "swap_peak_kb": 0, "sys_swap_mb_first": sys_swap()})
        r["last"], r["hwm_kb"], r["swap_peak_kb"] = now, max(r["hwm_kb"], s[1]), max(r["swap_peak_kb"], s[2])
        r["sys_swap_mb_last"] = sys_swap()
    for p in [p for p in seen if p not in live]:
        flush(p)
    time.sleep(0.5)
for p in list(seen):
    flush(p)
