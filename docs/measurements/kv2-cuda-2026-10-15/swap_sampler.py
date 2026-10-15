#!/usr/bin/env python3
"""Run a command and sample its memory and swap every 0.5 s until it exits (GV-C(c), bisected by day).

    python3 swap_sampler.py SUMMARY -- CMD [ARGS...]

Reads, per sample, /proc/<pid>/status of the command and of every descendant (VmRSS, VmHWM, VmSwap), the system's swap in
use (SwapTotal - SwapFree), and this session's user slice (memory.current, memory.swap.current) when cgroup v2 exposes it.
The slice's peak is the maximum of the sampled memory.current: its memory.peak is a lifetime figure and a user cannot reset it.
Writes SUMMARY (key: value lines) and prints it. The command's stdout and stderr pass through. Exits with its status.

VmHWM is the last sample's value (the kernel keeps the high-water mark); VmSwap and VmRSS are maxima over samples. A
descendant is reported as the largest VmSwap and VmRSS seen in any one of them, with its name.
"""
import os
import subprocess
import sys
import time

PERIOD = 0.5


def swap_mb():
    m = {}
    with open("/proc/meminfo") as f:
        for line in f:
            k, v = line.split(":", 1)
            m[k] = int(v.split()[0])
    return (m["SwapTotal"] - m["SwapFree"]) // 1024


def status(pid):
    out = {}
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                k, _, v = line.partition(":")
                if k in ("Name", "VmRSS", "VmHWM", "VmSwap"):
                    out[k] = v.strip() if k == "Name" else int(v.split()[0])
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return None
    return out


def descendants(root):
    kids = {}
    for d in os.listdir("/proc"):
        if not d.isdigit():
            continue
        try:
            with open(f"/proc/{d}/stat") as f:
                s = f.read()
            ppid = int(s[s.rindex(")") + 2:].split()[1])
        except (FileNotFoundError, ProcessLookupError, ValueError, IndexError):
            continue
        kids.setdefault(ppid, []).append(int(d))
    out, todo = [], [root]
    while todo:
        p = todo.pop()
        for c in kids.get(p, []):
            out.append(c)
            todo.append(c)
    return out


def slice_dir():
    try:
        with open("/proc/self/cgroup") as f:
            rel = f.read().strip().split(":", 2)[2]
    except (OSError, IndexError):
        return None
    d = "/sys/fs/cgroup" + rel
    # the user slice: user-<uid>.slice holds memory.high on this box
    while d != "/sys/fs/cgroup" and not os.path.basename(d).startswith("user-"):
        d = os.path.dirname(d)
    return d if os.path.basename(d).startswith("user-") else None


def read_int(path):
    try:
        with open(path) as f:
            return int(f.read().strip())
    except (OSError, ValueError):
        return None


def main():
    if len(sys.argv) < 4 or sys.argv[2] != "--":
        sys.exit(__doc__)
    summary, cmd = sys.argv[1], sys.argv[3:]
    sl = slice_dir()
    s0 = swap_mb()
    sl_swap0 = read_int(f"{sl}/memory.swap.current") if sl else None
    sl_cur0 = read_int(f"{sl}/memory.current") if sl else None
    t0 = time.time()
    p = subprocess.Popen(cmd)
    hwm = rss_max = swp_max = 0
    child = {"name": "", "rss": 0, "swap": 0}
    sys_max = s0
    sl_cur_max = sl_swap_max = 0
    n = 0
    while True:
        st = status(p.pid)
        if st:
            hwm = st.get("VmHWM", hwm)
            rss_max = max(rss_max, st.get("VmRSS", 0))
            swp_max = max(swp_max, st.get("VmSwap", 0))
        for c in descendants(p.pid):
            cs = status(c)
            if cs and (cs.get("VmSwap", 0) > child["swap"] or cs.get("VmRSS", 0) > child["rss"]):
                child = {"name": cs.get("Name", "?"), "rss": max(child["rss"], cs.get("VmRSS", 0)),
                         "swap": max(child["swap"], cs.get("VmSwap", 0))}
        sys_max = max(sys_max, swap_mb())
        if sl:
            sl_cur_max = max(sl_cur_max, read_int(f"{sl}/memory.current") or 0)
            sl_swap_max = max(sl_swap_max, read_int(f"{sl}/memory.swap.current") or 0)
        n += 1
        if p.poll() is not None:
            break
        time.sleep(PERIOD)
    rc = p.returncode
    s1 = swap_mb()
    sl_swap1 = read_int(f"{sl}/memory.swap.current") if sl else None
    lines = [
        f"command: {' '.join(cmd)}",
        f"exit: {rc}",
        f"wall_s: {time.time() - t0:.1f}",
        f"samples: {n} (every {PERIOD} s)",
        f"system_swap_before_mb: {s0}",
        f"system_swap_after_mb: {s1}",
        f"system_swap_delta_mb: {s1 - s0}",
        f"system_swap_peak_mb: {sys_max}",
        f"proc_VmHWM_mb: {hwm // 1024}",
        f"proc_VmRSS_max_mb: {rss_max // 1024}",
        f"proc_VmSwap_peak_mb: {swp_max // 1024}",
        f"descendant_max: {child['name'] or '-'} VmRSS {child['rss'] // 1024} MB, VmSwap {child['swap'] // 1024} MB",
    ]
    if sl:
        lines += [
            f"user_slice: {sl} (memory.high {open(sl + '/memory.high').read().strip()})",
            f"user_slice_memory_current_before_mb: {sl_cur0 // 2**20}",
            f"user_slice_memory_current_peak_mb: {sl_cur_max // 2**20}",
            f"user_slice_swap_before_mb: {sl_swap0 // 2**20}",
            f"user_slice_swap_after_mb: {sl_swap1 // 2**20}",
            f"user_slice_swap_peak_mb: {sl_swap_max // 2**20}",
        ]
    text = "\n".join(lines) + "\n"
    with open(summary, "w") as f:
        f.write(text)
    sys.stdout.write("== swap_sampler\n" + text)
    sys.exit(rc)


if __name__ == "__main__":
    main()
