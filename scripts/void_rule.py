#!/usr/bin/env python3
"""The heavy-load void rule for a timed cell: ONE definition that bench_peer.py, night.py and every shell runner call.

A cell may START only if both hold:
  (a) this box's available memory (the reading the fit guard decides loads by: on macOS memory_pressure's free percentage of
      physical memory less 500 MiB, on Linux MemAvailable; scripts/hostmem.py) is at least the cell's fit need, with no margin;
  (b) the 1-minute load average is not above half the core count (the cap is a parameter, 4.0 on the 8-core Mac).
A cell is VOID if system swap grows at all between its start and its end. The reclaimable reading is no longer a void term; it is
recorded, with load, available memory and swap, at the start and the end of every cell (bench_peer.py per cell, a shell runner through
the commands below, night.py per job as information only: the cell is the unit of the swap void), so a record taken under this rule can
be told from one taken under the old +2 GB rule.

  python3 scripts/void_rule.py start --need-mb N [--load-cap X] [--json-out F]   exit 0 clean, 75 void (the reason on stderr, as `void: ...`)
  python3 scripts/void_rule.py watch --json-out P --ready-file R [--poll-s 1]       samples swap until SIGTERM, writes the peak; R appears once the baseline is taken
  python3 scripts/void_rule.py end --start-json F [--peak-json P] [--json-out G]   exit 0, or 76 when swap grew, by the sampled peak when given
  python3 scripts/void_rule.py snapshot                                            prints the four readings as JSON

Both print the snapshot as one JSON line on stdout. 75 matches night.py's RETRY_RC, so a runner that exits with it is re-queued.
"""
import json
import os
import re
import signal
import subprocess
import sys
import threading
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


def swap_io_mb():
    """(swap-ins, swap-outs) since boot, MB: darwin's vm_stat, Linux's /proc/vmstat at 4 KiB pages; None for what neither reads. Both
    count the whole system, as macOS has no per-process counter."""
    try:
        if sys.platform == "darwin":
            text = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=5).stdout
            page = re.search(r"page size of (\d+) bytes", text)
            if not page:
                return None, None

            def read(name):
                m = re.search(rf"^{name}:\s+(\d+)\.", text, re.M)
                return int(m.group(1)) * int(page.group(1)) / (1 << 20) if m else None
            return read("Swapins"), read("Swapouts")
        counts = dict(line.split() for line in open("/proc/vmstat"))
        return tuple(int(counts[k]) * 4096 / (1 << 20) if k in counts else None for k in ("pswpin", "pswpout"))
    except Exception:
        return None, None


def _load1():
    try:
        return os.getloadavg()[0]
    except OSError:
        return None


def snapshot():
    """The readings a cell records at its start and its end; None for one that cannot be read (an unreadable core count stays unknown, so
    start_void skips the load term instead of imposing a cap of 0.5). swapin_mb and swapout_mb are cumulative since boot, for the
    swap-in and swap-out bars."""
    swapin, swapout = swap_io_mb()
    return {"t": time.strftime("%F %T"), "load1": _load1(), "ncores": os.cpu_count(),
            "available_mb": hostmem.available_mb(), "reclaimable_mb": hostmem.reclaimable_mb(), "swap_mb": swap_used_mb(),
            "swapin_mb": swapin, "swapout_mb": swapout}


def load_cap(ncores):
    """Half the cores: the cap a cell's start may not exceed."""
    return ncores / 2


def start_void(snap, need_mb, cap=None):
    """The reason a cell starting in state snap is void, or None. A guard fails closed: a load or available-memory reading that cannot be
    read voids the start, with that reason (an unreadable core count with no explicit cap is the one skipped term: the cap cannot be
    computed, and the callers say so). need_mb None or 0 means the cell loads no model, so the memory clause does not apply."""
    ncores = snap.get("ncores")
    if cap is None and ncores:
        cap = load_cap(ncores)
    # an unknown core count with no explicit cap is an unknown: the load term is skipped (callers say so), not a cap of 0.5
    if cap is not None:
        if snap.get("load1") is None:
            return "load average unreadable"
        if snap["load1"] > cap:
            return (f"load {snap['load1']:.2f} above half of {ncores} cores" if ncores and cap == ncores / 2
                    else f"load {snap['load1']:.2f} above the cap {cap:g}")
    if need_mb:
        avail = snap.get("available_mb")
        if avail is None:
            return "available memory unreadable"
        if avail < need_mb:
            return f"available memory {avail} MB below the fit need {need_mb:.0f} MB"
    return None


def swap_verdict(growth_mb, swapin_mb, swapout_mb, env=None):
    """The reason a cell whose swap grew by growth_mb (the PEAK over the cell where it was sampled), with swapin_mb / swapout_mb of
    swap-in and swap-out, is void, or None. With no bar set, any growth voids (the zero bar). BENCH_SWAP_VOID_MB replaces it;
    BENCH_SWAPIN_VOID_MB / BENCH_SWAPOUT_VOID_MB replace it with a swap-in / swap-out bar (nobara's, peer-vetted 2026-10-08: idle pages
    pushed out under page-cache pressure grew swap by 1.9 to 46 MB beside a 12 to 14 GB load with 58 GB free, which slows nothing being
    timed; a never-swap gate takes both at 0). The bar in force fails closed: a reading it needs that is None voids the cell."""
    env = os.environ if env is None else env
    lim = env.get("BENCH_SWAP_VOID_MB", "").strip()
    inlim = env.get("BENCH_SWAPIN_VOID_MB", "").strip()
    outlim = env.get("BENCH_SWAPOUT_VOID_MB", "").strip()
    if not lim and not inlim and not outlim:
        lim = "0"
    if lim:
        if growth_mb is None:
            return "swap unreadable: void"
        if growth_mb > float(lim):
            return f"swap grew +{growth_mb:.1f} MB during the cell (> {float(lim):.0f} MB): void"
    if inlim:
        if swapin_mb is None:
            return "swap-in count unreadable: void"
        if swapin_mb > float(inlim):
            return f"swap-ins of {swapin_mb:.2f} MB during the cell (> {float(inlim):.0f} MB): void"
    if outlim:
        if swapout_mb is None:
            return "swap-out count unreadable: void"
        if swapout_mb > float(outlim):
            return f"swap-outs of {swapout_mb:.2f} MB during the cell (> {float(outlim):.0f} MB): void"
    return None


def swap_void(start, end, peak_mb=None, env=None):
    """The reason a cell whose readings were `start` and `end` is void on swap, or None. The growth is the larger of the end against the
    start and the sampled peak (SwapSampler, in absolute MB) against the start: a spike freed again before the end still counts, and
    the peak is read against THIS cell's start reading, whatever baseline the sampler began from."""
    s, e = start.get("swap_mb"), end.get("swap_mb")
    grows = [v - s for v in (e, peak_mb) if v is not None and s is not None]
    growth = max(max(grows), 0.0) if grows else None

    def delta(k):
        a, b = start.get(k), end.get(k)
        return None if a is None or b is None else b - a
    return swap_verdict(growth, delta("swapin_mb"), delta("swapout_mb"), env)


class SwapSampler:
    """Swap in use, sampled while a cell runs, and the PEAK growth over its first reading. A spike that is freed again before the cell
    ends (a process's pages swapped out and back in) is invisible to the start and end readings and counts here. sample() takes one
    reading; start() polls every poll_s on a thread; stop() ends it and returns result()."""

    def __init__(self, read=None, poll_s=1.0, base=None):
        """base: the reading growth is measured from when the caller already has one (a cell's start reading); None takes the first
        sample as the baseline."""
        self.read, self.poll_s = read or (lambda: swap_used_mb()), poll_s
        self.base, self.peak = base, base
        self._stop = threading.Event()
        self._thread = None

    def sample(self):
        v = self.read()
        if v is None:
            return
        if self.base is None:
            self.base, self.peak = v, v
        self.peak = max(self.peak, v)

    def start(self):
        self.sample()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def _run(self):
        while not self._stop.wait(self.poll_s):
            self.sample()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        self.sample()
        return self.result()

    def result(self):
        if self.base is None:
            return {"base_mb": None, "peak_mb": None, "peak_growth_mb": None}
        return {"base_mb": self.base, "peak_mb": self.peak, "peak_growth_mb": round(self.peak - self.base, 3)}


def watch(json_out, ready_file, sampler, stop):
    """Samples swap until `stop` is set, then writes the result to json_out. The baseline is taken, and ready_file written with it, BEFORE
    this waits: a runner that waits for the ready file cannot stop the watcher ahead of its first reading, and a stop that arrived
    already still leaves a result (stop() takes a last sample)."""
    sampler.sample()
    if ready_file:
        json.dump({"base_mb": sampler.base}, open(ready_file, "w"))
    t = threading.Thread(target=sampler._run, daemon=True)
    t.start()
    stop.wait()
    json.dump(sampler.stop(), open(json_out, "w"))
    t.join(timeout=5)


def main(argv):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("start")
    s.add_argument("--need-mb", type=float, default=0)
    s.add_argument("--load-cap", type=float)
    s.add_argument("--need-source", help='where the need came from; "file size" is recorded as a lower bound (no KV or scratch)')
    s.add_argument("--json-out")
    e = sub.add_parser("end")
    e.add_argument("--start-json", required=True)
    e.add_argument("--peak-json", help="the file `watch` wrote: the peak swap growth sampled during the cell")
    e.add_argument("--json-out")
    w = sub.add_parser("watch", help="sample swap every --poll-s until SIGTERM or SIGINT, then write the peak to --json-out")
    w.add_argument("--json-out", required=True)
    w.add_argument("--ready-file", help="written, with the baseline, once the first reading is taken: wait for it before the cell starts")
    w.add_argument("--poll-s", type=float, default=1.0)
    sub.add_parser("snapshot")
    a = ap.parse_args(argv)
    if a.cmd == "watch":
        done = threading.Event()
        for sig in (signal.SIGTERM, signal.SIGINT):  # before the first reading: a stop that arrives early is not lost
            signal.signal(sig, lambda *_: done.set())
        watch(a.json_out, a.ready_file, SwapSampler(poll_s=a.poll_s), done)
        return 0
    snap = snapshot()
    if a.cmd == "snapshot":
        print(json.dumps(snap))
        return 0
    if a.cmd == "start":
        if not snap.get("ncores") and a.load_cap is None:
            print("core count unreadable: the load term is skipped", file=sys.stderr)
        why = start_void(snap, a.need_mb, a.load_cap)
        snap["need_mb"], snap["void"] = a.need_mb, why
        snap["need_source"], snap["need_is_lower_bound"] = a.need_source, a.need_source == "file size"
    else:
        peak = json.load(open(a.peak_json)).get("peak_mb") if a.peak_json else None
        why = swap_void(json.load(open(a.start_json)), snap, peak)
        snap["void"] = why
    if a.json_out:
        json.dump(snap, open(a.json_out, "w"))
    print(json.dumps(snap))
    if why:
        # lower case on purpose: a runner under night.py that re-runs a void cell in place and succeeds must not trip night.py's VOID
        # marker scan; the exit code (75 at a start, 76 at an end) is the runner's verdict to act on
        print("void: " + why, file=sys.stderr)
        return RETRY_RC if a.cmd == "start" else SWAP_VOID_RC
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
