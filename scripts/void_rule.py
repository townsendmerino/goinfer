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
import math
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
    swap-in and swap-out bars. Swap is read FIRST and t_epoch stamped right after it, before the slow hostmem readings: t_epoch is
    the time of the swap reading, the cell's baseline, which a watcher's samples are compared against (peak_since)."""
    swap = swap_used_mb()
    t_epoch = time.time()
    swapin, swapout = swap_io_mb()
    return {"t": time.strftime("%F %T"), "t_epoch": t_epoch, "load1": _load1(), "ncores": os.cpu_count(),
            "available_mb": hostmem.available_mb(), "reclaimable_mb": hostmem.reclaimable_mb(), "swap_mb": swap,
            "swapin_mb": swapin, "swapout_mb": swapout}


def load_cap(ncores):
    """Half the cores: the cap a cell's start may not exceed."""
    return ncores / 2


def write_json_atomic(path, obj):
    """Writes obj to path through a temp file and a rename, so a reader that sees the file sees all of it."""
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(obj, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def wait_ready(path, timeout_s=10.0, poll_s=0.05):
    """The parsed contents of the ready file `watch` writes once its baseline is taken, waiting up to timeout_s for a whole file; None if
    it never appears."""
    end = time.monotonic() + timeout_s
    while True:
        try:
            return json.load(open(path))
        except (OSError, ValueError):
            if time.monotonic() >= end:
                return None
            time.sleep(poll_s)


def start_void(snap, need_mb, cap=None, allow_unreadable_memory=False):
    """The reason a cell starting in state snap is void, or None. A guard fails closed: a load or available-memory reading that cannot be
    read voids the start, with that reason (an unreadable core count with no explicit cap is the one skipped term: the cap cannot be
    computed, and the callers say so). need_mb None or 0 means the cell loads no model, so the memory clause does not apply."""
    ncores = snap.get("ncores")
    load1, avail = _num(snap.get("load1")), _num(snap.get("available_mb"))
    if cap is None and ncores:
        cap = load_cap(ncores)
    # an unknown core count with no explicit cap is an unknown: the load term is skipped (callers say so), not a cap of 0.5
    if cap is not None:
        if load1 is None:
            return "load average unreadable"
        if load1 > cap:
            return (f"load {load1:.2f} above half of {ncores} cores" if ncores and cap == ncores / 2
                    else f"load {load1:.2f} above the cap {cap:g}")
    if need_mb:
        if avail is None:
            if allow_unreadable_memory:  # BENCH_ALLOW_NO_FREE_MEM=1: the documented override for a box with no memory reading
                return None
            return "available memory unreadable"
        if avail < need_mb:
            return f"available memory {avail} MB below the fit need {need_mb:.0f} MB"
    return None


def _mb(x):
    """x MB for a reason line: one decimal from 1 MB up, else significant digits (a 4 KiB page, 0.0039 MB, must not print as 0.0)."""
    return f"{x:.1f}" if x >= 1 else f"{x:.4g}"


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
    if growth_mb is None:
        return "swap unreadable: void"  # whichever bar is in force, a swap reading that failed voids the cell
    if not lim and not inlim and not outlim:
        lim = "0"
    if lim:
        if growth_mb > float(lim):
            return f"swap grew +{_mb(growth_mb)} MB during the cell (> {float(lim):.0f} MB): void"
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


def swap_void(start, end, peak_mb=None, env=None, peak_required=False):
    """The reason a cell whose readings were `start` and `end` is void on swap, or None. The growth is the larger of the end against the
    start and the sampled peak (SwapSampler, in absolute MB) against the start: a spike freed again before the end still counts, and
    the peak is read against THIS cell's start reading, whatever baseline the sampler began from."""
    given = peak_mb is not None  # a peak that was passed but is not a finite number is an unreadable reading, whether or not required
    s, e, peak_mb = _num(start.get("swap_mb")), _num(end.get("swap_mb")), _num(peak_mb)
    # Fails closed: the start and end readings are both part of the cell's record, so one that failed voids it whatever a sampler saw, and
    # a watcher that was asked for (peak_required) but has no peak voids it too.
    if s is None or e is None or ((peak_required or given) and peak_mb is None):
        growth = None
    else:
        growth = max(max(v - s for v in (e, peak_mb) if v is not None), 0.0)

    def delta(k):
        a, b = _num(start.get(k)), _num(end.get(k))
        return None if a is None or b is None else b - a
    return swap_verdict(growth, delta("swapin_mb"), delta("swapout_mb"), env)


def end_void(start, end, peak_mb=None, peak_required=False, env=None, allow_unreadable_memory=None, say=None):
    """The reason a cell is void at its end, or None: swap (swap_void), then a guard reading the end could not take. The load average
    is part of the record when the start could compute a cap (it had a core count); available memory when the start declared a model
    load (need_mb). allow_unreadable_memory (None: BENCH_ALLOW_NO_FREE_MEM=1 in the environment) skips the memory term, and say, if
    given, is called with the line that tells the run it did. The one function behind `void_rule.py end` and bench_peer's cell check."""
    why = swap_void(start, end, peak_mb, env, peak_required)
    if why:
        return why
    if start.get("ncores") and _num(end.get("load1")) is None:
        return "load average unreadable at the cell's end"
    if start.get("need_mb") and _num(end.get("available_mb")) is None:
        if allow_unreadable_memory is None:
            allow_unreadable_memory = os.environ.get("BENCH_ALLOW_NO_FREE_MEM") == "1"
        if not allow_unreadable_memory:
            return "available memory unreadable at the cell's end"
        if say:
            say("BENCH_ALLOW_NO_FREE_MEM=1: no available-memory reading at the cell's end, so the memory term is skipped for this cell")
    return None


def _num(x):
    """x as a float if it is a finite number, else None (a string, null, a bool, a list, NaN, an infinity: a reading that is not a reading)."""
    return float(x) if isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) else None


def peak_since(samples, t0):
    """The highest swap reading a watcher took at or after epoch time t0, the cell's own start reading. A watcher started before `start`
    saw swap the cell never did, and its last sample before t0 can predate a fall that happened before `start` read swap, so neither
    counts: the baseline is `start`'s reading, and the peak is over what the watcher saw from there on (it takes a last sample at
    stop). None if there is no such sample, if samples is not a list, or a sample is not a pair of numbers: the caller voids the cell."""
    if not isinstance(samples, list) or _num(t0) is None:
        return None
    vals = []
    for item in samples:
        try:
            t, v = _num(item[0]), _num(item[1])
        except (TypeError, IndexError, KeyError):
            return None
        if t is None or v is None:
            return None
        if t >= t0:
            vals.append(v)
    return max(vals) if vals else None


READ_TRIES, READ_DELAY = 20, 0.1  # about two seconds for a file a watcher is still writing; tests shorten them


def read_json_retry(path, tries=None, delay=None, ok=None):
    """The JSON object in path, retrying while it is missing or does not parse (a file a watcher is still writing), and, with ok, while
    ok(obj) is false (a file an earlier watcher left); None if it never parses to an object, else the last object read, ok or not."""
    tries, delay = tries or READ_TRIES, READ_DELAY if delay is None else delay
    last = None
    for i in range(tries):
        try:
            obj = json.load(open(path))
            if isinstance(obj, dict):
                last = obj
                if ok is None or ok(obj):
                    return obj
        except (OSError, ValueError):
            pass
        if i + 1 < tries:
            time.sleep(delay)
    return last


def peak_file_is_final(obj, t0):
    """Whether a watcher's result file is the one for the cell that began at t0: it was written (t_written, stamped at stop) or its
    last sample taken at or after t0. A file written before t0, readable or not, is an earlier watcher's, not yet overwritten; one
    that cannot be placed in time is not final either (the caller waits, then voids with the file's own reason). With no usable t0
    nothing can be placed, so any file is taken."""
    if _num(t0) is None:
        return True
    stamps = []
    samples = obj.get("samples")
    try:
        if isinstance(samples, list) and samples:
            stamps.append(_num(samples[-1][0]))
    except (TypeError, IndexError, KeyError):
        pass
    stamps.append(_num(obj.get("t_written")))
    return any(t is not None and t >= t0 for t in stamps)


class SwapSampler:
    """Swap in use, sampled while a cell runs, and the PEAK growth over its first reading. A spike that is freed again before the cell
    ends (a process's pages swapped out and back in) is invisible to the start and end readings and counts here. sample() takes one
    reading; start() polls every poll_s on a thread; stop() ends it and returns result()."""

    def __init__(self, read=None, poll_s=1.0, base=None):
        """base: the reading growth is measured from when the caller already has one (a cell's start reading); None takes the first
        sample as the baseline."""
        self.read, self.poll_s = read or (lambda: swap_used_mb()), poll_s
        self.base, self.peak = base, base
        self.unreadable = False  # a reading failed after the sampler began: the cell cannot be called clean
        self.n = 0               # readings taken: a sampler that never read has no evidence of a peak
        self.samples = []        # [epoch, MB], every reading (peak_since)
        self._stop = threading.Event()
        self._thread = None

    def sample(self):
        t = time.time()  # stamped BEFORE the read: a slow read must not make a reading older than the cell's start look newer
        try:
            v = self.read()
        except Exception:
            v = None
        self.observe(v, t)

    def observe(self, v, t=None):
        """Folds in a reading taken elsewhere (a job's end reading), stamped t (default: now); None or a non-finite value marks the
        sampler unreadable."""
        v = _num(v)
        if v is None:
            self.unreadable = True
            return
        self.n += 1
        # every reading is kept, repeats included: a repeat after `start` is the evidence that swap stayed where it was (one reading a
        # second is about 10,000 entries over a three-hour cell)
        self.samples.append([time.time() if t is None else t, v])
        if self.base is None:
            self.base, self.peak = v, v
        self.peak = max(self.peak, v)

    def start(self):
        self.sample()
        return self.start_thread()

    def start_thread(self):
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
        t_written = time.time()  # places the file in time even when a lost reading left no sample to place it by (peak_file_is_final)
        if self.base is None or self.n == 0 or self.unreadable:
            # no evidence of a peak, or a reading failed: no numeric peak for a caller to mistake for a clean one
            return {"base_mb": self.base, "peak_mb": None, "peak_growth_mb": None, "unreadable": True, "samples": self.samples,
                    "t_written": t_written}
        return {"base_mb": self.base, "peak_mb": self.peak, "peak_growth_mb": round(self.peak - self.base, 3),
                "unreadable": False, "samples": self.samples, "t_written": t_written}


def watch(json_out, ready_file, sampler, stop):
    """Samples swap until `stop` is set, then writes the result to json_out (atomically). The baseline is taken, and ready_file written
    with it, BEFORE this waits: a runner that waits for the ready file (`wait`) cannot stop the watcher ahead of its first reading, and
    a stop that arrived already still leaves a result (stop() takes a last sample). The sampling thread is the sampler's own, so
    stop() joins it before the result is read."""
    sampler.sample()
    if ready_file:
        write_json_atomic(ready_file, {"base_mb": sampler.base})
    sampler.start_thread()
    stop.wait()
    write_json_atomic(json_out, sampler.stop())


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
    r = sub.add_parser("wait", help="wait for the ready file `watch` writes; exit 77 if it never appears")
    r.add_argument("--ready-file", required=True)
    r.add_argument("--timeout-s", type=float, default=10.0)
    sub.add_parser("snapshot")
    a = ap.parse_args(argv)
    if a.cmd == "watch":
        done = threading.Event()
        for sig in (signal.SIGTERM, signal.SIGINT):  # before the first reading: a stop that arrives early is not lost
            signal.signal(sig, lambda *_: done.set())
        watch(a.json_out, a.ready_file, SwapSampler(poll_s=a.poll_s), done)
        return 0
    if a.cmd == "wait":
        got = wait_ready(a.ready_file, a.timeout_s)
        if got is None:
            print(f"void: the watcher never wrote {a.ready_file}", file=sys.stderr)
            return 77
        print(json.dumps(got))
        return 0
    snap = snapshot()
    if a.cmd == "snapshot":
        print(json.dumps(snap))
        return 0
    if a.cmd == "start":
        if not snap.get("ncores") and a.load_cap is None:
            print("core count unreadable: the load term is skipped", file=sys.stderr)
        allow = os.environ.get("BENCH_ALLOW_NO_FREE_MEM") == "1"
        if allow and a.need_mb and snap.get("available_mb") is None:
            print("BENCH_ALLOW_NO_FREE_MEM=1: no available-memory reading, so the memory term is skipped for this cell", file=sys.stderr)
        why = start_void(snap, a.need_mb, a.load_cap, allow_unreadable_memory=allow)
        snap["need_mb"], snap["void"] = a.need_mb, why
        snap["need_source"], snap["need_is_lower_bound"] = a.need_source, a.need_source == "file size"
    else:
        start = read_json_retry(a.start_json)  # a cell with no readable baseline is void, not a crash
        peak, peak_why = None, None
        if start is None:
            start = {}
            peak_why = "the start file never parsed"
        elif a.peak_json:
            # retried while the watcher is still writing it, and while it is an earlier watcher's file; None if it never parses
            pk = read_json_retry(a.peak_json, ok=lambda o: peak_file_is_final(o, start.get("t_epoch")))
            if pk is None:
                peak_why = "the watcher's peak file never parsed"
            elif pk.get("unreadable"):
                peak_why = "the watcher lost a swap reading" if peak_file_is_final(pk, start.get("t_epoch")) else \
                    "the watcher's peak file is an earlier attempt's"
            else:
                # against the cell's own start reading: the samples from its time on (a watcher started first saw swap the cell never did).
                # No fallback to the pre-start peak_mb: samples that cannot be placed against the start are a void, not a clean cell.
                peak = peak_since(pk.get("samples"), start.get("t_epoch"))
                if peak is None:
                    peak_why = "the watcher's swap samples unusable (none at or after the start, or not numbers, or no start time)"
        why = end_void(start, snap, peak, peak_required=bool(a.peak_json),
                       say=lambda m: print(m, file=sys.stderr))
        if why and peak_why and "swap unreadable" in why:
            why = f"{why} ({peak_why})"
        snap["void"] = why
    if a.json_out:
        write_json_atomic(a.json_out, snap)
    print(json.dumps(snap))
    if why:
        # lower case on purpose: a runner under night.py that re-runs a void cell in place and succeeds must not trip night.py's VOID
        # marker scan; the exit code (75 at a start, 76 at an end) is the runner's verdict to act on
        print("void: " + why, file=sys.stderr)
        return RETRY_RC if a.cmd == "start" else SWAP_VOID_RC
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
