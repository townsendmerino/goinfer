"""A night job's memory, sampled once a second into <rundir>/<job>.mem.tsv (Linux only): the goinfer servers in the job's process
tree, each with its descendants (VmSwap, VmRSS, RssAnon, RssShmem, RssFile, summed, in bytes), system swap in use, MemAvailable, and
the memory.current and memory.events of the user slice the job runs in. A swap-guard trip then names its source (the server's own swap
or the system's) and what the slice held at that moment: docs/measurements/swaptrip-c13-2026-11/README.md is the trip it was missing for.
NIGHT_MEM_SAMPLE_S sets the interval (default 1)."""
import os
import re
import sys
import threading
import time

SERVER = re.compile(r"^goinfer-serve")
FIELDS = ("VmSwap", "VmRSS", "RssAnon", "RssShmem", "RssFile")
COLUMNS = ("t", "servers", *FIELDS, "sys_swap_used", "MemAvailable", "slice", "slice_current", "slice_high", "slice_max", "slice_oom")


def _read(p):
    try:
        with open(p, errors="replace") as f:
            return f.read()
    except OSError:
        return ""


def _status(pid, proc):
    d = {}
    for line in _read(f"{proc}/{pid}/status").splitlines():
        k, _, v = line.partition(":")
        if k in FIELDS and v.split():
            d[k] = int(v.split()[0]) * 1024
    return d


def _children(proc):
    kids = {}
    for p in os.listdir(proc):
        if p.isdigit():
            st = _read(f"{proc}/{p}/stat")
            if st:
                kids.setdefault(int(st.rsplit(")", 1)[1].split()[1]), []).append(int(p))
    return kids


def _tree(root, kids):
    out, todo = [], [root]
    while todo:
        p = todo.pop()
        out.append(p)
        todo += kids.get(p, [])
    return out


def _is_server(pid, proc):
    argv0 = _read(f"{proc}/{pid}/cmdline").split("\0")[0]
    return bool(SERVER.match(os.path.basename(argv0)) or SERVER.match(_read(f"{proc}/{pid}/comm").strip()))


def _slice(cgroup_line):
    """The user slice in a /proc/<pid>/cgroup line (the deepest user-<uid>.slice), or the line's own cgroup when it is in none."""
    path = cgroup_line.strip().split("::", 1)[-1] if "::" in cgroup_line else ""
    parts = path.strip("/").split("/") if path else []
    for i in range(len(parts) - 1, -1, -1):
        if re.fullmatch(r"user-\d+\.slice", parts[i]):
            return "/" + "/".join(parts[: i + 1])
    return path


def sample(root, proc="/proc", cgroup_root="/sys/fs/cgroup", cgroup_line=None):
    """One row: the servers under root (a pid) and their trees, the system's swap and MemAvailable, and the job's slice."""
    kids = _children(proc)
    tree = _tree(root, kids)
    servers = [p for p in tree if _is_server(p, proc)]
    covered = set()
    for s in servers:
        covered.update(_tree(s, kids))
    row = dict.fromkeys(FIELDS, 0)
    for p in covered:
        for k, v in _status(p, proc).items():
            row[k] += v
    mi = {}
    for line in _read(f"{proc}/meminfo").splitlines():
        k, _, v = line.partition(":")
        if v.split():
            mi[k] = int(v.split()[0]) * 1024
    row["sys_swap_used"] = mi.get("SwapTotal", 0) - mi.get("SwapFree", 0)
    row["MemAvailable"] = mi.get("MemAvailable", 0)
    if cgroup_line is None:
        cgroup_line = next((l for l in _read(f"{proc}/{root}/cgroup").splitlines() if l.startswith("0::")), "")
    sl = _slice(cgroup_line)
    ev = {}
    for line in _read(f"{cgroup_root}{sl}/memory.events").splitlines():
        k, _, v = line.partition(" ")
        if v.strip().isdigit():
            ev[k] = int(v)
    cur = _read(f"{cgroup_root}{sl}/memory.current").strip()
    row.update(t=time.strftime("%H:%M:%S"), servers=",".join(str(s) for s in sorted(servers)), slice=sl,
               slice_current=int(cur) if cur.isdigit() else "", slice_high=ev.get("high", ""), slice_max=ev.get("max", ""),
               slice_oom=ev.get("oom", ""))
    return row


class Sampler:
    """Samples root's tree into path every interval seconds on a daemon thread until stop()."""

    def __init__(self, root, path, interval):
        self.root, self.path, self.interval = root, path, interval
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._t.start()
        return self

    def _run(self):
        with open(self.path, "w") as f:
            f.write("\t".join(COLUMNS) + "\n")
            while True:
                try:
                    r = sample(self.root)
                    f.write("\t".join(str(r[c]) for c in COLUMNS) + "\n")
                    f.flush()
                except Exception as e:  # a failed sample is a line in the file, never a failed job
                    f.write(f"# sample failed: {e}\n")
                if self._stop.wait(self.interval):
                    return

    def stop(self):
        self._stop.set()
        self._t.join(timeout=10)


def start(root, path):
    """A running Sampler of the job root into path, or None where there is no /proc (the Mac)."""
    if not sys.platform.startswith("linux"):
        return None
    try:
        interval = float(os.environ.get("NIGHT_MEM_SAMPLE_S", "1"))
    except ValueError:
        interval = 1.0
    return Sampler(root, path, max(interval, 0.05)).start()
