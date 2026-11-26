#!/usr/bin/env python3
"""The Mac memory instrument: a served model's memory through two long cold prompts, and the machine's memory around it (macOS).

One fresh `serve` per cell at serve's defaults except -backend and the cell's flags (the method of
docs/measurements/mem0-2026-10/README.md). Request 1 is a cold prompt of about 20,200 tokens (this repository's text at PIN, cut
to a character budget; a tower cell sends one 1920x1080 image). Request 2 goes --gap seconds after request 1 ends, with a
different prefix of the same length, so it prefills afresh: 200 is served, 503 is the server refusing (its body is kept), no
status is a hang (--req-timeout) or the server gone. Recorded per cell, in <out>/cell-<cell>.json:
- at five points (before, loaded, after request 1, --gap later, after request 2): the load average; `hostmem -v` and the two
  figures it names, available (the fit guard's) and reclaimable (a timed run's), in MiB; system swap used; and vm_stat's wired
  and file-backed pages with their change since `before`. phys_footprint does not count a mapped weight file's clean pages or
  the GPU working set, which show as file-backed and wired pages;
- every --interval through both requests: the server's own task memory (`hostmem -pid`), system swap, wired and file-backed
  pages, and their peaks;
- the server's guard lines (swap guard, fit:, KV plan:), each with the time it was first seen.
Nothing is graded. The start rule (system swap, the 1-minute load) is checked before each cell loads; a refusal writes the
cell's record and exits 75, the night queue's retry code.

  python3 scripts/mem_darwin.py --serve <serve binary> --out <dir> --repo <checkout holding PIN> --cell metal:1.5b [--cell ...]
"""
import argparse
import base64
import json
import os
import re
import signal
import struct
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import zlib

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PIN = "e42db592"
SOURCES = ["CLAUDE.md", "docs/tasks/task-kv-architecture-2026-10.md", "docs/completed/task-forward-architecture-2026-10.md", "RELEASING.md",
           "docs/server.md"]
CHARS = int(20000 * 3.6)
HOME = os.path.expanduser("~")
CELLS = {  # name: (serve -model path, extra flags)
    "cpu:1.5b": (f"{HOME}/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf", ["-ctx", "32768"]),
    "metal:1.5b": (f"{HOME}/models/qwen2.5-coder-1.5b-instruct-q4_k_m.int4.metal.v18.giw", ["-ctx", "32768"]),
    "metal:e2b": (f"{HOME}/models/gemma-4-e2b-gguf/gemma-4-E2B_q4_0-it.int4.metal.v18.giw", ["-ctx", "32768"]),
    "metal:7b": (f"{HOME}/models/qwen2.5-7b-instruct-q4_k_m.int4.metal.v18.giw", ["-ctx", "32768"]),
    "metal:qwen35-9b": (f"{HOME}/models/Qwen3.5-9B-Q4_K_M.int4.metal.v18.giw", ["-ctx", "32768"]),
    "tower:qwen3vl-2b": (f"{HOME}/models/qwen3-vl-2b-instruct", ["-vision", f"{HOME}/models/qwen3-vl-2b-instruct"]),
}
PORT = 18297
MiB = 1 << 20
RETRY_RC = 75
GUARD_RE = re.compile(r"swap guard|^\s*fit:|^\s*KV plan:", re.I)
HEARTBEAT_S = 30


def stamp():
    return time.strftime("%H:%M:%S")


def log(msg):
    print(f"[{stamp()}] {msg}", flush=True)


# ---------------------------------------------------------------- parsing (pure)

def parse_loadavg(text):
    """`sysctl -n vm.loadavg` ("{ 3.66 4.16 4.05 }") as [load1, load5, load15], or None."""
    v = re.findall(r"[\d.]+", text or "")
    return [float(x) for x in v[:3]] if len(v) >= 3 else None


def parse_swap_used_mib(text):
    """System swap used in MiB from `sysctl -n vm.swapusage`, or None."""
    m = re.search(r"used = ([\d.]+)M", text or "")
    return float(m.group(1)) if m else None


def parse_vm_stat(text):
    """vm_stat's page counts as MiB: wired, file_backed, free, compressor (pages occupied by the compressor); None when the
    page size or a count is missing."""
    ps = re.search(r"page size of (\d+) bytes", text or "")
    keys = {"wired": "Pages wired down", "file_backed": "File-backed pages", "free": "Pages free",
            "compressor": "Pages occupied by compressor"}
    got = {k: re.search(re.escape(v) + r":\s+(\d+)", text or "") for k, v in keys.items()}
    if not ps or not all(got.values()):
        return None
    return {f"{k}_MiB": round(int(m.group(1)) * int(ps.group(1)) / MiB, 1) for k, m in got.items()}


def parse_task(text):
    """`hostmem -pid N`'s line as {field: bytes}."""
    return {k: int(v) for k, v in re.findall(r"(\w+) (\d+)", text or "")}


def int_or_none(text):
    try:
        return int((text or "").strip())
    except ValueError:
        return None


def guard_lines(lines):
    """The server log lines that say what its guards decided: swap guard, the fit banner, the KV plan."""
    return [l.rstrip() for l in lines if GUARD_RE.search(l)]


def deltas(p, base):
    """p's wired and file-backed MiB less base's, rounded; a missing reading gives None."""
    out = {}
    for k in ("wired_MiB", "file_backed_MiB"):
        a, b = p.get(k), base.get(k)
        out[k.replace("_MiB", "_delta_MiB")] = round(a - b, 1) if a is not None and b is not None else None
    return out


def second_prompt(text):
    """The same text with its halves swapped: the same length, a different prefix, so no KV is reused from the first."""
    half = len(text) // 2
    return text[half:] + "\n\n" + text[:half]


def png(w, h, grid=120):
    """A w x h RGB PNG of gradients and a grid every `grid` pixels, deterministic, written with zlib alone."""
    rows = bytearray()
    for y in range(h):
        rows.append(0)
        for x in range(w):
            g = 255 if (x % grid < 3 or y % grid < 3) else 0
            rows += bytes(((x * 255) // w, (y * 255) // h, g))
    chunk = lambda t, d: struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xffffffff)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)) + \
        chunk(b"IDAT", zlib.compress(bytes(rows), 6)) + chunk(b"IEND", b"")


def messages(cell, text, second):
    """The chat messages for a cell's request; the second differs in its prefix (the text) or its image (the grid)."""
    if cell.startswith("tower:"):
        img = base64.b64encode(png(1920, 1080, 97 if second else 120)).decode()
        return [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64," + img}},
                                             {"type": "text", "text": "In one word, what is in this picture?"}]}]
    return [{"role": "system", "content": second_prompt(text) if second else text},
            {"role": "user", "content": "In one word, what is this text about?" if second else "In one word, what is this document about?"}]


# ---------------------------------------------------------------- the machine

def _out(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=60).stdout
    except (OSError, subprocess.TimeoutExpired):
        return ""


class Box:
    """This Mac's readings. Every reading is None (or "") when its command fails; the tests replace the class."""

    def __init__(self, hostmem):
        self.hostmem_bin = hostmem

    def loadavg(self):
        return parse_loadavg(_out(["sysctl", "-n", "vm.loadavg"]))

    def swap_used_mib(self):
        return parse_swap_used_mib(_out(["sysctl", "-n", "vm.swapusage"]))

    def pages(self):
        return parse_vm_stat(_out(["vm_stat"])) or {}

    def hostmem(self, *args):
        return _out([self.hostmem_bin, *args]).strip()


def point(box, label, base=None):
    """The machine at one point of a cell; with base (the `before` point), the wired and file-backed change since then."""
    p = {"label": label, "t": stamp(), "loadavg": box.loadavg(), "swap_used_MiB": box.swap_used_mib(), "hostmem_v": box.hostmem("-v"),
         "available_MiB": int_or_none(box.hostmem("-mb")), "reclaimable_MiB": int_or_none(box.hostmem("-reclaimable", "-mb")),
         **box.pages()}
    if base is not None:
        p.update(deltas(p, base))
    return p


def start_refusal(box, max_swap_mib, max_load):
    """Why the start rule refuses a cell now, or None. A negative limit waives that term; an unreadable reading refuses."""
    sw, la = box.swap_used_mib(), box.loadavg()
    if max_swap_mib >= 0 and (sw is None or sw > max_swap_mib):
        return f"system swap {sw} MiB over the start rule's {max_swap_mib:g}"
    if max_load >= 0 and (la is None or la[0] > max_load):
        return f"load1 {la[0] if la else None} over the start rule's {max_load:g}"
    return None


# ---------------------------------------------------------------- the server

def wait_ready(port, proc, limit=900):
    t0 = time.time()
    while time.time() - t0 < limit:
        if proc.poll() is not None:
            return False
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=2).read()
            return True
        except Exception:
            time.sleep(1)
    return False


def request(port, msgs, timeout, proc):
    """One chat completion, max_tokens 8 at temperature 0: status, wall seconds and usage; a refusal keeps its body, a hang or
    a dropped connection its error and whether the server has exited."""
    body = {"model": "m", "max_tokens": 8, "temperature": 0, "messages": msgs}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.loads(r.read())
            return {"status": r.status, "s": round(time.time() - t0, 1), "usage": data.get("usage"),
                    "text": data["choices"][0]["message"]["content"]}
    except urllib.error.HTTPError as e:
        return {"status": e.code, "s": round(time.time() - t0, 1), "body": e.read().decode(errors="replace")[:500]}
    except Exception as e:
        return {"status": None, "s": round(time.time() - t0, 1), "error": repr(e), "server_exit": proc.poll()}


class Sampler(threading.Thread):
    """Every interval: the server's task memory, system swap, wired and file-backed pages, and any new guard line in its log.
    A heartbeat line every HEARTBEAT_S seconds."""

    def __init__(self, box, pid, logpath, interval, phase):
        super().__init__(daemon=True)
        self.box, self.pid, self.logpath, self.interval, self.phase = box, pid, logpath, interval, phase
        self.samples, self.guard, self.stop = [], [], threading.Event()
        self.seen = 0

    def read_guard(self):
        try:
            with open(self.logpath, errors="replace") as f:
                lines = f.read().splitlines()
        except OSError:
            return
        for l in guard_lines(lines[self.seen:]):
            self.guard.append({"t": stamp(), "line": l})
        self.seen = len(lines)

    def sample_once(self):
        """One pass: any new guard line, then one sample labelled with the phase the caller has set."""
        self.read_guard()
        s = {"t": stamp(), "phase": self.phase[0], **parse_task(self.box.hostmem("-pid", str(self.pid))),
             "swap_used_MiB": self.box.swap_used_mib(), **self.box.pages()}
        self.samples.append(s)
        return s

    def run(self):
        last = time.time()
        while not self.stop.is_set():
            s = self.sample_once()
            if time.time() - last >= HEARTBEAT_S:
                last = time.time()
                fp = s.get("phys_footprint")
                log(f"  {self.phase[0]}: {len(self.samples)} samples, swap {s['swap_used_MiB']} MiB, footprint "
                    f"{round(fp / MiB) if fp else None} MiB, wired {s.get('wired_MiB')} MiB")
            self.stop.wait(self.interval)
        self.read_guard()


def peaks(samples, before):
    out = {}
    for k in ("phys_footprint", "compressed", "compressed_peak", "resident"):
        v = [s[k] for s in samples if k in s]
        out[f"{k}_MiB"] = round(max(v) / MiB) if v else None
    for k in ("swap_used_MiB", "wired_MiB", "file_backed_MiB"):
        v = [s[k] for s in samples if s.get(k) is not None]
        out[k] = max(v) if v else None
    out.update({f"{k}_from_before": round(out[k] - before[k], 1) if out[k] is not None and before.get(k) is not None else None
                for k in ("swap_used_MiB", "wired_MiB", "file_backed_MiB")})
    return out


def cell(name, serve, box, text, port, out, gap=60.0, interval=1.0, req_timeout=900.0, settle=3.0, path=None, extra=None):
    """One cell: load a fresh server, settle, two requests gap seconds apart, the readings. Returns the record, also when the
    server did not come up (`void`, with its log's tail). path and extra replace the cell's model and flags."""
    if path is None:
        path, extra = CELLS[name]
    extra = list(extra or [])
    backend = "cpu" if name.startswith("cpu:") else "metal"
    logpath = os.path.join(out, f"serve-{name.replace(':', '-')}.log")
    before = point(box, "before")
    rec = {"cell": name, "model": path, "backend": backend, "flags": extra, "gap_s": gap, "points": [before]}
    logf = open(logpath, "w")
    t_load = time.time()
    log(f"cell {name}: loading {path}")
    proc = subprocess.Popen([serve, "-backend", backend, "-model", f"m={path}", "-addr", f"127.0.0.1:{port}", *extra],
                            stdout=logf, stderr=subprocess.STDOUT, start_new_session=True)
    sampler = None
    try:
        if not wait_ready(port, proc):
            rec["void"] = "the server did not come up" + (f" (exit {proc.poll()})" if proc.poll() is not None else "")
            return rec
        rec["load_s"] = round(time.time() - t_load, 1)
        time.sleep(settle)
        rec["points"].append(point(box, "loaded", before))
        rec["task_loaded"] = parse_task(box.hostmem("-pid", str(proc.pid)))
        phase = ["request 1"]  # set before the thread starts, which may take its first sample at once
        sampler = Sampler(box, proc.pid, logpath, interval, phase)
        sampler.start()
        log(f"cell {name}: request 1")
        rec["request1"] = request(port, messages(name, text, False), req_timeout, proc)
        rec["points"].append(point(box, "after request 1", before))
        phase[0] = "gap"
        log(f"cell {name}: request 1 {rec['request1']['status']} in {rec['request1']['s']} s; {gap:g} s gap")
        time.sleep(gap)
        rec["points"].append(point(box, f"{gap:g} s after request 1", before))
        phase[0] = "request 2"
        rec["request2"] = request(port, messages(name, text, True), req_timeout, proc)
        log(f"cell {name}: request 2 {rec['request2']['status']} in {rec['request2']['s']} s")
        rec["server_alive_after"] = proc.poll() is None
        rec["points"].append(point(box, "after request 2", before))
        return rec
    finally:
        if sampler is not None:
            sampler.stop.set()
            sampler.join()
            rec["samples"] = sampler.samples
            rec["peaks"] = peaks(sampler.samples, before)
            rec["guard_lines"] = sampler.guard
        try:
            os.killpg(proc.pid, signal.SIGTERM)
            proc.wait(timeout=30)
        except ProcessLookupError:
            pass
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
        logf.close()
        with open(logpath, errors="replace") as f:
            lines = f.read().splitlines()
        if sampler is None:
            rec["guard_lines"] = [{"t": None, "line": l} for l in guard_lines(lines)]
        if rec.get("void"):
            rec["log_tail"] = lines[-20:]


def preamble(repo):
    text = ""
    for p in SOURCES:
        text += subprocess.run(["git", "-C", repo, "show", f"{PIN}:{p}"], capture_output=True, text=True, check=True).stdout
        if len(text) >= CHARS:
            break
    return text[:CHARS]


def build_hostmem(out):
    dst = os.path.join(out, "hostmem")
    r = subprocess.run(["go", "build", "-o", dst, "./cmd/hostmem"], cwd=REPO, capture_output=True, text=True)
    if r.returncode:
        sys.exit(f"mem_darwin: go build ./cmd/hostmem failed in {REPO}: {r.stderr.strip()}")
    return dst


def main(argv=None, box=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--serve", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--repo", required=True, help=f"a checkout holding {PIN}, for the prompt's text")
    ap.add_argument("--cell", action="append", required=True, choices=sorted(CELLS))
    ap.add_argument("--hostmem", help="a built cmd/hostmem (default: built from this checkout into --out)")
    ap.add_argument("--gap", type=float, default=60, help="seconds between request 1's end and request 2")
    ap.add_argument("--interval", type=float, default=1.0, help="seconds between samples")
    ap.add_argument("--req-timeout", type=float, default=900, help="a request still running after this is recorded as a hang")
    ap.add_argument("--max-swap-mib", type=float, default=2048, help="start rule: system swap at most this (negative waives it)")
    ap.add_argument("--max-load", type=float, default=4, help="start rule: 1-minute load at most this (negative waives it)")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    if box is None:
        box = Box(a.hostmem or build_hostmem(a.out))
    text = None
    for i, name in enumerate(a.cell):
        f = os.path.join(a.out, f"cell-{name.replace(':', '-')}.json")
        why = start_refusal(box, a.max_swap_mib, a.max_load)
        if why:
            with open(f, "w") as fh:
                json.dump({"cell": name, "refused": why, "start_rule": {"max_swap_mib": a.max_swap_mib, "max_load": a.max_load}}, fh, indent=1)
            log(f"cell {name}: REFUSED: {why}")
            return RETRY_RC
        text = text if text is not None or name.startswith("tower:") else preamble(a.repo)
        t0 = time.time()
        rec = cell(name, a.serve, box, text, PORT + i, a.out, a.gap, a.interval, a.req_timeout)
        rec["start_rule"] = {"max_swap_mib": a.max_swap_mib, "max_load": a.max_load}
        rec["cell_wall_s"] = round(time.time() - t0, 1)
        with open(f, "w") as fh:
            json.dump(rec, fh, indent=1)
        print(json.dumps({k: v for k, v in rec.items() if k != "samples"}, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
