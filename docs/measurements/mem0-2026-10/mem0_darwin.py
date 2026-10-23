#!/usr/bin/env python3
"""MEM0, the Mac half (docs/tasks/task-memory-campaign-2026-10.md §3): the server's memory through one long cold prompt, macOS.

The Mac port of mem0_linux.py, on g40_mem.py's macOS readings. One fresh `serve` per cell at serve's defaults except
-backend and -ctx; one request at temperature 0, max_tokens 8: a cold prompt of about 20,200 tokens (KV9's preamble, this
repository's text at e42db592, cut to a character budget), or, for the tower cell, one 1920x1080 image and a short question.
The server process's phys_footprint (`footprint -p`) and RSS are sampled every 250 ms; readings per cell: after load,
the peak during the request, at the end of the request, and 10 s after, with footprint's categories at each of those
points. Available memory and swap-used are read before and after (context only). Nothing is graded.

  python3 mem0_darwin.py --serve <metal serve binary> --out <dir> --repo <checkout with e42db592> --cell metal:1.5b [--cell ...]
cells: cpu:1.5b (warm its CPU sidecar first: --warm), metal:1.5b, metal:e2b, tower:qwen3vl-2b (Metal), metal:qwen35-9b (KH1-G5)
"""
import argparse, base64, json, os, re, signal, struct, subprocess, threading, time, urllib.request, zlib

PIN = "e42db592"
SOURCES = ["CLAUDE.md", "docs/tasks/task-kv-architecture-2026-10.md", "docs/tasks/task-forward-architecture-2026-10.md", "RELEASING.md", "docs/server.md"]
CHARS = int(20000 * 3.6)
HOME = os.path.expanduser("~")
CELLS = {  # name: (serve -model path, extra flags)
    "cpu:1.5b": (f"{HOME}/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf", ["-ctx", "32768"]),
    "metal:1.5b": (f"{HOME}/models/qwen2.5-coder-1.5b-instruct-q4_k_m.int4.metal.v17.giw", ["-ctx", "32768"]),
    "metal:e2b": (f"{HOME}/models/gemma-4-e2b-gguf/gemma-4-E2B_q4_0-it.int4.metal.v17.giw", ["-ctx", "32768"]),
    "tower:qwen3vl-2b": (f"{HOME}/models/qwen3-vl-2b-instruct", ["-vision", f"{HOME}/models/qwen3-vl-2b-instruct"]),
    "metal:qwen35-9b": (f"{HOME}/models/Qwen3.5-9B-Q4_K_M.int4.metal.v17.giw", ["-ctx", "32768"]),  # KS-H1's KH1-G5
}
MiB = 1 << 20
PAGE = 16384
PORT = 18297


def preamble(repo):
    text = ""
    for p in SOURCES:
        text += subprocess.run(["git", "-C", repo, "show", f"{PIN}:{p}"], capture_output=True, text=True, check=True).stdout
        if len(text) >= CHARS:
            break
    return text[:CHARS]


def png(w, h):
    """A w x h RGB PNG of smooth gradients and a grid, deterministic, written with zlib alone."""
    rows = bytearray()
    for y in range(h):
        rows.append(0)
        for x in range(w):
            g = 255 if (x % 120 < 3 or y % 120 < 3) else 0
            rows += bytes(((x * 255) // w, (y * 255) // h, g))
    chunk = lambda t, d: struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xffffffff)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)) + \
        chunk(b"IDAT", zlib.compress(bytes(rows), 6)) + chunk(b"IEND", b"")


def available():
    out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    pages = sum(int(m.group(1)) for k in ("Pages free", "Pages inactive", "Pages speculative", "Pages purgeable")
                if (m := re.search(k + r":\s+(\d+)", out)))
    return pages * PAGE


def swap_used():
    out = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout
    m = re.search(r"used = ([\d.]+)M", out)
    return float(m.group(1)) * MiB if m else 0.0


UNIT = {"B": 1, "KB": 1 << 10, "MB": 1 << 20, "GB": 1 << 30}


def footprint(pid, categories=False):
    out = subprocess.run(["footprint", "-p", str(pid)], capture_output=True, text=True).stdout
    m = re.search(r"Footprint: ([\d.]+) (B|KB|MB|GB)", out)
    total = float(m.group(1)) * UNIT[m.group(2)] if m else None
    if not categories:
        return total
    cats = {}  # each category's dirty bytes, which the footprint counts
    for line in out.splitlines():
        c = re.match(r"\s*([\d.]+) (B|KB|MB|GB)\s+[\d.]+ (?:B|KB|MB|GB)\s+[\d.]+ (?:B|KB|MB|GB)\s+\d+\s+(.+?)\s*$", line)
        if c and c.group(3) != "TOTAL":
            cats[c.group(3)] = round(float(c.group(1)) * UNIT[c.group(2)] / MiB, 1)
    top = dict(sorted(cats.items(), key=lambda kv: -kv[1])[:8])
    return total, top


def rss(pid):
    out = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)], capture_output=True, text=True).stdout.strip()
    return int(out) * 1024 if out else None


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


def cell(name, binary, text, port, out, interval):
    path, extra = CELLS[name]
    rec = {"cell": name, "model": path, "available_before_MiB": round(available() / MiB), "swap_used_before_MiB": round(swap_used() / MiB)}
    log = open(f"{out}/serve-{name.replace(':', '-')}.log", "w")
    t_load = time.time()
    backend = "cpu" if name.startswith("cpu:") else "metal"
    rec["backend"] = backend
    proc = subprocess.Popen([binary, "-backend", backend, "-model", f"m={path}", "-addr", f"127.0.0.1:{port}", *extra],
                            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    try:
        if not wait_ready(port, proc):
            rec.update(void="the server did not come up")
            return rec
        rec["load_s"] = round(time.time() - t_load, 1)
        time.sleep(3)
        load_fp, load_cats = footprint(proc.pid, True)
        peak = {"fp": 0.0, "rss": 0}
        stop = threading.Event()

        def sample():
            while not stop.is_set():
                f, r = footprint(proc.pid), rss(proc.pid)
                if f:
                    peak["fp"] = max(peak["fp"], f)
                if r:
                    peak["rss"] = max(peak["rss"], r)
                stop.wait(interval)

        th = threading.Thread(target=sample, daemon=True)
        th.start()
        if name.startswith("tower:"):
            img = base64.b64encode(png(1920, 1080)).decode()
            msgs = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64," + img}},
                                                 {"type": "text", "text": "In one word, what is in this picture?"}]}]
        else:
            msgs = [{"role": "system", "content": text}, {"role": "user", "content": "In one word, what is this document about?"}]
        body = {"model": "m", "max_tokens": 8, "temperature": 0, "messages": msgs}
        t0 = time.time()
        req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=3600) as r:
            usage = json.loads(r.read()).get("usage", {})
        wall = time.time() - t0
        stop.set()
        th.join()
        end_fp, end_cats = footprint(proc.pid, True)
        time.sleep(10)
        after_fp, after_cats = footprint(proc.pid, True)
        rec.update(prompt_tokens=usage.get("prompt_tokens"), request_s=round(wall, 1),
                   footprint_MiB={"load": round(load_fp / MiB), "peak": round(peak["fp"] / MiB), "end": round(end_fp / MiB),
                                  "plus10s": round(after_fp / MiB)},
                   peak_rss_MiB=round(peak["rss"] / MiB), categories_MiB={"load": load_cats, "end": end_cats, "plus10s": after_cats},
                   available_after_MiB=round(available() / MiB), swap_used_after_MiB=round(swap_used() / MiB))
        return rec
    finally:
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
        log.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--serve", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--cell", action="append", required=True, choices=sorted(CELLS))
    ap.add_argument("--interval", type=float, default=0.25)
    ap.add_argument("--warm", action="store_true", help="start each cell's server once and stop it when ready, so a sidecar it writes is not in a cell's load reading")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    text = preamble(a.repo)
    port = PORT
    if a.warm:
        for name in a.cell:
            path, extra = CELLS[name]
            proc = subprocess.Popen([a.serve, "-backend", "cpu" if name.startswith("cpu:") else "metal", "-model", f"m={path}",
                                     "-addr", f"127.0.0.1:{port}", *extra], stdout=open(f"{a.out}/warm-{name.replace(':', '-')}.log", "w"),
                                    stderr=subprocess.STDOUT, start_new_session=True)
            print(json.dumps({"warm": name, "ready": wait_ready(port, proc, limit=1200)}), flush=True)
            os.killpg(proc.pid, signal.SIGTERM)
            proc.wait(timeout=60)
            port += 1
    for name in a.cell:
        t0 = time.time()
        print(f"[{time.strftime('%H:%M:%S')}] cell {name} starting", flush=True)
        rec = cell(name, a.serve, text, port, a.out, a.interval)
        rec["cell_wall_s"] = round(time.time() - t0, 1)
        with open(f"{a.out}/cell-{name.replace(':', '-')}.json", "w") as f:
            json.dump(rec, f, indent=1)
        print(json.dumps(rec), flush=True)
        port += 1


if __name__ == "__main__":
    main()
