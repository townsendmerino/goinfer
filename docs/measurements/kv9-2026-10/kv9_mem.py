#!/usr/bin/env python3
"""KV9 step 0: the resident memory of CPU serve with N sessions sharing an agent preamble (exploratory, not timed).

One fresh `serve -backend cpu` per cell (model x preamble length). Session k (k = 1..N) sends the same system preamble
and its own short user message. Session 1 prefills cold. With KV2's CPU copies, sessions 2..N copy the preamble's rows
from session 1 (`usage.prefill_reused_tokens` shows it), and each holds its own copy. After each session the server's
memory is read twice: `footprint` (phys_footprint, which counts compressed pages too) and `ps` RSS.

Before every session a guard reads the Mac's memory. It stops the cell if the next session's KV, with headroom, would
not fit in free+inactive+speculative+purgeable pages, or if swap has grown more than 256 MB since the cell started.
A stopped cell records why; its later sessions are not measured. The guard reads only between sessions, so a cold
prefill can still push the machine into swap mid-request (the 20K cells did, on 2026-10-14).

--settle S waits after each session until the footprint has held within 8 MB for S seconds (at most 6 x S), reading
it every 15 s. The cold prefill's scratch stays in Go's heap after the request, where later sessions' copies land
without moving the footprint; settling lets the runtime return it first, so each later step is the copy's own size.

--pair-start sends sessions 1 and 2 at once. On serve as built, a new conversation whose prompt keeps more of the only
idle session than it discards takes that session over in place (internal/serveapp/sessions.go:pickSession), so
sessions sent one at a time with short tails all land in one session and nothing is copied. Two concurrent requests
each get their own session (a busy session is never taken), and later conversations then copy the preamble from an
idle one (sessionLRU.donorFor). A probe at the end continues conversation 1, to show it kept its own rows.

  python3 kv9_mem.py --serve <binary> --out <dir> [--sessions 4] [--cells 0.5b:6000,1.5b:6000] [--settle 60] [--pair-start]

The preamble is text from this repository at a pinned commit, cut to a character budget; the token count is what
serve reports. Prints one JSON line per measurement and writes <out>/kv9-mem.json.
"""
import argparse
import concurrent.futures
import json
import os
import re
import signal
import subprocess
import sys
import time
import urllib.request

PIN = "e42db592"
SOURCES = ["CLAUDE.md", "docs/tasks/task-kv-architecture-2026-10.md", "docs/tasks/task-forward-architecture-2026-10.md",
           "RELEASING.md", "docs/server.md"]
CHARS_PER_TOKEN = 3.6  # a first guess for this text; the record uses serve's own count

HOME = os.path.expanduser("~")
MODELS = {
    # name: (path, layers, kv heads, head dim) -- Qwen2.5's published configs
    "0.5b": (f"{HOME}/models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf", 24, 2, 64),
    "1.5b": (f"{HOME}/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf", 28, 2, 128),
}
CELLS = [("0.5b", 6000), ("1.5b", 6000), ("0.5b", 20000), ("1.5b", 20000)]
PAGE = 16384


def preamble(tokens, repo):
    text = ""
    for p in SOURCES:
        text += subprocess.run(["git", "-C", repo, "show", f"{PIN}:{p}"], capture_output=True, text=True, check=True).stdout
        if len(text) >= tokens * CHARS_PER_TOKEN:
            break
    return text[: int(tokens * CHARS_PER_TOKEN)]


def available():
    out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    pages = 0
    for key in ("Pages free", "Pages inactive", "Pages speculative", "Pages purgeable"):
        m = re.search(key + r":\s+(\d+)", out)
        pages += int(m.group(1)) if m else 0
    return pages * PAGE


def swap_used():
    out = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout
    m = re.search(r"used = ([\d.]+)M", out)
    return float(m.group(1)) * (1 << 20) if m else 0.0


def footprint(pid):
    out = subprocess.run(["footprint", "-p", str(pid)], capture_output=True, text=True).stdout
    m = re.search(r"Footprint: ([\d.]+) (KB|MB|GB)", out)
    if not m:
        return None
    return float(m.group(1)) * {"KB": 1 << 10, "MB": 1 << 20, "GB": 1 << 30}[m.group(2)]


def rss(pid):
    out = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)], capture_output=True, text=True).stdout.strip()
    return int(out) * 1024 if out else None


def post(port, body, timeout):
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def wait_ready(port, proc, limit=300):
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


def settle(pid, secs, cell, k, records, t_start):
    """Read the footprint every 15 s until it has held within 8 MB for secs seconds, or 6 x secs have passed."""
    t0, last, held = time.time(), footprint(pid), 0
    while held < secs and time.time() - t0 < 6 * secs:
        time.sleep(15)
        fp = footprint(pid)
        held = held + 15 if fp is not None and last is not None and abs(fp - last) <= (8 << 20) else 0
        last = fp
    emit({**cell, "event": "settled", "session": k, "footprint": last, "rss": rss(pid),
          "waited_s": round(time.time() - t0), "elapsed_s": round(time.time() - t_start)}, records)


def emit(rec, records):
    records.append(rec)
    print(json.dumps(rec), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--serve", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--logs", default=f"{HOME}/goinfer-logs/kv9-2026-10")
    ap.add_argument("--sessions", type=int, default=4)
    ap.add_argument("--repo", default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--cells", default="", help="model:tokens,... (default: all four)")
    ap.add_argument("--settle", type=int, default=0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--pair-start", action="store_true")
    a = ap.parse_args()
    cells = [(c.split(":")[0], int(c.split(":")[1])) for c in a.cells.split(",")] if a.cells else CELLS
    # SIGTERM must take the running server down too: the default handler skips the finally below
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(SystemExit(143)))
    os.makedirs(a.out, exist_ok=True)
    os.makedirs(a.logs, exist_ok=True)
    records = []
    t_start = time.time()
    for ci, (name, target) in enumerate(cells):
        path, layers, kvh, hd = MODELS[name]
        bpp = layers * 2 * kvh * hd * 4  # f32 K and V per position
        text = preamble(target, a.repo)
        port = 18431 + ci
        log = open(f"{a.logs}/serve-{name}-{target}{a.tag}.log", "w")
        proc = subprocess.Popen([a.serve, "-backend", "cpu", "-model", f"m={path}", "-addr", f"127.0.0.1:{port}",
                                 "-kv-sessions", str(a.sessions), "-max-concurrent", str(a.sessions)],
                                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        cell = {"model": name, "target_tokens": target, "bytes_per_position": bpp, "pid": proc.pid}
        try:
            if not wait_ready(port, proc):
                emit({**cell, "event": "not ready", "elapsed_s": round(time.time() - t_start)}, records)
                continue
            time.sleep(2)
            swap0 = swap_used()
            emit({**cell, "event": "loaded", "footprint": footprint(proc.pid), "rss": rss(proc.pid),
                  "available": available(), "swap_used": swap0, "elapsed_s": round(time.time() - t_start)}, records)
            def body_for(k):
                return {"model": "m", "max_tokens": 8, "temperature": 0, "messages": [
                    {"role": "system", "content": text},
                    {"role": "user", "content": f"Session {k}: in one word, what is this document about?"}]}

            def record(k, r):
                u = r.get("usage", {})
                emit({**cell, "event": "session", "session": k, "prompt_tokens": u.get("prompt_tokens"),
                      "completion_tokens": u.get("completion_tokens"), "reused": u.get("prefill_reused_tokens"),
                      "footprint": footprint(proc.pid), "rss": rss(proc.pid), "available": available(),
                      "swap_grown": swap_used() - swap0, "elapsed_s": round(time.time() - t_start)}, records)

            replies = {}
            first = 1
            if a.pair_start:
                need = 2 * (int(target * bpp * 1.3) + (512 << 20))
                if available() < need:
                    emit({**cell, "event": "stopped", "session": 1, "available": available(), "need": need,
                          "swap_grown": swap_used() - swap0, "elapsed_s": round(time.time() - t_start)}, records)
                    continue
                with concurrent.futures.ThreadPoolExecutor(2) as ex:
                    futs = {k: ex.submit(post, port, body_for(k), 1800) for k in (1, 2)}
                    for k in (1, 2):
                        replies[k] = futs[k].result()
                time.sleep(2)
                for k in (1, 2):
                    record(k, replies[k])
                first = 3
            for k in range(first, a.sessions + 1):
                need = int(target * bpp * 1.3) + (512 << 20)
                avail, grown = available(), swap_used() - swap0
                if avail < need or grown > (256 << 20):
                    emit({**cell, "event": "stopped", "session": k, "available": avail, "need": need,
                          "swap_grown": grown, "elapsed_s": round(time.time() - t_start)}, records)
                    break
                replies[k] = post(port, body_for(k), timeout=1800)
                time.sleep(2)
                record(k, replies[k])
                if a.settle:
                    settle(proc.pid, a.settle, cell, k, records, t_start)
            if a.pair_start and 1 in replies:
                b = body_for(1)
                b["messages"] += [{"role": "assistant", "content": replies[1]["choices"][0]["message"]["content"]},
                                  {"role": "user", "content": "And in two words?"}]
                u = post(port, b, timeout=1800).get("usage", {})
                emit({**cell, "event": "probe", "session": 1, "prompt_tokens": u.get("prompt_tokens"),
                      "reused": u.get("prefill_reused_tokens"), "footprint": footprint(proc.pid),
                      "elapsed_s": round(time.time() - t_start)}, records)
        finally:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
            log.close()
    with open(f"{a.out}/kv9-mem{a.tag}.json", "w") as f:
        json.dump(records, f, indent=1)


if __name__ == "__main__":
    sys.exit(main())
