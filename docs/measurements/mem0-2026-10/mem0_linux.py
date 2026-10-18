#!/usr/bin/env python3
"""MEM0, nobara half (docs/tasks/task-memory-campaign-2026-10.md §3): the server's memory through one long cold prompt, Linux.

A Linux port of docs/measurements/g40-2026-10/g40_mem.py's cell: one fresh `serve` per cell, at serve's defaults except -backend and -ctx 32768 (the backend default is 16384 on CUDA, under the prompt), one request
(a cold prompt of about 20,193 tokens: KV9's preamble, this repository's text at e42db592, cut to a character budget; max_tokens 8,
temperature 0). The server process is read every 250 ms from /proc/<pid>/status: VmRSS, RssAnon (heap, anonymous buffers),
RssFile (mapped weights), VmSwap, VmHWM; on CUDA the process's device memory from nvidia-smi every second. Readings per cell:
after load, the peak during the request, at the end of the request, and 10 s after. Swap-used is read before and after (context only).
Nothing is graded: this is the census that sizes MEM1-MEM13.

  python3 mem0_linux.py --serve <serve binary> --out <dir> --repo <checkout with e42db592> --cell cuda:1.5b [--cell ...]
cells: <backend>:<model> with backend cpu|cuda and model 1.5b|7b|e2b
"""
import argparse, json, os, re, signal, subprocess, threading, time, urllib.request

PIN = "e42db592"
SOURCES = ["CLAUDE.md", "docs/tasks/task-kv-architecture-2026-10.md", "docs/tasks/task-forward-architecture-2026-10.md", "RELEASING.md", "docs/server.md"]
CHARS = int(20000 * 3.6)
HOME = os.path.expanduser("~")
MODELS = {"1.5b": f"{HOME}/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf", "7b": f"{HOME}/models/qwen2.5-coder-7b-instruct-q4_k_m.gguf",
          "e2b": f"{HOME}/models/gemma-4-e2b-gguf/gemma-4-E2B_q4_0-it.gguf"}
MiB = 1 << 20
PORT = 18197


def preamble(repo):
    text = ""
    for p in SOURCES:
        text += subprocess.run(["git", "-C", repo, "show", f"{PIN}:{p}"], capture_output=True, text=True, check=True).stdout
        if len(text) >= CHARS:
            break
    return text[:CHARS]


def status(pid):
    d = {}
    try:
        for line in open(f"/proc/{pid}/status"):
            k, _, v = line.partition(":")
            if k in ("VmRSS", "RssAnon", "RssFile", "RssShmem", "VmHWM", "VmSwap"):
                d[k] = int(v.split()[0]) * 1024
    except OSError:
        pass
    return d


def meminfo():
    m = {}
    for line in open("/proc/meminfo"):
        k, _, v = line.partition(":")
        m[k] = int(v.split()[0]) * 1024
    return m


def vram(pid):
    out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
    for line in out.splitlines():
        p, _, mb = line.partition(",")
        if p.strip() == str(pid):
            return int(mb) * MiB
    return None


def wait_ready(port, proc, limit=600):
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


def cell(serve, backend, model, repo, out):
    name = f"{backend}-{model}"
    mi = meminfo()
    log = open(os.path.join(out, f"serve-{name}.log"), "w")
    swap0 = mi["SwapTotal"] - mi["SwapFree"]
    t0 = time.time()
    proc = subprocess.Popen([serve, "-model", f"bench={MODELS[model]}", "-backend", backend, "-ctx", "32768", "-addr", f"127.0.0.1:{PORT}"], stdout=log, stderr=subprocess.STDOUT,
                            start_new_session=True)
    rec = {"cell": name, "model_file": MODELS[model], "swap_before": swap0, "mem_available_before": mi["MemAvailable"], "started": time.strftime("%FT%T")}
    try:
        if not wait_ready(PORT, proc):
            rec["error"] = "server did not come up"
            return rec
        rec["load_secs"] = round(time.time() - t0, 1)
        time.sleep(3)
        rec["after_load"] = status(proc.pid)
        if backend == "cuda":
            rec["vram_after_load"] = vram(proc.pid)
        peak, stop, vmax = {}, threading.Event(), [0]

        def sample():
            last = 0
            while not stop.is_set():
                s = status(proc.pid)
                for k, v in s.items():
                    peak[k] = max(peak.get(k, 0), v)
                if backend == "cuda" and time.time() - last > 1:
                    v = vram(proc.pid)
                    vmax[0] = max(vmax[0], v or 0)
                    last = time.time()
                stop.wait(0.25)

        th = threading.Thread(target=sample, daemon=True)
        th.start()
        body = json.dumps({"model": "bench", "messages": [{"role": "user", "content": preamble(repo) + "\n\nReply with one word."}], "max_tokens": 8, "temperature": 0}).encode()
        t1 = time.time()
        r = json.load(urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", body, {"Content-Type": "application/json"}), timeout=1500))
        rec["request_secs"] = round(time.time() - t1, 1)
        rec["usage"] = r.get("usage")
        rec["at_end"] = status(proc.pid)
        time.sleep(10)
        stop.set()
        th.join()
        rec["peak_during_request"] = peak
        rec["after_10s"] = status(proc.pid)
        if backend == "cuda":
            rec["vram_peak"] = vmax[0] or None
            rec["vram_after_10s"] = vram(proc.pid)
        mi = meminfo()
        rec["swap_after"] = mi["SwapTotal"] - mi["SwapFree"]
    finally:
        try:
            stop.set()
        except NameError:
            pass
        try:
            os.killpg(proc.pid, signal.SIGTERM)
            proc.wait(timeout=30)
        except Exception:
            os.killpg(proc.pid, signal.SIGKILL)
    return rec


def mib(d):
    return {k: round(v / MiB, 1) for k, v in (d or {}).items()}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--serve", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--cell", action="append", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for c in a.cell:
        backend, model = c.split(":")
        print(f"# {time.strftime('%T')} cell {c}", flush=True)
        rec = cell(a.serve, backend, model, a.repo, a.out)
        json.dump(rec, open(os.path.join(a.out, f"cell-{backend}-{model}.json"), "w"), indent=1)
        print(json.dumps({k: (mib(v) if isinstance(v, dict) and k not in ("usage",) else v) for k, v in rec.items()}), flush=True)
