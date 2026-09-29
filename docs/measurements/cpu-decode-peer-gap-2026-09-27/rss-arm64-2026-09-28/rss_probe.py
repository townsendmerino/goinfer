#!/usr/bin/env python3
"""Peak process-group RSS of goinfer-serve on the 1.5B, CPU int4, arm64: the pre-L1 build against HEAD, each on its
OWN sidecar (task-cpu-decode-peer-gap-2026-09.md, L1 "RSS, observed, not explained", item 4). Not a timed measurement.

Per load: start serve, poll the group's RSS every 50 ms (bench_peer.py's method: `ps -eo pgid,rss`, summed), wait for
it to listen, send one greedy 64-token completion, stop, kill. Loads alternate old/new. A load that transcodes a
sidecar is reported and not counted.
"""
import json, os, signal, subprocess, sys, threading, time, urllib.request

D = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.expanduser("~/models")
GGUF = "qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"
OLD = (os.path.expanduser("~/goinfer-bench/l1-mac/serve-cpu-3cd62e6d"), os.path.join(D, "old", GGUF), [])
NEW = (os.path.join(D, sys.argv[1]), os.path.join(MODELS, GGUF), ["-embed-int4=false"])
PORT = 18931


def group_rss_kb(pgid):
    out = subprocess.run(["ps", "-eo", "pgid,rss"], capture_output=True, text=True).stdout
    tot, seen = 0, False
    for ln in out.splitlines()[1:]:
        p = ln.split()
        if len(p) == 2 and p[0].isdigit() and int(p[0]) == pgid:
            tot += int(p[1]); seen = True
    return tot if seen else None


def load(label, spec, ctx):
    binary, path, extra = spec
    args = [binary, "-model", f"bench={path}", "-backend", "cpu", "-quant", "int4", "-addr", f"127.0.0.1:{PORT}"] + extra
    if ctx:
        args += ["-ctx", str(ctx)]
    errp = os.path.join(D, f"stderr-{label}.log")
    proc = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=open(errp, "w"), preexec_fn=os.setsid)
    peak, stop = [0], threading.Event()

    def poll():
        while not stop.is_set():
            r = group_rss_kb(proc.pid)
            if r:
                peak[0] = max(peak[0], r)
            time.sleep(0.05)
    th = threading.Thread(target=poll, daemon=True); th.start()
    t0 = time.time()
    while time.time() - t0 < 600:
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/v1/models", timeout=2).read(); break
        except Exception:
            if proc.poll() is not None:
                break
            time.sleep(0.2)
    ok = proc.poll() is None
    if ok:
        body = json.dumps({"model": "bench", "messages": [{"role": "user", "content": "Write a haiku about memory."}],
                           "max_tokens": 64, "temperature": 0}).encode()
        req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", body, {"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=600).read()
    stop.set(); th.join()
    os.killpg(proc.pid, signal.SIGTERM); proc.wait()
    err = open(errp).read()
    transcoded = "transcod" in err.lower()
    ctxline = next((l for l in err.splitlines() if "ctx" in l.lower() or "context" in l.lower()), "")
    return peak[0], ok, transcoded, ctxline[:160]


if __name__ == "__main__":
    for ctx in (4096, None):
        print(f"== ctx {ctx or 'auto (as the L1 cells ran)'}  (free memory: "
              f"{subprocess.run(['sh', '-c', 'vm_stat | grep -E \"Pages free|Pages inactive\"'], capture_output=True, text=True).stdout.split()})")
        for i in range(4):
            for label, spec in (("old-3cd62e6d", OLD), ("new-" + sys.argv[1].split("-")[-1], NEW)):
                peak, ok, tr, ctxline = load(label, spec, ctx)
                tag = " TRANSCODED (not counted)" if tr else ""
                print(f"  {label:22s} peak {peak / 1024:8.1f} MB  {'ok' if ok else 'FAILED'}{tag}   {ctxline}", flush=True)
