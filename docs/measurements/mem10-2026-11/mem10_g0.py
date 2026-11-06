#!/usr/bin/env python3
"""MEM10-G0 (README.md here): the host footprint of the Gemma 4 26B-A4B on CUDA as `serve` loads it today. One server, started at serve's defaults except --backend cuda, --ctx 4096 and --moe-cache-experts;
/proc/<pid>/status is read every 250 ms until the server answers /v1/models, again 5 s after, and after one 16-token greedy completion; the process's device memory from nvidia-smi.

  python3 mem10_g0.py --serve <bin> --model <gguf> --out <dir>

Reads, grades nothing. Writes <out>/reading.json and prints the numbers.
"""
import argparse, json, os, signal, subprocess, threading, time, urllib.request

PORT = 18223
BASE = f"http://127.0.0.1:{PORT}"
MiB = 1 << 20


def status(pid):
    d = {}
    try:
        for line in open(f"/proc/{pid}/status"):
            k, _, v = line.partition(":")
            if k in ("RssAnon", "RssFile", "RssShmem", "VmHWM", "VmSwap", "VmLck"):
                d[k] = int(v.split()[0]) * 1024
    except FileNotFoundError:
        pass
    return d


def vram(pid):
    out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
    for ln in out.splitlines():
        p, _, m = ln.partition(",")
        if p.strip() == str(pid):
            return int(m.strip())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--serve", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    load0 = float(open("/proc/loadavg").read().split()[0])
    avail0 = [int(l.split()[1]) for l in open("/proc/meminfo") if l.startswith("MemAvailable")][0] // 1024
    log = open(os.path.join(a.out, "serve.log"), "w")
    p = subprocess.Popen([a.serve, "--model", a.model, "--backend", "cuda", "--ctx", "4096", "--moe-cache-experts", "--addr", f"127.0.0.1:{PORT}"], stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
    peak = {}
    stop = threading.Event()

    def sample():
        while not stop.is_set():
            for k, v in status(p.pid).items():
                peak[k] = max(peak.get(k, 0), v)
            time.sleep(0.25)

    threading.Thread(target=sample, daemon=True).start()
    t0 = time.time()
    rec = {"load_avg": load0, "mem_available_mb": avail0}
    try:
        while True:
            if p.poll() is not None:
                raise RuntimeError(f"the server exited {p.returncode} (serve.log)")
            if time.time() - t0 > 900:
                raise RuntimeError("the server did not listen in 900 s")
            try:
                urllib.request.urlopen(BASE + "/v1/models", timeout=1).read()
                break
            except Exception:
                time.sleep(0.5)
        rec["load_s"] = round(time.time() - t0, 1)
        rec["decode_path"] = [l.strip() for l in open(os.path.join(a.out, "serve.log")) if "decode path:" in l]
        rec["at_listen"] = status(p.pid)
        time.sleep(5)
        rec["after_5s"] = status(p.pid)
        rec["vram_mib"] = vram(p.pid)
        mid = json.load(urllib.request.urlopen(BASE + "/v1/models", timeout=5))["data"][0]["id"]
        body = {"model": mid, "prompt": "The capital of France is", "max_tokens": 16, "temperature": 0}
        r = json.load(urllib.request.urlopen(urllib.request.Request(BASE + "/v1/completions", json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=300))
        rec["completion"] = r["choices"][0]["text"]
        rec["after_request"] = status(p.pid)
    finally:
        stop.set()
        rec["peak_during_all"] = dict(peak)
        os.killpg(p.pid, signal.SIGTERM)
        try:
            p.wait(timeout=60)
        except Exception:
            os.killpg(p.pid, signal.SIGKILL)
    json.dump(rec, open(os.path.join(a.out, "reading.json"), "w"), indent=1)
    for k in ("at_listen", "after_5s", "after_request", "peak_during_all"):
        print(k, {x: round(v / MiB) for x, v in rec[k].items()}, "MiB")
    print("decode path:", rec.get("decode_path"))
    print("load", rec["load_s"], "s; device", rec["vram_mib"], "MiB; load avg", rec["load_avg"], "; available", rec["mem_available_mb"], "MB; completion", repr(rec["completion"]))


if __name__ == "__main__":
    main()
