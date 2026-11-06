#!/usr/bin/env python3
"""D8b-cuda's cell on the served path (README.md here): one qwen2.5-coder-0.5b q4k checkpoint on CUDA, loaded by `serve` through its sidecar .giw (the default) and
with -direct-load, one server per arm, direct first. After the load each server process's /proc/<pid>/status is read (RssAnon, RssFile, RssShmem, VmSwap), then one
greedy completion of 16 tokens is taken and the process's device memory is read from nvidia-smi.

  python3 d8b_cuda_served.py --serve <bin> --out <dir>

Prints the readings and a VERDICT line; writes <out>/readings.json. Exit 0 PASS, 1 FAIL, 2 VOID, 3 PARKED.
VOID (exit 2), enforced: an arm starts with the one-minute load average above LOAD_VOID (8.0), or its server's log does not say `decode path: cuda-resident`
(the model ran on the CPU, or declined), or a server does not start. A void arm is a void run, not a reading.
"""
import argparse, json, os, signal, subprocess, sys, time, urllib.request

PORT = 18211
BASE = f"http://127.0.0.1:{PORT}"
MODEL = os.path.expanduser("~/models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf")
PROMPT = "Write a Go function that returns the sum of a slice of ints.\n\nfunc Sum("
MiB = 1 << 20
LOAD_VOID = 8.0


class Void(Exception):
    pass


def status(pid):
    d = {}
    for line in open(f"/proc/{pid}/status"):
        k, _, v = line.partition(":")
        if k in ("RssAnon", "RssFile", "RssShmem", "VmSwap", "VmHWM"):
            d[k] = int(v.split()[0]) * 1024
    return d


def vram_mib(pid):
    try:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
        for ln in out.splitlines():
            p, _, m = ln.partition(",")
            if p.strip() == str(pid):
                return int(m.strip())
    except Exception:
        pass
    return None


def arm(a, name, extra):
    log = open(os.path.join(a.out, f"serve-{name}.log"), "w")
    load = float(open("/proc/loadavg").read().split()[0])
    if load > LOAD_VOID:
        raise Void(f"{name}: one-minute load average {load:.2f} over {LOAD_VOID} at the arm's start")
    argv = [a.serve, "--model", MODEL, "--backend", "cuda", "--quant", "q4k", "--ctx", "4096", "--addr", f"127.0.0.1:{PORT}"] + extra
    t0 = time.time()
    p = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
    try:
        while True:
            if p.poll() is not None:
                raise RuntimeError(f"{name}: the server exited {p.returncode} (see serve-{name}.log)")
            if time.time() - t0 > 300:
                raise RuntimeError(f"{name}: the server did not listen in 300 s")
            try:
                urllib.request.urlopen(BASE + "/v1/models", timeout=1).read()
                break
            except Exception:
                time.sleep(0.3)
        load_s = time.time() - t0
        time.sleep(2)
        path = [l.strip() for l in open(os.path.join(a.out, f"serve-{name}.log")) if "decode path:" in l]
        if not any("cuda-resident" in l for l in path):
            raise Void(f"{name}: the server's log does not say `decode path: cuda-resident` (it says {path or 'nothing'})")
        after_load = status(p.pid)
        body = {"model": json.load(urllib.request.urlopen(BASE + "/v1/models", timeout=5))["data"][0]["id"], "prompt": PROMPT, "max_tokens": 16, "temperature": 0}
        req = urllib.request.Request(BASE + "/v1/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
        r = json.load(urllib.request.urlopen(req, timeout=120))
        text = r["choices"][0]["text"]
        return {"arm": name, "load_s": round(load_s, 1), "load_avg": load, "after_load": after_load, "after_request": status(p.pid), "vram_mib": vram_mib(p.pid),
                "text": text, "usage": r.get("usage"), "decode_path": path}
    finally:
        os.killpg(p.pid, signal.SIGTERM)
        try:
            p.wait(timeout=30)
        except Exception:
            os.killpg(p.pid, signal.SIGKILL)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--serve", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    try:
        recs = [arm(a, "direct", ["-direct-load"]), arm(a, "sidecar", [])]
    except Void as e:
        print(f"VERDICT: VOID ({e})")
        sys.exit(2)
    json.dump(recs, open(os.path.join(a.out, "readings.json"), "w"), indent=1)
    d, s = recs
    for r in recs:
        al = r["after_load"]
        print(f"{r['arm']}: load {r['load_s']} s (load avg {r['load_avg']:.2f}); after load RssAnon {al['RssAnon']/MiB:.0f} MiB, RssFile {al['RssFile']/MiB:.0f}, RssShmem {al['RssShmem']/MiB:.0f}, VmSwap {al['VmSwap']/MiB:.0f}; device {r['vram_mib']} MiB; {r['decode_path']}; text {r['text']!r}")
    ratio = s["after_load"]["RssAnon"] / d["after_load"]["RssAnon"]
    same = d["text"] == s["text"]
    print(f"RssAnon after load, sidecar over direct: {ratio:.3f}; the 16-token completions {'equal' if same else 'DIFFER'}")
    if not same or ratio > 0.5:
        print(f"VERDICT: FAIL (ratio {ratio:.3f}, completions {'equal' if same else 'differ'})")
        sys.exit(1)
    if ratio > 0.25:
        print(f"VERDICT: PARKED (ratio {ratio:.3f})")
        sys.exit(3)
    print(f"VERDICT: PASS (ratio {ratio:.3f})")


if __name__ == "__main__":
    main()
