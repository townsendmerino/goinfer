#!/usr/bin/env python3
"""Cold time-to-first-token of Gemma 4 E4B at several prompt sizes, two binaries interleaved cell by cell, with load and free memory recorded per cell
(README.md in this directory).

  python3 ttft_e4b.py --before <bin> --after <bin> --out <dir> [--sizes 12288,16384,24576] [--cap 900]
"""
import argparse, json, os, signal, subprocess, sys, time, urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "qwen35-reuse-probe-2026-10-18"))
from probe import BASE, PORT, filler, post  # noqa: E402

MODEL = os.path.expanduser("~/models/gemma-4-e4b-gguf/gemma-4-E4B_q4_0-it.gguf")


def loadavg():
    return float(open("/proc/loadavg").read().split()[0])


def memavail_gb():
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable:"):
            return round(int(line.split()[1]) / 1048576, 1)


def gpu_free_mib():
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout.strip()
    return int(out.split()[0]) if out else None


def cold(chars, cap):
    prompt = f"run {time.time_ns()}\n" + "Reference notes:\n" + filler(chars) + "\n\nSummary:"
    body = {"model": "probe", "prompt": prompt, "stream": True, "max_tokens": 1, "temperature": 0, "stream_options": {"include_usage": True}}
    req = urllib.request.Request(BASE + "/v1/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0, first, ptoks = time.time(), None, None
    try:
        with urllib.request.urlopen(req, timeout=cap) as r:
            for raw in r:
                line = raw.decode().strip()
                if not line.startswith("data:") or line == "data: [DONE]":
                    continue
                d = json.loads(line[5:])
                if d.get("usage"):
                    ptoks = d["usage"].get("prompt_tokens")
                for c in d.get("choices", []):
                    if c.get("text") and first is None:
                        first = time.time() - t0
    except Exception as e:  # the cap: a recorded stop, not a retry
        return None, ptoks, f"stopped at {cap}s ({type(e).__name__})"
    return first, ptoks, None


def cell(arm, binary, size, out, cap):
    rec = {"arm": arm, "target_tokens": size, "load_before": loadavg(), "memavail_before_gb": memavail_gb(), "gpu_free_before_mib": gpu_free_mib()}
    rec["load_flagged"] = rec["load_before"] > 2.0
    log = open(os.path.join(out, f"serve-{arm}-{size}.log"), "w")
    t0 = time.time()
    proc = subprocess.Popen([binary, "-model", f"probe={MODEL}", "-backend", "cuda", "-addr", f"127.0.0.1:{PORT}", "-ctx", "32768"], stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    try:
        while time.time() - t0 < 900:
            try:
                urllib.request.urlopen(BASE + "/v1/models", timeout=1).read()
                break
            except Exception:
                if proc.poll() is not None:
                    rec["error"] = "server exited during load"
                    return rec
                time.sleep(0.5)
        rec["ready_s"] = round(time.time() - t0, 1)
        probe_text = filler(4000)
        n = post("/v1/messages/count_tokens", {"model": "probe", "messages": [{"role": "user", "content": probe_text}]}).get("input_tokens")
        cpt = len(probe_text) / n
        cold(400, 120)  # warm-up
        t1 = time.time()
        first, ptoks, err = cold(int(size * cpt), cap)
        rec.update(prompt_tokens=ptoks, ttft_s=None if first is None else round(first, 2), tok_per_s=None if not first or not ptoks else round(ptoks / first, 1), wall_s=round(time.time() - t1, 1))
        if err:
            rec["stopped"] = err
        rec.update(load_after=loadavg(), memavail_after_gb=memavail_gb(), gpu_free_at_first_token_mib=gpu_free_mib())
    finally:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
            proc.wait(timeout=60)
        except Exception:
            os.killpg(proc.pid, signal.SIGKILL)
        log.close()
        base = rec.get("gpu_free_before_mib")
        for _ in range(60):  # the next cell starts from the same free VRAM
            f = gpu_free_mib()
            if base is None or (f is not None and f >= base - 64):
                break
            time.sleep(1)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--before", required=True)
    ap.add_argument("--after", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sizes", default="12288,16384,24576")
    ap.add_argument("--cap", type=int, default=900)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    bins = {"before": a.before, "after": a.after}
    rows = []
    for i, size in enumerate(int(x) for x in a.sizes.split(",")):
        order = ("before", "after") if i % 2 == 0 else ("after", "before")
        for arm in order:
            r = cell(arm, bins[arm], size, a.out, a.cap)
            rows.append(r)
            print(json.dumps(r), flush=True)
            json.dump({"before": a.before, "after": a.after, "rows": rows}, open(os.path.join(a.out, "ttft-e4b.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
