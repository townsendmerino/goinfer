#!/usr/bin/env python3
"""Cold time-to-first-token of one model at several prompt sizes, each with a fresh nonce at the start so nothing is reused.

The 9B's long-prompt prefill fell to the sequential path past 12,160 attended keys (the 48 KiB shared-memory bound of attn_batched, whose head dim 256 the fused
attention does not serve); this reads the same cells after the row-batched exact attention is on main. /v1/completions, plain text, one new token: no chat template,
no reuse, nothing but the prefill.

  python3 ttft_cold.py --serve <binary> --model <gguf> --out <dir> [--sizes 4096,8192,16384,31744] [--ctx 32768]
"""
import argparse, json, os, signal, subprocess, sys, time, urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "qwen35-reuse-probe-2026-10-18"))
from probe import BASE, PORT, filler, post  # noqa: E402


def cold(size_chars):
    prompt = f"run {time.time_ns()}\n" + "Reference notes:\n" + filler(size_chars) + "\n\nSummary:"
    body = {"model": "probe", "prompt": prompt, "stream": True, "max_tokens": 1, "temperature": 0, "stream_options": {"include_usage": True}}
    req = urllib.request.Request(BASE + "/v1/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0, first, ptoks = time.time(), None, None
    with urllib.request.urlopen(req, timeout=3000) as r:
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
    return first, ptoks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--serve", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sizes", default="4096,8192,16384,31744")
    ap.add_argument("--ctx", default="32768")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    log = open(os.path.join(a.out, "serve-ttft.log"), "w")
    proc = subprocess.Popen([a.serve, "-model", f"probe={a.model}", "-backend", "cuda", "-addr", f"127.0.0.1:{PORT}", "-ctx", a.ctx],
                            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    rows = []
    try:
        t0 = time.time()
        while time.time() - t0 < 900:
            try:
                urllib.request.urlopen(BASE + "/v1/models", timeout=1).read()
                break
            except Exception:
                if proc.poll() is not None:
                    sys.exit("server exited during load; see serve-ttft.log")
                time.sleep(0.5)
        print(f"server up in {time.time() - t0:.1f}s", flush=True)
        probe_text = filler(4000)
        n = post("/v1/messages/count_tokens", {"model": "probe", "messages": [{"role": "user", "content": probe_text}]}).get("input_tokens")
        cpt = len(probe_text) / n
        cold(400)  # warm up
        for size in [int(x) for x in a.sizes.split(",")]:
            first, ptoks = cold(int(size * cpt))
            rows.append({"target": size, "prompt_tokens": ptoks, "ttft_s": round(first, 2), "tok_per_s": round(ptoks / first, 1)})
            print(f"  {size}: {ptoks} prompt tokens, first token {first:.2f}s ({ptoks / first:.0f} tok/s)", flush=True)
    finally:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
            proc.wait(timeout=30)
        except Exception:
            os.killpg(proc.pid, signal.SIGKILL)
        log.close()
    json.dump({"serve": a.serve, "model": a.model, "rows": rows}, open(os.path.join(a.out, "ttft-cold.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
