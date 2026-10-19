#!/usr/bin/env python3
"""probe.py without the chat template: /v1/completions with a plain-text prompt, then the same prompt plus the exact text the model generated, then once more.
If the append-only request reuses the held sequence here and not through /v1/chat/completions, the engine's strict-extension rule works and the chat rendering
(the generation prompt, the assistant header) is what breaks it.

  python3 probe_raw.py --serve <binary> --model <gguf> --out <dir> [--size 4096] [--ctx 16384]
"""
import argparse, json, os, signal, subprocess, sys, time, urllib.request
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from probe import BASE, PORT, filler, post  # noqa: E402


def complete(prompt, n=24):
    body = {"model": "probe", "prompt": prompt, "stream": True, "max_tokens": n, "temperature": 0, "stream_options": {"include_usage": True}}
    req = urllib.request.Request(BASE + "/v1/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0, first, text, ptoks = time.time(), None, "", None
    with urllib.request.urlopen(req, timeout=900) as r:
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
                text += c.get("text") or ""
    return first, text, ptoks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--serve", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", type=int, default=4096)
    ap.add_argument("--ctx", default="16384")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    log = open(os.path.join(a.out, "serve-raw.log"), "w")
    proc = subprocess.Popen([a.serve, "-model", f"probe={a.model}", "-backend", "cuda", "-addr", f"127.0.0.1:{PORT}", "-ctx", a.ctx],
                            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    rows = []
    try:
        t0 = time.time()
        while time.time() - t0 < 600:
            try:
                urllib.request.urlopen(BASE + "/v1/models", timeout=1).read()
                break
            except Exception:
                if proc.poll() is not None:
                    sys.exit("server exited during load")
                time.sleep(0.5)
        probe_text = filler(4000)
        n = post("/v1/messages/count_tokens", {"model": "probe", "messages": [{"role": "user", "content": probe_text}]}).get("input_tokens")
        cpt = len(probe_text) / n
        complete("warm up", 4)
        p = "Reference notes:\n" + filler(int(a.size * cpt)) + "\n\nSummary:"
        for turn in (1, 2, 3):
            first, text, ptoks = complete(p)
            rows.append({"turn": turn, "ttft_s": round(first, 3), "prompt_tokens": ptoks, "generated": text})
            print(f"  raw turn {turn}: ttft {first:.2f}s, {ptoks} prompt tokens, generated {text!r}", flush=True)
            p = p + text + " Next:"
        first, _, ptoks = complete(f"run {time.time_ns()}\n" + p)
        rows.append({"turn": "cold control", "ttft_s": round(first, 3), "prompt_tokens": ptoks})
        print(f"  raw cold control: ttft {first:.2f}s, {ptoks} prompt tokens", flush=True)
    finally:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
            proc.wait(timeout=30)
        except Exception:
            os.killpg(proc.pid, signal.SIGKILL)
        log.close()
    json.dump({"serve": a.serve, "model": a.model, "rows": rows}, open(os.path.join(a.out, "probe-raw.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
