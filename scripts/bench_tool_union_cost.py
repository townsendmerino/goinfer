#!/usr/bin/env python3
"""bench_tool_union_cost.py — gate B of docs/measurements/tool-union-2026-09-24.md: what does arming the lazy
multi-tool union cost on an `auto` turn whose answer is PROSE? Any LogitProcessor turns off the decoder's on-device
greedy/sampling fast paths for the whole turn, so this measures decode tok/s, default vs GOINFER_TOOL_UNION=0, on a
12-tool request that answers in prose. Fresh server per arm, ABBA x2 per cell, 3 decodes of 256 tokens per server;
decode tok/s = 256 / (t(257 tokens) - t(1 token)).

  GOINFER_SERVE_CUDA=... python3 scripts/bench_tool_union_cost.py out.json
"""
import json, os, signal, socket, subprocess, sys, time, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
SERVE = os.environ["GOINFER_SERVE_CUDA"]
PORT = 8094
M = os.path.expanduser("~/models/")
MODELS = {"1.5B": M + "qwen2.5-coder-1.5b-instruct-q4_k_m.gguf", "7B": M + "qwen2.5-7b-instruct-q4_k_m.gguf"}
TOOLS = json.load(open(os.path.join(HERE, "w4_transcript_base.json")))["tools"]
PROMPT = "Explain how a token bucket rate limiter works, in prose. Do not call any tools."


def up():
    t = time.time()
    while time.time() - t < 900:
        try:
            with socket.create_connection(("127.0.0.1", PORT), 1):
                return
        except OSError:
            time.sleep(0.2)
    raise RuntimeError("server did not come up")


def post(p):
    r = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", data=json.dumps(p).encode(),
                               headers={"Content-Type": "application/json"})
    t = time.perf_counter()
    b = json.loads(urllib.request.urlopen(r, timeout=900).read())
    return time.perf_counter() - t, b


def arm(model, union_on, temp):
    env = dict(os.environ)
    env["GOINFER_TOOL_UNION"] = "1" if union_on else "0"
    p = subprocess.Popen([SERVE, "-model", "m=" + MODELS[model], "-backend", "cuda", "-addr", f"127.0.0.1:{PORT}", "-quant", "int4"],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, preexec_fn=os.setsid, env=env)
    try:
        up()
        base = {"model": "m", "messages": [{"role": "user", "content": PROMPT}], "tools": TOOLS, "tool_choice": "auto", "temperature": temp}
        post({**base, "max_tokens": 8, "seed": 1})
        rows = []
        for i in range(3):
            t1, _ = post({**base, "max_tokens": 1, "seed": 100 + i})
            t2, b = post({**base, "max_tokens": 257, "seed": 100 + i})
            m = b["choices"][0]["message"]
            n = b["usage"]["completion_tokens"]
            rows.append({"tok_s": (n - 1) / (t2 - t1), "n": n, "called": bool(m.get("tool_calls")),
                         "has_opener": "<tool_call>" in (m.get("content") or "")})
        return rows
    finally:
        os.killpg(os.getpgid(p.pid), signal.SIGTERM)
        p.wait(60)
        time.sleep(4)


out = {"serve": SERVE, "cells": {}}
for model in ("1.5B", "7B"):
    for temp in (0.0, 0.7):
        key = f"{model}@T{temp}"
        blocks = []
        for order in ((True, False), (False, True), (True, False), (False, True)):
            blk = {}
            for on in order:
                blk["on" if on else "off"] = arm(model, on, temp)
            blocks.append(blk)
            med = lambda rs: sorted(r["tok_s"] for r in rs)[1]
            print(f"{key} block {len(blocks)}: on {med(blk['on']):.2f} off {med(blk['off']):.2f} ratio {med(blk['on'])/med(blk['off']):.4f}", flush=True)
        out["cells"][key] = blocks
json.dump(out, open(sys.argv[1], "w"), indent=1)
print("DONE", flush=True)
