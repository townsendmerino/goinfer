#!/usr/bin/env python3
"""KV4's G3c (kv4-cuda-2026-10-21.md §2): one request of about the per-conversation cap the banner names (Qwen2.5-7B-Instruct, serve
defaults) completes on the GPU path: a prompt of four depth-8,000 bench prompts with a unique prefix, 64 generated tokens, served by
the binary given. Prints the prompt tokens served, the tokens generated, the finish reason and the server's decode path line.

  g3c_long_request.py <serve binary> <log>
"""
import json, os, signal, subprocess, sys, time, urllib.request
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../scripts"))
import bench_peer_prefill as bp

binary, log = sys.argv[1], sys.argv[2]
MODEL = os.path.expanduser("~/models/qwen2.5-7b-instruct-q4_k_m.gguf")
p = subprocess.Popen([binary, "-model", f"bench={MODEL}", "-backend", "cuda", "-addr", f"127.0.0.1:{bp.GPORT}", "-quant", "int4", "-require-backend"],
                     stdout=open(log, "w"), stderr=subprocess.STDOUT, start_new_session=True)
try:
    assert bp.wait_port(bp.GPORT, 300), "serve did not come up"
    text = bp.uniq(1) + " ".join([bp.PROMPTS["7B:8000"]["text"]] * 4)
    body = json.dumps({"model": "bench", "max_tokens": 64, "temperature": 0, "messages": [{"role": "user", "content": text}]}).encode()
    t0 = time.time()
    req = urllib.request.Request(f"http://127.0.0.1:{bp.GPORT}/v1/chat/completions", body, {"Content-Type": "application/json"})
    try:
        r = json.load(urllib.request.urlopen(req, timeout=1800))
        u = r.get("usage", {})
        print(f"HTTP 200 in {time.time() - t0:.1f}s: prompt {u.get('prompt_tokens')} tokens, generated {u.get('completion_tokens')}, "
              f"finish {r['choices'][0].get('finish_reason')}, total {u.get('prompt_tokens', 0) + u.get('completion_tokens', 0)}")
    except urllib.error.HTTPError as e:
        print(f"HTTP {e.code}: {e.read()[:400]!r}")
finally:
    os.killpg(p.pid, signal.SIGTERM)
    time.sleep(5)
