#!/usr/bin/env bash
W=$HOME/goinfer-bench/peer-reanchor-2026-11-19; O=$HOME/goinfer-work/repro8000; port=18199
one() { # one <label> <serve args...>
  local label=$1; shift
  $W/bin/serve-cuda-e29da447 -model bench=$HOME/models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf -addr 127.0.0.1:$port "$@" > $O/serve-$label.log 2>&1 & local pid=$!
  for _ in $(seq 1 240); do grep -q 'goinfer serving on' $O/serve-$label.log && break; kill -0 $pid 2>/dev/null || break; sleep 0.5; done
  LABEL=$label PORT=$port python3 - <<'PY'
import json, os, urllib.request
p = json.load(open("scripts/prompts.json"))["0.5B:8000"]; text = p["texts"][0] if "texts" in p else p["text"]
body = {"model": "bench", "max_tokens": 64, "temperature": 0, "seed": 1, "messages": [{"role": "user", "content": text}]}
try:
    r = json.load(urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{os.environ['PORT']}/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=600))
    ch = r["choices"][0]; print(f"{os.environ['LABEL']:14s} n={r['usage']['completion_tokens']:2d} finish={ch.get('finish_reason')} {repr(ch['message'].get('content'))[:110]}")
except Exception as e: print(os.environ['LABEL'], "ERROR", e)
PY
  echo "               decode path: $(grep -m1 'decode path:' $O/serve-$label.log | sed 's/^ *decode path: //') | $(grep -m1 'prefill path:' $O/serve-$label.log | sed 's/^ *prefill path: //' | cut -c1-80)"
  kill $pid; wait $pid 2>/dev/null
}
one cuda-f16kv -backend cuda -ctx 8192 -quant int4
one cuda-f32kv -backend cuda -ctx 8192 -quant int4 -kv f32
one cuda-exactpf -backend cuda -ctx 8192 -quant int4 -exact-prefill
one cuda-ctx16k -backend cuda -ctx 16384 -quant int4
one cpu -backend cpu -ctx 8192 -quant int4
