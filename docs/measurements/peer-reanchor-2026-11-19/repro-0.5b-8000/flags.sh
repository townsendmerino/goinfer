#!/usr/bin/env bash
O=$HOME/goinfer-work/repro8000; P=$HOME/goinfer-bench/peer-reanchor-2026-11-19/wt/scripts/prompts.json; b=$HOME/goinfer-bench/peer-reanchor-2026-11-19/bin/serve-cuda-e29da447
for extra in "-embed-int4=false" "-embed-int4=false -kv f32" "-max-concurrent 1"; do
  $b -model bench=$HOME/models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf -addr 127.0.0.1:18199 -backend cuda -ctx 8192 -quant int4 $extra > $O/flags.log 2>&1 & pid=$!
  for _ in $(seq 1 240); do grep -q 'serving on' $O/flags.log && break; kill -0 $pid 2>/dev/null || break; sleep 0.25; done
  X="$extra" P=$P python3 - <<'PY'
import json, os, urllib.request
p = json.load(open(os.environ["P"]))["0.5B:8000"]; text = p["texts"][0] if "texts" in p else p["text"]
try:
    r = json.load(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:18199/v1/chat/completions", json.dumps({"model": "bench", "max_tokens": 64, "temperature": 0, "seed": 1, "messages": [{"role": "user", "content": text}]}).encode(), {"Content-Type": "application/json"}), timeout=600))
    print(f"{os.environ['X']:28s} n={r['usage']['completion_tokens']:2d} {r['choices'][0]['message']['content'][:60]!r}")
except Exception as e: print(os.environ["X"], "ERROR", e)
PY
  kill $pid; wait $pid 2>/dev/null
done
