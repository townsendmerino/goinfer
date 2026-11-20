#!/usr/bin/env bash
O=$HOME/goinfer-work/repro8000; P=$HOME/goinfer-bench/peer-reanchor-2026-11-19/wt/scripts/prompts.json; b=$HOME/goinfer-bench/peer-reanchor-2026-11-19/bin/serve-cuda-e29da447
for extra in "" "-kv f32" "-embed-int4=false" "-embed-int4=false -kv f32"; do
  for key in 0.5B:1024 0.5B:8000; do
  $b -model bench=$HOME/models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf -addr 127.0.0.1:18199 -backend cuda -ctx 8192 -quant int4 $extra > $O/prec.log 2>&1 & pid=$!
  for _ in $(seq 1 240); do grep -q 'serving on' $O/prec.log && break; kill -0 $pid 2>/dev/null || break; sleep 0.25; done
  X="${extra:-default}" KEY=$key P=$P python3 - <<'PY'
import json, os, urllib.request
p = json.load(open(os.environ["P"]))[os.environ["KEY"]]; text = p["texts"][0] if "texts" in p else p["text"]
body = {"model": "bench", "max_tokens": 1, "temperature": 0, "seed": 1, "logprobs": True, "top_logprobs": 3, "messages": [{"role": "user", "content": text}]}
r = json.load(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:18199/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=600))
print(f"{os.environ['X']:26s} {os.environ['KEY']:10s}", [(t["token"], round(t["logprob"], 2)) for t in r["choices"][0]["logprobs"]["content"][0]["top_logprobs"]])
PY
  kill $pid; wait $pid 2>/dev/null
  done
done
