#!/usr/bin/env bash
W=$HOME/goinfer-bench/peer-reanchor-2026-11-19; O=$HOME/goinfer-work/repro8000
for be in cuda cpu; do
  for key in 0.5B:8000 0.5B:3900; do
  $W/bin/serve-cuda-e29da447 -model bench=$HOME/models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf -addr 127.0.0.1:18199 -backend $be -ctx 8192 -quant int4 > $O/lp-$be.log 2>&1 & pid=$!
  for _ in $(seq 1 240); do grep -q 'goinfer serving on' $O/lp-$be.log && break; sleep 0.5; done
  BE=$be KEY=$key python3 - <<'PY'
import json, os, urllib.request
p = json.load(open("scripts/prompts.json"))[os.environ["KEY"]]; text = p["texts"][0] if "texts" in p else p["text"]
body = {"model": "bench", "max_tokens": 1, "temperature": 0, "seed": 1, "logprobs": True, "top_logprobs": 5, "messages": [{"role": "user", "content": text}]}
try:
    r = json.load(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:18199/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=600))
    lp = (r["choices"][0].get("logprobs") or {}).get("content") or []
    tops = [(t["token"], round(t["logprob"], 3)) for t in lp[0]["top_logprobs"]] if lp else r["choices"][0]
    print(os.environ["BE"], os.environ["KEY"], tops)
except Exception as e: print(os.environ["BE"], os.environ["KEY"], "ERROR", e)
PY
  kill $pid; wait $pid 2>/dev/null
  done
done
