#!/usr/bin/env bash
# Cold first-token top-3 for each prompt length on a FRESH server per (backend, model, length): no prefix reuse anywhere.
W=$HOME/goinfer-bench/peer-reanchor-2026-11-19; O=$HOME/goinfer-work/repro8000
for m in 0.5B 1.5B; do
  case $m in 0.5B) M=$HOME/models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf;; 1.5B) M=$HOME/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf;; esac
  for k in 256 512 1024 2048 3900; do
    for be in cuda cpu; do
      [ $m = 1.5B ] && [ $be = cpu ] && [ $k -gt 2048 ] && continue
      $W/bin/serve-cuda-e29da447 -model bench=$M -addr 127.0.0.1:18199 -backend $be -ctx 8192 -quant int4 > $O/len.log 2>&1 & pid=$!
      for _ in $(seq 1 240); do grep -q 'goinfer serving on' $O/len.log && break; sleep 0.25; done
      KEY=$m:$k BE=$be python3 - <<'PY'
import json, os, urllib.request
p = json.load(open("scripts/prompts.json"))[os.environ["KEY"]]; text = p["texts"][0] if "texts" in p else p["text"]
body = {"model": "bench", "max_tokens": 1, "temperature": 0, "seed": 1, "logprobs": True, "top_logprobs": 3, "messages": [{"role": "user", "content": text}]}
r = json.load(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:18199/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=600))
lp = r["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
print(f"{os.environ['KEY']:10s} {os.environ['BE']:4s} ptok={r['usage']['prompt_tokens']:5d}", [(t["token"], round(t["logprob"], 2)) for t in lp])
PY
      kill $pid; wait $pid 2>/dev/null
    done
  done
done
