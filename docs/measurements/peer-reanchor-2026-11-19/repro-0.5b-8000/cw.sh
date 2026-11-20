#!/usr/bin/env bash
W=$HOME/goinfer-bench/peer-reanchor-2026-11-19; O=$HOME/goinfer-work/repro8000
for extra in "" "-exact-prefill"; do
  $W/bin/serve-cuda-e29da447 -model bench=$HOME/models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf -addr 127.0.0.1:18199 -backend cuda -ctx 8192 -quant int4 $extra > $O/cw.log 2>&1 & pid=$!
  for _ in $(seq 1 240); do grep -q 'goinfer serving on' $O/cw.log && break; sleep 0.5; done
  EXTRA="${extra:-default}" python3 - <<'PY'
import json, os, urllib.request
P = json.load(open("scripts/prompts.json"))
def ask(key, tag):
    p = P[key]; text = p["texts"][0] if "texts" in p else p["text"]
    body = {"model": "bench", "max_tokens": 1, "temperature": 0, "seed": 1, "logprobs": True, "top_logprobs": 5, "messages": [{"role": "user", "content": text}]}
    r = json.load(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:18199/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=600))
    lp = r["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
    print(f"{os.environ['EXTRA']:14s} {tag:22s} reused={r['usage'].get('prefill_reused_tokens')}", [(t["token"], round(t["logprob"], 2)) for t in lp])
ask("0.5B:8000", "8000 cold")
ask("0.5B:8000", "8000 again")
ask("0.5B:3900", "3900 after 8000")
PY
  kill $pid; wait $pid 2>/dev/null
done
