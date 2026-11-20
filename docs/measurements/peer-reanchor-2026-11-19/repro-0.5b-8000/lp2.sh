#!/usr/bin/env bash
W=$HOME/goinfer-bench/peer-reanchor-2026-11-19; O=$HOME/goinfer-work/repro8000
probe() { PORT=$1 TAG=$2 python3 - <<'PY'
import json, os, urllib.request
for key in ("0.5B:128", "0.5B:3900", "0.5B:8000"):
    p = json.load(open("scripts/prompts.json"))[key]; text = p["texts"][0] if "texts" in p else p["text"]
    body = {"model": "bench", "max_tokens": 1, "temperature": 0, "seed": 1, "logprobs": True, "top_logprobs": 5, "messages": [{"role": "user", "content": text}]}
    try:
        r = json.load(urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{os.environ['PORT']}/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=600))
        lp = r["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
        print(f"{os.environ['TAG']:9s} {key:10s}", [(t["token"], round(t["logprob"], 2)) for t in lp])
    except Exception as e: print(os.environ["TAG"], key, "ERROR", e)
PY
}
unset LD_LIBRARY_PATH
$HOME/mycode/peers/llama.cpp/build/bin/llama-server --model $HOME/models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf --port 18301 --host 127.0.0.1 --ctx-size 8192 -ngl 99 > $O/llama2.log 2>&1 & lp=$!
for _ in $(seq 1 120); do curl -s -m 1 http://127.0.0.1:18301/health | grep -q ok && break; sleep 0.5; done
probe 18301 llamacpp; kill $lp; wait $lp 2>/dev/null
for be in cuda cpu; do
  $W/bin/serve-cuda-e29da447 -model bench=$HOME/models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf -addr 127.0.0.1:18199 -backend $be -ctx 8192 -quant int4 > $O/lp2-$be.log 2>&1 & pid=$!
  for _ in $(seq 1 240); do grep -q 'goinfer serving on' $O/lp2-$be.log && break; sleep 0.5; done
  probe 18199 goinfer-$be; kill $pid; wait $pid 2>/dev/null
done
