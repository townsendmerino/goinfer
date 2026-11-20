#!/usr/bin/env bash
# The cold probe on several builds: a fresh server per build and prompt, greedy at the cell's flags.
O=$HOME/goinfer-work/repro8000; P=$HOME/goinfer-bench/peer-reanchor-2026-11-19/wt/scripts/prompts.json
for b in $HOME/goinfer-bench/mc3-cuda-2026-09-27/serve-cuda-7a44a58e $HOME/goinfer-bench/b35-e2e/bin/serve-cuda-base-fac209c2 \
         $HOME/goinfer-bench/s4-pinned/bin/serve-cuda-7545a43c $HOME/goinfer-bench/f16-speed/serve-cuda-f5f06f4b2 \
         $HOME/goinfer-bench/peer-reanchor-2026-11-19/bin/serve-cuda-e29da447; do
  for key in 0.5B:1024 0.5B:8000; do
    $b -model bench=$HOME/models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf -addr 127.0.0.1:18199 -backend cuda -ctx 8192 -quant int4 > $O/old.log 2>&1 & pid=$!
    for _ in $(seq 1 240); do grep -q 'serving on' $O/old.log && break; kill -0 $pid 2>/dev/null || break; sleep 0.25; done
    B=$(basename $b) KEY=$key P=$P python3 - <<'PY'
import json, os, urllib.request
p = json.load(open(os.environ["P"]))[os.environ["KEY"]]; text = p["texts"][0] if "texts" in p else p["text"]
def post(body):
    return json.load(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:18199/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=600))
try:
    r = post({"model": "bench", "max_tokens": 64, "temperature": 0, "seed": 1, "messages": [{"role": "user", "content": text}]})
    print(f"{os.environ['B']:26s} {os.environ['KEY']:10s} n={r['usage']['completion_tokens']:2d} {r['choices'][0].get('finish_reason'):6s} {r['choices'][0]['message']['content'][:60]!r}")
except Exception as e: print(os.environ["B"], os.environ["KEY"], "ERROR", e)
PY
    kill $pid; wait $pid 2>/dev/null
  done
done
