#!/usr/bin/env bash
# One greedy request of the 0.5B:8000 prompt to the pinned CUDA serve at the cell's flags; prints usage and finish_reason.
W=$HOME/goinfer-bench/peer-reanchor-2026-11-19; O=$HOME/goinfer-work/repro8000
$W/bin/serve-cuda-e29da447 -model bench=$HOME/models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf -backend cuda -addr 127.0.0.1:18199 -ctx 8192 -quant int4 > $O/serve.log 2>&1 & pid=$!
for _ in $(seq 1 120); do grep -q 'goinfer serving on' $O/serve.log && break; sleep 0.5; done
python3 - <<'PY'
import json, urllib.request
p = json.load(open("scripts/prompts.json"))["0.5B:8000"]
text = p["texts"][0] if "texts" in p else p["text"]
for ctx_note, mt in (("max_tokens 64", 64),):
    body = {"model": "bench", "max_tokens": mt, "temperature": 0, "seed": 1, "messages": [{"role": "user", "content": text}]}
    r = json.load(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:18199/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=300))
    ch = r["choices"][0]
    print(ctx_note, "usage", r.get("usage"), "finish_reason", ch.get("finish_reason"), "content", repr(ch["message"].get("content"))[:200])
    print("other keys", {k: v for k, v in r.items() if k not in ("choices", "usage")})
print("calibrated tokens", p.get("tokens"))
PY
kill $pid; wait $pid 2>/dev/null
grep -iE "context|ctx|truncat|exceed|length|limit|warn" $O/serve.log | grep -v "^  harness\|request body cap" | tail -8
