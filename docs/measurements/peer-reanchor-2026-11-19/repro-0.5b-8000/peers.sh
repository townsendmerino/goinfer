#!/usr/bin/env bash
O=$HOME/goinfer-work/repro8000
unset LD_LIBRARY_PATH
OLLAMA_MODELS=$HOME/ollama-0325/models OLLAMA_HOST=127.0.0.1:18300 $HOME/ollama-0325/bin/ollama serve > $O/ollama.log 2>&1 & op=$!
$HOME/mycode/peers/llama.cpp/build/bin/llama-server --model $HOME/models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf --port 18301 --host 127.0.0.1 --ctx-size 8192 -ngl 99 > $O/llama.log 2>&1 & lp=$!
for _ in $(seq 1 120); do curl -s -m 1 http://127.0.0.1:18301/health | grep -q ok && curl -s -m1 http://127.0.0.1:18300/ >/dev/null && break; sleep 0.5; done
python3 - <<'PY'
import json, urllib.request
p = json.load(open("scripts/prompts.json"))["0.5B:8000"]; text = p["texts"][0] if "texts" in p else p["text"]
def post(url, body):
    return json.load(urllib.request.urlopen(urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=600))
r = post("http://127.0.0.1:18300/api/chat", {"model": "q05", "stream": False, "messages": [{"role": "user", "content": text}], "options": {"num_predict": 64, "num_ctx": 8192, "temperature": 0, "seed": 1}})
print("ollama   n=%s done=%s %r" % (r.get("eval_count"), r.get("done_reason"), r["message"]["content"][:110]))
r = post("http://127.0.0.1:18301/v1/chat/completions", {"max_tokens": 64, "temperature": 0, "seed": 1, "messages": [{"role": "user", "content": text}]})
print("llamacpp n=%s finish=%s %r" % (r["usage"]["completion_tokens"], r["choices"][0].get("finish_reason"), r["choices"][0]["message"]["content"][:110]))
PY
kill $op $lp; wait $op $lp 2>/dev/null
