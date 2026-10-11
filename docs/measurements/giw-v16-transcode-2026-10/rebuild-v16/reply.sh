#!/bin/bash
# reply.sh <serve binary> <model file> <port> <log>: serve the model on the CPU, ask one fixed question greedily, print
# the reply's JSON, and stop the server by its PID.
bin=$1; model=$2; port=$3; log=$4
"$bin" -model "$model" -backend cpu -addr 127.0.0.1:$port > "$log" 2>&1 &
pid=$!
for i in $(seq 1 600); do
  grep -q 'goinfer serving on' "$log" && break
  kill -0 $pid 2>/dev/null || { echo "serve exited before it was serving: $(tail -2 "$log")"; exit 1; }
  sleep 1
done
name=$(curl -s -m 20 127.0.0.1:$port/v1/models | python3 -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])')
curl -s -m 900 127.0.0.1:$port/v1/chat/completions -H 'content-type: application/json' -d '{"model":"'"$name"'","max_tokens":24,"temperature":0,"messages":[{"role":"user","content":"List the first five prime numbers and say in one sentence why 1 is not among them."}]}' | python3 -c 'import json,sys; d=json.load(sys.stdin); print(json.dumps({"content": d["choices"][0]["message"]["content"], "finish": d["choices"][0]["finish_reason"], "usage": d.get("usage")}))'
kill $pid 2>/dev/null; wait $pid 2>/dev/null
