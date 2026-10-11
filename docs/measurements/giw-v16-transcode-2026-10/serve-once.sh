#!/bin/bash
# serve-once.sh <binary> <model> <log> <port>: start serve on Metal, wait until it is serving, send one chat request, stop it by PID.
bin=$1; model=$2; log=$3; port=$4
"$bin" -model "$model" -backend metal -addr 127.0.0.1:$port > "$log" 2>&1 &
pid=$!
for i in $(seq 1 240); do
  grep -q 'goinfer serving on' "$log" && break
  kill -0 $pid 2>/dev/null || { echo "serve exited before it was serving"; break; }
  sleep 1
done
if grep -q 'goinfer serving on' "$log"; then
  curl -s -m 60 127.0.0.1:$port/v1/chat/completions -H 'content-type: application/json' -d '{"model":"'"$5"'","max_tokens":12,"temperature":0,"messages":[{"role":"user","content":"Say hello in three words."}]}' | head -c 400; echo
fi
kill $pid 2>/dev/null; wait $pid 2>/dev/null
echo "serve pid $pid stopped"
