#!/bin/bash
# system-wide wired memory (MB) around load and the first request, alias vs copy, on the 1.5B
MODEL=$HOME/models/qwen2.5-coder-1.5b-instruct-q4_k_m.int4.metal.giw
wired() { vm_stat | awk '/Pages wired down/{gsub("\\.","",$4); printf "%d", $4*16384/1048576}'; }
for A in 0 1 0 1; do
  W0=$(wired)
  GOINFER_METAL_ALIAS=$A /tmp/goinfer-metal-serve-s6 -addr 127.0.0.1:1813$A -model $MODEL -backend metal -ctx 512 > /tmp/wired-$A.log 2>&1 &
  P=$!
  for i in $(seq 1 60); do grep -q "goinfer serving on" /tmp/wired-$A.log && break; sleep 0.5; done
  W1=$(wired)
  curl -s -m 60 http://127.0.0.1:1813$A/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"x","messages":[{"role":"user","content":"Say hi."}],"max_tokens":16,"temperature":0}' > /dev/null
  W2=$(wired)
  sleep 1; W3=$(wired)
  kill $P; wait $P 2>/dev/null; sleep 1; W4=$(wired)
  echo "alias=$A wired MB: before=$W0 loaded=$W1 after-request=$W2 (+1s: $W3) after-exit=$W4"
done
