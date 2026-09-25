#!/bin/bash
L=~/goinfer-logs/s2
run() { # label, port, model args...
  lab=$1; port=$2; shift 2
  /tmp/serve-s2-linux -addr 127.0.0.1:$port -quant int4 -ctx 1024 "$@" > $L/greedy-$lab.server.log 2>&1 &
  P=$!
  for i in $(seq 1 600); do curl -sf 127.0.0.1:$port/v1/models >/dev/null 2>&1 && break; kill -0 $P 2>/dev/null || break; sleep 1; done
  curl -s 127.0.0.1:$port/v1/completions -H "Content-Type: application/json" \
    -d "{\"model\":\"m\",\"prompt\":\"The capital of France is\",\"max_tokens\":16,\"temperature\":0}" > $L/greedy-$lab.json
  kill $P; wait $P 2>/dev/null
  echo "$lab $(date +%T) $(python3 -c "import json;print(repr(json.load(open(\"$L/greedy-$lab.json\"))[\"choices\"][0][\"text\"]))" 2>&1)" >> $L/greedy.txt
}
rm -f $L/greedy.txt
run direct 18301 -model m=/home/francis/models/gemma4-26b-q4_k_m.gguf -direct-load
run sidecar 18302 -model m=/home/francis/models/s2-stream2.int4.metal.giw
echo DONE >> $L/greedy.txt
