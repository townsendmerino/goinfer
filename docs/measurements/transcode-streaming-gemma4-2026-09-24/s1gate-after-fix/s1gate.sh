#!/bin/bash
O=/home/francis/goinfer-logs/s2/s1gate-2026-09-24; mkdir -p $O
arm() { lab=$1; port=$2; shift 2
  /tmp/serve-fix-linux -addr 127.0.0.1:$port -quant int4 -ctx 1024 "$@" > $O/$lab.server.log 2>&1 & P=$!
  for i in $(seq 1 900); do curl -sf 127.0.0.1:$port/v1/models >/dev/null 2>&1 && break; kill -0 $P 2>/dev/null || break; sleep 1; done
  n=0; for pr in "The capital of France is" "def quicksort(arr):" "Three facts about the Moon:"; do n=$((n+1))
    curl -s 127.0.0.1:$port/v1/completions -H "Content-Type: application/json" -d "{\"model\":\"m\",\"prompt\":\"$pr\",\"max_tokens\":64,\"temperature\":0}" > $O/$lab.$n.json
  done
  kill $P; wait $P 2>/dev/null; echo "$lab done $(date +%T)" >> $O/progress.txt
}
echo "start $(date +%T)" > $O/progress.txt
arm sidecar 18307 -model m=/home/francis/models/gemma4-26b-q4_k_m.gguf
arm direct 18308 -model m=/home/francis/models/gemma4-26b-q4_k_m.gguf -direct-load
python3 - >> $O/progress.txt <<PY
import json
for n in (1,2,3):
    a=json.load(open("$O/sidecar.%d.json"%n))["choices"][0]; d=json.load(open("$O/direct.%d.json"%n))
    print("prompt %d: identical=%s  tokens=%s  sidecar=%r"%(n, a["text"]==d["choices"][0]["text"], d["usage"]["completion_tokens"], a["text"][:70]))
PY
echo ALLDONE >> $O/progress.txt
