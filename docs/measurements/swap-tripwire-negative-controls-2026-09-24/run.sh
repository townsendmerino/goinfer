#!/bin/bash
# S3 negative controls: the serving swap guard (default +512 MB) must never trip across 100 completions,
# on a 1.5B .gguf through its sidecar (CPU) and a 7B .gguf on Metal. Swap-used is also sampled externally.
cd /Users/francistownsend-merino/tmcode/goinfer
O=/tmp/s3neg
unset GOINFER_SWAP_GUARD GOINFER_GGUF_DIRECT
( while true; do echo "$(date +%T) $(sysctl -n vm.swapusage)"; sleep 1; done ) > $O/swap-external.log 2>&1 &
SW=$!
run() { # label, port, extra args
  L=$1; P=$2; shift 2
  echo "== $L start $(date +%T)" >> $O/progress.txt
  /tmp/serve-s3neg -addr 127.0.0.1:$P "$@" > $O/$L.server.log 2>&1 &
  S=$!
  for i in $(seq 1 900); do curl -sf 127.0.0.1:$P/v1/models >/dev/null 2>&1 && break; kill -0 $S 2>/dev/null || break; sleep 1; done
  echo "   ready $(date +%T)" >> $O/progress.txt
  ok=0; bad=0
  prompts=("Write a haiku about the sea." "Explain what a hash map is in two sentences." "List three prime numbers and why they are prime." "def fib(n):" "Summarize the water cycle.")
  for n in $(seq 1 100); do
    pr=${prompts[$(( n % 5 ))]}
    code=$(curl -s -o $O/$L.last.json -w '%{http_code}' 127.0.0.1:$P/v1/completions -H 'Content-Type: application/json' \
      -d "{\"model\":\"m\",\"prompt\":\"$pr\",\"max_tokens\":64,\"temperature\":0}")
    if [ "$code" = 200 ]; then ok=$((ok+1)); else bad=$((bad+1)); echo "   #$n HTTP $code $(head -c 200 $O/$L.last.json)" >> $O/progress.txt; fi
    [ $((n % 20)) = 0 ] && echo "   $L $n/100 ok=$ok bad=$bad $(date +%T) $(sysctl -n vm.swapusage | grep -oE 'used = [0-9.]+M')" >> $O/progress.txt
  done
  kill $S; wait $S 2>/dev/null
  echo "== $L done $(date +%T) ok=$ok bad=$bad trips=$(grep -c 'TRIPPED' $O/$L.server.log) guard-lines: $(grep -i 'swap guard' $O/$L.server.log | head -3 | tr '\n' '|')" >> $O/progress.txt
}
run s15b-sidecar-cpu 18401 -model m=$HOME/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf -backend cpu -quant int4
run s7b-metal 18402 -model m=$HOME/models/qwen2.5-7b-instruct-q4_k_m.gguf -backend metal -quant int4
kill $SW
echo ALLDONE $(date +%T) >> $O/progress.txt
