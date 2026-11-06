#!/bin/bash
# P29 Phi-3 diagnosis: fit check, build, probe on Phi-3 and the 0.5B. Runs under the exclusive timing lock.
set -u
cd "$HOME/tmcode/goinfer-fit"
OUT="$HOME/goinfer-work/p29-phi3/out"
PHI3="$HOME/models/phi3-mini-4k-gguf/Phi-3-mini-4k-instruct-q4.gguf"
Q05="$HOME/models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf"
echo "start $(date '+%F %T')  load: $(uptime | sed 's/.*load averages: //')" 
echo "free (reclaimable): $(go run ./cmd/hostmem -reclaimable 2>&1 | tail -1)"
echo "== fit check (phi3, q4k, ctx 2048)"
(cd backend/metal && go run ./cmd/chat fit "$PHI3" -quant q4k -ctx 2048 2>&1 | tail -12)
echo "== build"
(cd backend/metal && go test -c -tags goinfer_testhooks -o "$OUT/metal.test" . 2>&1 | tail -5) || exit 1
for M in "$PHI3" "$Q05"; do
  echo "== probe $(basename "$M")  $(date '+%T')  load: $(uptime | sed 's/.*load averages: //')"
  (cd backend/metal && GOINFER_P29_PROBE="$M" GOMAXPROCS=4 caffeinate -i "$OUT/metal.test" -test.run '^TestP29Probe$' -test.v -test.timeout 20m 2>&1 | grep -E "^\[probe\]|^(---|ok|FAIL|PASS)|SKIP|panic")
done
echo "end $(date '+%F %T')"
