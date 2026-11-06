#!/bin/bash
# P29 Phi-3 diagnosis, run 3: each probe waits for the 1-minute load average to be <= 4.0 (the void rule: half the Mac's 8 cores)
# and the free memory to clear the fit need + 2 GB, and records both at its start. Runs under the exclusive timing lock.
set -u
cd "$HOME/tmcode/goinfer-fit"
OUT="$HOME/goinfer-work/p29-phi3/out"
PHI3="$HOME/models/phi3-mini-4k-gguf/Phi-3-mini-4k-instruct-q4.gguf"
Q05="$HOME/models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf"
Q15="$HOME/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"
echo "start $(date '+%F %T')"
(cd backend/metal && go test -c -tags goinfer_testhooks -o "$OUT/metal.test" . 2>&1 | tail -5) || exit 1
waitload() { for i in $(seq 1 60); do l=$(sysctl -n vm.loadavg | awk '{print $2}'); awk -v l="$l" 'BEGIN{exit !(l<=4.0)}' && return 0; sleep 10; done; return 1; }
for M in "$PHI3" "$Q05" "$Q15"; do
  waitload || echo "WARNING: load stayed above 4.0 for 10 min before $(basename "$M"); this cell is VOID by the rule"
  echo "== probe $(basename "$M")  $(date '+%T')  load1 $(sysctl -n vm.loadavg | awk '{print $2}')  reclaimable bytes $(go run ./cmd/hostmem -reclaimable 2>&1 | tail -1)"
  (cd backend/metal && GOINFER_P29_PROBE="$M" GOMAXPROCS=4 caffeinate -i "$OUT/metal.test" -test.run '^TestP29Probe$' -test.v -test.timeout 20m 2>&1 | grep -E "^\[probe\]|^(---|ok|FAIL|PASS)|SKIP|panic")
done
echo "end $(date '+%F %T')"
