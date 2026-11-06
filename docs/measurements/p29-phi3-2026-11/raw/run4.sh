#!/bin/bash
# P29 Phi-3 diagnosis, run 4: Phi-3 only, waits for load1 <= 4.0 AND reclaimable memory >= 7.7 GB (the fit need 5.68 GB + 2 GB).
set -u
cd "$HOME/tmcode/goinfer-fit"
OUT="$HOME/goinfer-work/p29-phi3/out"
PHI3="$HOME/models/phi3-mini-4k-gguf/Phi-3-mini-4k-instruct-q4.gguf"
echo "start $(date '+%F %T')"
for i in $(seq 1 90); do
  l=$(sysctl -n vm.loadavg | awk '{print $2}'); m=$(go run ./cmd/hostmem -reclaimable 2>&1 | tail -1)
  awk -v l="$l" -v m="$m" 'BEGIN{exit !(l<=4.0 && m>=7.7e9)}' && break
  sleep 10
done
echo "== probe phi3  $(date '+%T')  load1 $(sysctl -n vm.loadavg | awk '{print $2}')  reclaimable bytes $(go run ./cmd/hostmem -reclaimable 2>&1 | tail -1)"
(cd backend/metal && GOINFER_P29_PROBE="$PHI3" GOMAXPROCS=4 caffeinate -i "$OUT/metal.test" -test.run '^TestP29Probe$' -test.v -test.timeout 20m 2>&1 | grep -E "^\[probe\]|^(---|ok|FAIL|PASS)|SKIP|panic")
echo "end $(date '+%F %T')"
