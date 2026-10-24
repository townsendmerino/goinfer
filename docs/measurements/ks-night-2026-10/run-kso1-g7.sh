#!/bin/bash
# KS-O1's O1-G7 by day: TestKSO1_speed from a test binary built at the commit named in
# $HOME/goinfer-bench/kso1-bin/SHA256SUMS, under the timing lock held exclusively. It waits for the 1-minute load to fall
# under 3.5 first and records load, available memory and swap at the start. Exits 75 on a fit refusal.
set -u
B=$HOME/goinfer-bench/kso1-bin
LOG=$HOME/goinfer-logs/kso1-g7-$(date +%Y-%m-%d-%H%M).log
cd "$HOME/goinfer-bench/ks-o1/metal" || exit 1
(cd "$B" && shasum -a 256 -c SHA256SUMS) || exit 1
for _ in $(seq 1 60); do
  awk -v l="$(sysctl -n vm.loadavg | awk '{print $2}')" 'BEGIN{exit !(l<3.5)}' && break
  sleep 10
done
python3 ../scripts/timing_lock.py run --wait 3600 --label kso1-g7 -- bash -c '
  echo "$(date "+%H:%M:%S") start: load $(sysctl -n vm.loadavg); available $('"$B"'/hostmem | head -1) bytes; swap $(sysctl -n vm.swapusage)"
  GOINFER_KSO1_TIMING=1 '"$B"'/metal-kso1.test -test.v -test.count=1 -test.timeout 30m -test.run "^TestKSO1_speed$"
  rc=$?
  echo "$(date "+%H:%M:%S") DONE rc=$rc: load $(sysctl -n vm.loadavg); swap $(sysctl -n vm.swapusage)"
  exit $rc' 2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}
grep -q 'FIT-REFUSED\|was NOT loaded' "$LOG" && { echo "FIT REFUSED (exit 75)"; exit 75; }
exit $rc
