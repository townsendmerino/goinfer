#!/bin/bash
# KS-O1a's O1a-G3 (with O1a-G2 inside it) by day: TestKSO1a_phi3Speed from a test binary built at the commit named in
# $HOME/goinfer-bench/kso1-bin/SHA256SUMS, under the timing lock held exclusively. It waits up to 10 minutes for the 1-minute
# load to fall under 3.5, then takes the lock and checks the load again against the void line (4.0, half the Mac's 8 cores), and
# records load, available memory and swap at the start. Exits 75 when the load never falls or is over the line under the lock,
# and on a fit refusal.
set -u
B=$HOME/goinfer-bench/kso1-bin
LOG=$HOME/goinfer-logs/kso1a-g3-$(date +%Y-%m-%d-%H%M).log
cd "$HOME/goinfer-bench/ks-o1/metal" || exit 1
(cd "$B" && shasum -a 256 -c SHA256SUMS) || exit 1
calm=0
for _ in $(seq 1 60); do
  awk -v l="$(sysctl -n vm.loadavg | awk '{print $2}')" 'BEGIN{exit !(l<3.5)}' && { calm=1; break; }
  sleep 10
done
if [ "$calm" != 1 ]; then
  echo "VOID: the 1-minute load never fell under 3.5 in 10 minutes ($(sysctl -n vm.loadavg)); not timed (exit 75)" | tee "$LOG"
  exit 75
fi
python3 ../scripts/timing_lock.py run --wait 3600 --label kso1a-g3 -- bash -c '
  l=$(sysctl -n vm.loadavg | awk "{print \$2}")
  if ! awk -v l="$l" "BEGIN{exit !(l<4.0)}"; then
    echo "VOID: the 1-minute load is $l under the lock, over the 4.0 line; not timed (exit 75)"
    exit 75
  fi
  echo "$(date "+%H:%M:%S") start: load $(sysctl -n vm.loadavg); available $('"$B"'/hostmem | head -1) bytes; swap $(sysctl -n vm.swapusage)"
  GOINFER_KSO1_TIMING=1 '"$B"'/metal-kso1.test -test.v -test.count=1 -test.timeout 30m -test.run "^TestKSO1a_phi3Speed$"
  rc=$?
  echo "$(date "+%H:%M:%S") DONE rc=$rc: load $(sysctl -n vm.loadavg); swap $(sysctl -n vm.swapusage)"
  exit $rc' 2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}
grep -q 'FIT-REFUSED\|was NOT loaded' "$LOG" && { echo "FIT REFUSED (exit 75)"; exit 75; }
exit $rc
