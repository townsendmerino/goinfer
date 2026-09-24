#!/bin/bash
# External, out-of-process swap-growth kill switch for the S5 pool-mode M35 validation run.
# Independent of decoder's own S3 in-process tripwire (threshold +512MB) — R11(c)'s own lesson
# was that an in-process RSS read can look fine while swap actually explodes, so this watches
# REAL swap-used via sysctl directly and kills on a real delta, not on what the process reports
# about itself.
set -u
PID="$1"
LOG="$2"
BASELINE_MB=$(sysctl -n vm.swapusage | grep -oE 'used = [0-9.]+M' | grep -oE '[0-9.]+')
KILL_DELTA_MB=2048   # kill if swap-used grows more than 2 GB over this run's own baseline
DISK_FLOOR_GB=2       # kill if free disk on / drops under this (paranoia; pread is read-only)
START=$(date +%s)
echo "$(date '+%H:%M:%S') killwatch: baseline swap-used ${BASELINE_MB}MB, kill at +${KILL_DELTA_MB}MB, watching pid $PID" | tee -a "$LOG"

while kill -0 "$PID" 2>/dev/null; do
  NOW_MB=$(sysctl -n vm.swapusage | grep -oE 'used = [0-9.]+M' | grep -oE '[0-9.]+')
  RSS_KB=$(ps -o rss= -p "$PID" 2>/dev/null | tr -d ' ')
  FREE_GB=$(df -g / | tail -1 | awk '{print $4}')
  DELTA=$(echo "$NOW_MB - $BASELINE_MB" | bc)
  ELAPSED=$(( $(date +%s) - START ))
  echo "$(date '+%H:%M:%S') t+${ELAPSED}s swap_used=${NOW_MB}MB delta=+${DELTA}MB rss=${RSS_KB}KB free_disk=${FREE_GB}G" | tee -a "$LOG"

  if (( $(echo "$DELTA > $KILL_DELTA_MB" | bc -l) )); then
    echo "$(date '+%H:%M:%S') killwatch: SWAP DELTA EXCEEDED (+${DELTA}MB > ${KILL_DELTA_MB}MB) -- KILLING $PID" | tee -a "$LOG"
    kill -9 "$PID" 2>/dev/null
    exit 1
  fi
  if [ "$FREE_GB" -lt "$DISK_FLOOR_GB" ]; then
    echo "$(date '+%H:%M:%S') killwatch: FREE DISK BELOW FLOOR (${FREE_GB}G < ${DISK_FLOOR_GB}G) -- KILLING $PID" | tee -a "$LOG"
    kill -9 "$PID" 2>/dev/null
    exit 1
  fi
  sleep 3
done
echo "$(date '+%H:%M:%S') killwatch: pid $PID exited on its own, stopping watch" | tee -a "$LOG"
