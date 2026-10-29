#!/bin/bash
cd $HOME/goinfer-bench/mac-sb11m-wt/metal
deadline=$(( $(date +%s) + 3000 ))
n=0
while :; do
  n=$((n+1))
  LOG=$HOME/goinfer-logs/sb11/g-sb11-18-$(date +%Y%m%d-%H%M%S)-try$n.log
  { echo "== commit $(git rev-parse HEAD)"; (cd .. && go run ./cmd/hostmem -reclaimable 2>&1 | head -3)
    GOINFER_SB11_OFFCOST_MODEL=$HOME/models/qwen2.5-coder-0.5b-instruct-q4_k_m.int4.metal.v17.giw \
      python3 ../scripts/timing_lock.py run --wait 1200 --label "mac-sb11m G-SB11-18" -- $HOME/.claude/jobs/756f841f/tmp/inner.sh
    echo "== rc $?"; } > $LOG 2>&1
  rc=$(tail -1 $LOG | awk '{print $3}')
  echo "$(date '+%T') try $n rc $rc log $LOG"
  [ "$rc" != 75 ] && { cat $LOG; exit 0; }
  [ $(date +%s) -gt $deadline ] && { echo "TIMEOUT"; cat $LOG; exit 3; }
  sleep 45
done
