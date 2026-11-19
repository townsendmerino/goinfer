#!/bin/bash
# After the 14:30 PST power loss: re-run d3900 block 4 whole (as 104), then block 5, then void block d128 b1 (as 101).
export PATH=/usr/local/go/bin:$HOME/go/bin:$PATH TMPDIR=$HOME/goinfer-work/review-tmp/nobara-splitkv
WT=$HOME/mycode/splitkv-phi3; OUT=$HOME/goinfer-logs/splitkv-phi3-2026-11-19; BIN=$HOME/goinfer-work/splitkv-phi3/cuda-id.test
echo "$(date) waiting for boot load to settle (1-min load < 2.0)"
for i in $(seq 1 60); do awk '{exit !($1 < 2.0)}' /proc/loadavg && break; sleep 5; done
echo "$(date) load $(cut -d' ' -f1 /proc/loadavg)"
for spec in "3900 104" "3900 5" "128 101"; do
  set -- $spec
  if [ "$(date +%H%M)" -ge 2130 ]; then echo "STOP: 21:30 PT reached before d$1 b$2"; exit 3; fi
  python3 $WT/scripts/timing_lock.py run --wait 7200 --label "splitkv-phi3 d$1 b$2" -- \
    python3 $WT/docs/measurements/splitkv-phi3-2026-11-19/run_ab.py --depth $1 --block $2 --bin $BIN --out $OUT/cells.tsv --logdir $OUT/logs \
    || echo "block d$1 b$2 exited $?"
  sync
done
echo "$(date) RECOVER-DONE"
