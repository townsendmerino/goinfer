#!/bin/bash
# MC3 per-pass prefill attribution (docs/measurements/mc3-prefill-attr-2026-09-28.md), Mac night queue. Pinned: the serve
# binary was built from 9e1d31c8, and the harness copy lives in the job dir. Runs under night.py (TE9 lock held).
set -u
B=$HOME/goinfer-bench/mc3-prefill-attr-2026-09-28
for f in serve-metal-9e1d31c8 bench_w7_plain.py mc3_prefill_attr.py; do
  [ -e "$B/$f" ] || { echo "missing $B/$f"; exit 1; }
done
echo "=== $(date '+%F %T %Z') START mc3-prefill-attr"
python3 -B "$B/mc3_prefill_attr.py" "$B" --reps 3 --max-tokens 128 --clients 4 --models 1.5B,7B 2>&1 | tee -a "$B/run.log"
rc=${PIPESTATUS[0]}
echo "=== $(date '+%F %T %Z') END rc=$rc"
exit "$rc"
