#!/bin/bash
# The 413 prefill-share fix end to end (docs/measurements/fix413-e2e-2026-09-28.md), Mac night queue. Pinned: both serve
# binaries were built from 33217487 (fixed / fix reverted), and the harness copies and frozen prompts live in the job
# dir, so the tree moving before tonight changes nothing. Runs under night.py (the TE9 timing lock is held).
set -u
B=$HOME/goinfer-bench/fix413-e2e-2026-09-28
for f in serve-metal-fixed-33217487 serve-metal-unfixed-33217487 ballast.py fix413_e2e.py bench_spec_copy.py \
         bench_w7_plain.py sections.json; do
  [ -e "$B/$f" ] || { echo "missing $B/$f"; exit 1; }
done
echo "=== $(date '+%F %T %Z') START fix413-e2e; swap $(sysctl -n vm.swapusage)"
python3 -B "$B/fix413_e2e.py" "$B" 2>&1 | tee -a "$B/run.log"
rc=${PIPESTATUS[0]}
echo "=== $(date '+%F %T %Z') END rc=$rc; swap $(sysctl -n vm.swapusage)"
exit "$rc"
