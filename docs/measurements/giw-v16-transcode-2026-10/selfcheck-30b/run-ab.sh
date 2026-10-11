#!/bin/bash
# The 30B self-check investigation, the intervention: the same build with and without an fsync of the bundle before
# its self-check, four pairs in alternating order, each run sampled. Exploratory. Estimate: 14 min.
set -uo pipefail
D=$HOME/goinfer-bench/selfcheck
BASE=$HOME/goinfer-bench/giw-v16/bin-t8/prequant-new
FS=$D/prequant-fsync
echo "started $(date '+%Y-%m-%d %H:%M:%S %Z'); $(uptime | sed 's/.*load/load/')"
for spec in "b1 $BASE" "f1 $FS" "f2 $FS" "b2 $BASE" "b3 $BASE" "f3 $FS" "f4 $FS" "b4 $BASE"; do
  set -- $spec
  find "$D/out" -name '30b.*' -delete
  bash "$D/e2.sh" $1 "$2"
done
find "$D/out" -name '30b.*' -delete
echo "finished $(date '+%H:%M:%S')"
