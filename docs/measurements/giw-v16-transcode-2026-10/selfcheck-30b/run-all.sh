#!/bin/bash
# The 30B self-check investigation (docs/measurements/giw-v16-transcode-2026-10, "the 30B's self-check"): three sampled
# transcodes with one fast build, each output removed before the next as the night driver does, then the reference
# load timings on the last output. Exploratory. Estimate: 10 min.
set -uo pipefail
D=$HOME/goinfer-bench/selfcheck
BIN=$HOME/goinfer-bench/giw-v16/bin-t8/prequant-new
echo "started $(date '+%Y-%m-%d %H:%M:%S %Z'); $(uptime | sed 's/.*load/load/'); $(grep -E 'MemAvailable|^Cached|Dirty' /proc/meminfo | tr -s ' ' | tr '\n' ' ')"
for r in r1 r2 r3; do
  find "$D/out" -name '30b.*' -delete
  bash "$D/e2.sh" $r "$BIN"
done
bash "$D/e1.sh" "$D/out/30b.r3.giw"
find "$D/out" -name '30b.*' -delete
echo "finished $(date '+%H:%M:%S')"
