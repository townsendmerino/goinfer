#!/bin/bash
# KH1-G5 (docs/tasks/task-kernel-survey-2026-10.md, "KS-H1 gates" and its 2026-10-23 amendment): Qwen3.5-9B on Metal, one
# ~20K-token cold prompt per fresh serve, its footprint and RSS sampled by mem0_darwin.py; serve built at 421a4347 (main just
# before KS-H1's merge) then at a570f8df (the merge). Night case (a): exits 75 when the fit guard refuses a load.
set -u
B=$HOME/goinfer-bench/ks-night-2026-10-23
REPO=$HOME/goinfer-bench/ks-night-wt
OUT=$HOME/goinfer-logs/kh1-g5-$(date +%Y-%m-%d-%H%M)
cd "$REPO" || exit 1
(cd "$B/bin" && shasum -a 256 -c SHA256SUMS) || exit 1
echo "$(date '+%H:%M:%S') start: load $(sysctl -n vm.loadavg); out $OUT"
for rev in 421a4347 a570f8df; do
  python3 -u docs/measurements/mem0-2026-10/mem0_darwin.py --serve "$B/bin/serve-metal-$rev" --out "$OUT/$rev" --repo "$REPO" --cell metal:qwen35-9b
  rc=$?
  if grep -q 'NOT loaded' "$OUT/$rev/serve-metal-qwen35-9b.log" 2>/dev/null; then echo "FIT REFUSED at $rev (exit 75)"; exit 75; fi
  [ $rc -eq 0 ] || { echo "FAILED at $rev, rc $rc"; exit $rc; }
done
echo "$(date '+%H:%M:%S') DONE: load $(sysctl -n vm.loadavg)"
