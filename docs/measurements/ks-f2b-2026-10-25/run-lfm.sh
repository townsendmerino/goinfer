#!/bin/bash
# KS-F2b's LFM2.5-2.6B cells (docs/measurements/ks-f2b-2026-10-25/README.md), pinned binaries in ~/goinfer-bench/ksf2b-night.
set -u
D=$HOME/goinfer-bench/ksf2b-night
(cd "$D" && shasum -a 256 -c SHA256SUMS) || exit 1
echo "$(date '+%F %T') start: load $(sysctl -n vm.loadavg) reclaimable $("$D/hostmem-cur" -reclaimable)"
python3 "$D/run_ksf2b.py" "$HOME/models/lfm2.5-2.6b" 128 8.7 "$D/ksf2b-readings.jsonl" || exit 1
python3 "$D/run_ksf2b.py" "$HOME/models/lfm2.5-2.6b" 1024 8.7 "$D/ksf2b-readings.jsonl"
rc=$?
echo "$(date '+%F %T') DONE rc=$rc"
exit $rc
