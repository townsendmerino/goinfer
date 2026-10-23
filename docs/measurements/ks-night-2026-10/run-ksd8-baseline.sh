#!/bin/bash
# KS-D8 baseline (docs/tasks/task-kernel-survey-2026-10.md, "KS-D8 baseline"): NVIDIA-Nemotron-Nano-9B-v2 resident on Metal,
# a 512-token prompt's sequential prefill (TestKSD8_sequentialPrefillBaseline), from a test binary built at the registering
# commit. Night case (a): exits 75 when the fit guard refuses the load.
set -u
B=$HOME/goinfer-bench/ks-night-2026-10-23
LOG=$HOME/goinfer-logs/ksd8-baseline-$(date +%Y-%m-%d-%H%M).log
cd "$HOME/goinfer-bench/ks-night-wt/metal" || exit 1
(cd "$B/bin" && shasum -a 256 -c SHA256SUMS) || exit 1
echo "$(date '+%H:%M:%S') start: load $(sysctl -n vm.loadavg); log $LOG"
GOINFER_KSD8_BASELINE=1 "$B/bin/metal-ksd8.test" -test.v -test.count=1 -test.timeout 30m -test.run '^TestKSD8_sequentialPrefillBaseline$' 2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}
grep -q 'FIT-REFUSED' "$LOG" && { echo "FIT REFUSED (exit 75)"; exit 75; }
echo "$(date '+%H:%M:%S') DONE rc=$rc: load $(sysctl -n vm.loadavg)"
exit $rc
