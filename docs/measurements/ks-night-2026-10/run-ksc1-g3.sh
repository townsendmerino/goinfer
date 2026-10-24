#!/bin/bash
# KS-C1's C1-G3 (docs/tasks/task-kernel-survey-2026-10.md, "KS-C1 (a)+(b) gates"): G-Q2 (TestQ4KLane_realPhi3NonInferiority) with
# the new q4k-lane GEMVs, from a test binary built at b4165838 (branch ksc1-q4k-gemv-2026-10). Night case (a): exits 75 when the fit
# guard refuses Phi-3.
set -u
B=$HOME/goinfer-bench/ks-night-2026-10-23
LOG=$HOME/goinfer-logs/ksc1-g3-$(date +%Y-%m-%d-%H%M).log
cd "$HOME/goinfer-bench/ks-night-wt/metal" || exit 1
(cd "$B/bin" && shasum -a 256 -c SHA256SUMS) || exit 1
echo "$(date '+%H:%M:%S') start: load $(sysctl -n vm.loadavg); log $LOG"
GOINFER_HEAVY_TESTS=1 "$B/bin/metal-ksc1-b4165838.test" -test.v -test.count=1 -test.timeout 60m -test.run '^TestQ4KLane_realPhi3NonInferiority$' 2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}
grep -q 'was NOT loaded\|declines' "$LOG" && { echo "FIT REFUSED (exit 75)"; exit 75; }
echo "$(date '+%H:%M:%S') DONE rc=$rc: load $(sysctl -n vm.loadavg)"
exit $rc
