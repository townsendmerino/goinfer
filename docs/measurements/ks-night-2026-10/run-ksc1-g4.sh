#!/bin/bash
# KS-C1's C1-G4 (docs/tasks/task-kernel-survey-2026-10.md, "KS-C1 (a)+(b) gates"): Phi-3 mini 4k at --quant q4k, the q4k lane's
# new rows kernels against the old one-row kernels on one resident, whole-token decode (TestKSC1_phi3Speed), from a test binary
# built at ce9547d7 (branch ksc1-q4k-gemv-2026-10). Night case (a): exits 75 when the fit guard refuses Phi-3.
set -u
B=$HOME/goinfer-bench/ks-night-2026-10-23
LOG=$HOME/goinfer-logs/ksc1-g4-$(date +%Y-%m-%d-%H%M).log
cd "$HOME/goinfer-bench/ks-night-wt/metal" || exit 1
(cd "$B/bin" && shasum -a 256 -c SHA256SUMS) || exit 1
echo "$(date '+%H:%M:%S') start: load $(sysctl -n vm.loadavg); log $LOG"
GOINFER_KSC1_TIMING=1 "$B/bin/metal-ksc1-ce9547d7.test" -test.v -test.count=1 -test.timeout 30m -test.run '^TestKSC1_phi3Speed$' 2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}
grep -q 'FIT-REFUSED' "$LOG" && { echo "FIT REFUSED (exit 75)"; exit 75; }
echo "$(date '+%H:%M:%S') DONE rc=$rc: load $(sysctl -n vm.loadavg)"
exit $rc
