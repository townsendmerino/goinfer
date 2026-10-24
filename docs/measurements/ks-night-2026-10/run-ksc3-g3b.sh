#!/bin/bash
# KS-C3's C3-G3b (docs/tasks/task-kernel-survey-2026-10.md, "KS-C3 read" and its 12:50 amendment): the int8 rows kernels
# against the one-row kernels, whole-token decode on Qwen2.5-Coder-1.5B's .gguf at int8int8 (TestKSC3_decodeSpeed), from a
# test binary built at 7629c7b4 (branch ksc3-w8-gemv-rows-2026-10). Night case (a): exits 75 when the fit guard refuses.
set -u
B=$HOME/goinfer-bench/ks-night-2026-10-23
LOG=$HOME/goinfer-logs/ksc3-g3b-$(date +%Y-%m-%d-%H%M).log
cd "$HOME/goinfer-bench/ks-night-wt/metal" || exit 1
(cd "$B/bin" && shasum -a 256 -c SHA256SUMS) || exit 1
echo "$(date '+%H:%M:%S') start: load $(sysctl -n vm.loadavg); log $LOG"
GOINFER_KSC3_TIMING=1 GOINFER_KSC3_MODEL=$HOME/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf \
  "$B/bin/metal-ksc3-7629c7b4.test" -test.v -test.count=1 -test.timeout 30m -test.run '^TestKSC3_decodeSpeed$' 2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}
grep -q 'FIT-REFUSED' "$LOG" && { echo "FIT REFUSED (exit 75)"; exit 75; }
echo "$(date '+%H:%M:%S') DONE rc=$rc: load $(sysctl -n vm.loadavg)"
exit $rc
