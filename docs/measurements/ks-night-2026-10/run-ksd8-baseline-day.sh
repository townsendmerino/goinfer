#!/bin/bash
# The KS-D8 baseline by day, 2026-10-24: run-ksd8-baseline.sh's instrument from a test binary built at 0140adf1 (the registering
# commit) with #203's hostmem reading applied (internal/hostmem, decoder/hostram_darwin.go; nothing the read measures), so the fit
# guard reads memory_pressure's figure. Exits 75 on a fit refusal.
set -u
B=$HOME/goinfer-bench/ks-day-2026-10-24
LOG=$HOME/goinfer-logs/ksd8-baseline-day-$(date +%Y-%m-%d-%H%M).log
cd "$HOME/goinfer-bench/ks-day-0140adf1-wt/metal" || exit 1
(cd "$B/bin" && shasum -a 256 -c SHA256SUMS) || exit 1
echo "$(date '+%H:%M:%S') start: load $(sysctl -n vm.loadavg); swap $(sysctl -n vm.swapusage); log $LOG"
GOINFER_KSD8_BASELINE=1 "$B/bin/metal-ksd8-0140adf1-hm203.test" -test.v -test.count=1 -test.timeout 30m -test.run '^TestKSD8_sequentialPrefillBaseline$' 2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}
grep -q 'FIT-REFUSED\|was NOT loaded' "$LOG" && { echo "FIT REFUSED (exit 75)"; exit 75; }
echo "$(date '+%H:%M:%S') DONE rc=$rc: load $(sysctl -n vm.loadavg); swap $(sysctl -n vm.swapusage)"
exit $rc
