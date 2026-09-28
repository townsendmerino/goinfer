#!/usr/bin/env bash
# L1 S0 gates (pre-registered 7e17f520). goinfer branch f16-scales-s0 + aikit branch f16-scales-s0 via go.work.
set -u
cd /home/francis/mycode/goinfer
D=/home/francis/goinfer-bench/cpu-decode-2026-09-27
idle() { for i in $(seq 1 90); do l=$(cut -d' ' -f1 /proc/loadavg); awk -v l="$l" 'BEGIN{exit !(l<0.8)}' && return; echo "=== waiting for idle: loadavg $l"; sleep 20; done; }
idle; echo "=== $(date '+%F %T %Z') START S0 gate 1"
GOINFER_HEAVY_TESTS=1 GOINFER_NO_FIT_GUARD=1 go test -tags goinfer_testhooks ./decoder/ -run 'TestS0F16Scales_logitsBitIdentical$' -v -count=1 -timeout 30m > $D/s0-gate1-bitident.log 2>&1; rc=$?
echo "=== $(date '+%F %T %Z') END S0 gate 1 rc=$rc"
[ $rc -ne 0 ] && { echo "=== gate 1 failed: gate 2 not run"; exit 1; }
idle; echo "=== $(date '+%F %T %Z') START S0 gate 2"
GOINFER_HEAVY_TESTS=1 GOINFER_NO_FIT_GUARD=1 go test -tags goinfer_testhooks ./decoder/ -run 'TestS0F16Scales_AB$' -v -count=1 -timeout 45m > $D/s0-gate2-ab.log 2>&1; rc=$?
echo "=== $(date '+%F %T %Z') END S0 gate 2 rc=$rc"
