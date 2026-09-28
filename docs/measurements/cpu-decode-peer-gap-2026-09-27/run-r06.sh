#!/usr/bin/env bash
# L2 (R-06 default-on) gates 1 and 3, docs/tasks/task-cpu-decode-peer-gap-2026-09.md. Tree cc1769c7 + the test edits.
set -u
cd /home/francis/mycode/goinfer
D=/home/francis/goinfer-bench/cpu-decode-2026-09-27
for i in $(seq 1 90); do l=$(cut -d' ' -f1 /proc/loadavg); awk -v l="$l" 'BEGIN{exit !(l<0.8)}' && break; echo "=== waiting for idle: loadavg $l"; sleep 20; done
echo "=== $(date '+%F %T %Z') START gate 1 (logits bit-identical)"
GOINFER_HEAVY_TESTS=1 GOINFER_NO_FIT_GUARD=1 go test -tags goinfer_testhooks ./decoder/ -run 'TestCPURoofline_w4a8Batch_logitsBitIdentical$' -v -count=1 -timeout 30m > $D/r06-gate1-bitident.log 2>&1; rc=$?
echo "=== $(date '+%F %T %Z') END gate 1 rc=$rc"
for i in $(seq 1 90); do l=$(cut -d' ' -f1 /proc/loadavg); awk -v l="$l" 'BEGIN{exit !(l<0.8)}' && break; echo "=== waiting for idle: loadavg $l"; sleep 20; done
echo "=== $(date '+%F %T %Z') START gate 3 (paired A/B)"
GOINFER_HEAVY_TESTS=1 go test -tags goinfer_testhooks ./decoder/ -run 'TestCPURoofline_w4a8Batch$' -v -count=1 -timeout 45m > $D/r06-gate3-ab.log 2>&1; rc=$?
echo "=== $(date '+%F %T %Z') END gate 3 rc=$rc"
