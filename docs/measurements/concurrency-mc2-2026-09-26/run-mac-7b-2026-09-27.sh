#!/bin/bash
# The Mac 7B MC2 cell (docs/tasks/task-concurrency-2026-09.md MC3c, registered in 796ed628): decides whether
# -cpu-batch auto batches on darwin. M1 Pro, qwen2.5-7b-instruct int4 via its .int4.cpu-arm64.giw.
set -u
cd /Users/francistownsend-merino/tmcode/goinfer || exit 1
B=/Users/francistownsend-merino/goinfer-bench/mac7b-cpu-2026-09-27
M=$HOME/models/qwen2.5-7b-instruct-q4_k_m.int4.cpu-arm64.giw
ts() { date '+%H:%M:%S'; }
gate() { waited=0; while :; do l1=$(sysctl -n vm.loadavg | awk '{print $2}'); busy=$(pgrep -f "go test|serve-metal|serve-cuda|bench_" | grep -v $$ | wc -l | tr -d ' '); if awk -v l="$l1" 'BEGIN{exit !(l <= 2.0)}' && [ "$busy" = "0" ]; then echo "$(ts) idle: load1=$l1"; return 0; fi; [ "$waited" -ge 1800 ] && { echo "$(ts) NOT IDLE after 1800s — stopping"; exit 1; }; [ $((waited % 60)) -eq 0 ] && echo "$(ts) waiting for idle: load1=$l1 other=$busy (${waited}s)"; sleep 10; waited=$((waited + 10)); done; }
echo "$(ts) == Mac 7B MC2 cell; tree $(git rev-parse --short HEAD); $(sysctl -n vm.loadavg); $(sysctl -n vm.swapusage)"
gate
echo "$(ts) identity (int4)"
GOINFER_MC2_BACKEND=cpu GOINFER_MC2_MODEL=$M go test -count=1 -timeout 60m -run '^TestMC2_decodeMultiStepBitIdentical$/.*/^int4$' -v ./decoder/ > $B/identity.log 2>&1; rc=$?
grep -E "^(--- |    --- |ok|FAIL)|bit-identical" $B/identity.log
# A skip or an empty match is not a pass: the int4 subtest must have run and printed its identity line.
if [ $rc -ne 0 ] || ! grep -q "bit-identical to the single-token forward" $B/identity.log; then echo "$(ts) identity did not pass (rc=$rc) — stopping"; exit 1; fi
for D in 128 512; do
  gate
  echo "$(ts) throughput depth $D; $(sysctl -n vm.loadavg)"
  GOINFER_MC2=1 GOINFER_MC2_BACKEND=cpu GOINFER_MC2_MODEL=$M GOINFER_MC2_DEPTH=$D go test -count=1 -timeout 90m -run '^TestMC2_batchedDecodeThroughput$' -v ./decoder/ > $B/throughput-d$D.log 2>&1
  echo "$(ts) depth $D exit $?"
  grep -E "^\[mc2|FAIL|panic" $B/throughput-d$D.log | tail -14
done
echo "$(ts) == DONE"
