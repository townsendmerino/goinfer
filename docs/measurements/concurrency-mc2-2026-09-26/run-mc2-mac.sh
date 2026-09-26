#!/bin/bash
set -u
cd /Users/francistownsend-merino/tmcode/goinfer || exit 1
export GOWORK=/Users/francistownsend-merino/tmcode/goinfer/go.work
L=$HOME/goinfer-bench/mc2-2026-09-26
ts() { date '+%H:%M:%S'; }
waited=0
while :; do
  l1=$(sysctl -n vm.loadavg | awk '{print $2}')
  if awk -v l="$l1" 'BEGIN{exit !(l <= 2.0)}'; then echo "$(ts) idle: load1=$l1"; break; fi
  [ "$waited" -ge 1800 ] && { echo "$(ts) NOT IDLE after 1800s — stopping"; exit 1; }
  [ $((waited % 60)) -eq 0 ] && echo "$(ts) waiting for idle: load1=$l1 (${waited}s)"
  sleep 10; waited=$((waited + 10))
done
echo "$(ts) == MC2 Mac CPU start; commit $(git rev-parse --short HEAD) + uncommitted decoder/batchdecode_mc2_test.go"
for M in 0.5b 1.5b; do for D in 128 512; do
  echo "== $(ts) $M depth $D int4"
  GOINFER_MC2=1 GOINFER_MC2_MODEL=$HOME/models/qwen2.5-coder-$M-instruct-q4_k_m.gguf GOINFER_MC2_DEPTH=$D \
    /usr/local/go/bin/go test -count=1 -timeout 30m -run '^TestMC2_batchedDecodeThroughput$' -v ./decoder/ 2>&1 | grep -E "^\[mc2|PASS|FAIL|panic"
done; done
echo "$(ts) == DONE"
