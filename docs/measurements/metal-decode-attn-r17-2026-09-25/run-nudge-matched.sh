#!/bin/bash
cd /Users/francistownsend-merino/tmcode/goinfer
echo "$(date +%T) start commit $(git rev-parse --short HEAD) (+ uncommitted metal/decode_attn_r17_test.go, metal/r2_gate_test.go)"
n=0
for k in 5 -5 6 -6 7 -7; do
  n=$((n+1)); echo "##### MATCHED NUDGE $n/6 k=$k $(date +%T) load1=$(sysctl -n vm.loadavg | awk '{print $2}')"
  GOINFER_HEAVY_TESTS=1 GOINFER_METAL_R17_CAND=exact-nudge GOINFER_METAL_R17_NUDGE=$k go test -tags goinfer_testhooks -count=1 -timeout 60m -v -run '^TestR17_decodeFidelityGate$' ./metal/ 2>&1
  echo "##### exit=$?"
done
echo "$(date +%T) DONE"
