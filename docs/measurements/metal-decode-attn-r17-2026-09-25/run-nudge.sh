#!/bin/bash
cd /Users/francistownsend-merino/tmcode/goinfer
D=$HOME/goinfer-bench/metal-decode-r17-2026-09-25
until grep -q DONE $D/step2-null-gates.log; do sleep 20; done
echo "$(date +%T) start commit $(git rev-parse --short HEAD) (+ uncommitted metal/decode_attn_r17_test.go, metal/r2_gate_test.go)"
n=0
for k in 1 -1 2 -2 3 -3; do
  n=$((n+1)); echo "##### NUDGE $n/6 k=$k $(date +%T) load1=$(sysctl -n vm.loadavg | awk '{print $2}')"
  GOINFER_HEAVY_TESTS=1 GOINFER_METAL_R17_CAND=exact-nudge GOINFER_METAL_R17_NUDGE=$k go test -tags goinfer_testhooks -count=1 -timeout 60m -v -run '^TestR17_decodeFidelityGate$' ./metal/ 2>&1
  echo "##### exit=$?"
done
echo "##### ACCURACY 7B $(date +%T)"
GOINFER_METAL_R17=1 GOINFER_METAL_R17_MODEL=$HOME/models/qwen2.5-7b-instruct-q4_k_m.int4.metal.giw go test -tags goinfer_testhooks -count=1 -timeout 30m -v -run '^TestR17KernelAccuracy$' ./metal/ 2>&1
echo "##### exit=$?"
echo "$(date +%T) DONE"
