#!/bin/bash
cd /Users/francistownsend-merino/tmcode/goinfer
echo "$(date +%T) start commit $(git rev-parse --short HEAD) (+ uncommitted metal/decode_attn_r17_test.go, metal/r2_gate_test.go)"
n=0
for spec in vchunk:32 vchunk:64 vchunk:128 vchunk:256 vrev8:0; do
  n=$((n+1)); c=exact-${spec%%:*}; C=${spec##*:}
  echo "##### VSUM $n/5 cand=$c C=$C $(date +%T) load1=$(sysctl -n vm.loadavg | awk '{print $2}')"
  GOINFER_HEAVY_TESTS=1 GOINFER_METAL_R17_CAND=$c GOINFER_METAL_R17_VCHUNK=$C go test -tags goinfer_testhooks -count=1 -timeout 60m -v -run '^TestR17_decodeFidelityGate$' ./metal/ 2>&1
  echo "##### exit=$?"
done
echo "$(date +%T) DONE"
