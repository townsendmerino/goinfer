#!/bin/bash
cd /Users/francistownsend-merino/tmcode/goinfer
wait_idle() { while :; do l=$(sysctl -n vm.loadavg | awk '{print $2}'); if awk -v l=$l 'BEGIN{exit !(l<2.0)}'; then echo "$(date +%T) idle: load1=$l"; return; fi; echo "$(date +%T) waiting for idle, load1=$l"; sleep 30; done; }
echo "$(date +%T) start commit $(git rev-parse --short HEAD) (+ uncommitted metal/decode_attn_r17_test.go, metal/r2_gate_test.go)"
n=0
for spec in exact-null:0 attention_fa:8 attention_fa:20 attention_fa:28 attention_fa:32 proto:8 proto:32 proto:0; do
  n=$((n+1)); c=${spec%%:*}; s=${spec##*:}
  wait_idle; echo "##### NULL $n/8 cand=$c S=$s $(date +%T)"
  GOINFER_HEAVY_TESTS=1 GOINFER_METAL_R17_CAND=$c GOINFER_METAL_R17_SPLIT=$s go test -tags goinfer_testhooks -count=1 -timeout 60m -v -run '^TestR17_decodeFidelityGate$' ./metal/ 2>&1
  echo "##### exit=$?"
done
echo "$(date +%T) DONE"
