#!/bin/bash
cd /Users/francistownsend-merino/tmcode/goinfer
wait_idle() { while :; do l=$(sysctl -n vm.loadavg | awk '{print $2}'); if awk -v l=$l 'BEGIN{exit !(l<2.0)}'; then echo "$(date +%T) idle: load1=$l"; return; fi; echo "$(date +%T) waiting for idle, load1=$l"; sleep 30; done; }
echo "$(date +%T) start commit $(git rev-parse --short HEAD) (+ uncommitted metal/decode_attn_r17_test.go, metal/r2_gate_test.go)"
wait_idle; echo "##### GATE R17 candidate S=16"
GOINFER_HEAVY_TESTS=1 GOINFER_METAL_R17_SPLIT=16 go test -tags goinfer_testhooks -count=1 -timeout 60m -v -run '^TestR17_decodeFidelityGate$' ./metal/ 2>&1; echo "##### exit=$?"
wait_idle; echo "##### GATE R2 (attention_fa, via the refactored helper)"
GOINFER_HEAVY_TESTS=1 go test -tags goinfer_testhooks -count=1 -timeout 60m -v -run '^TestR2_decodeFidelityGate$' ./metal/ 2>&1; echo "##### exit=$?"
echo "$(date +%T) DONE"
