#!/bin/bash
cd /Users/francistownsend-merino/tmcode/goinfer
wait_idle() { while :; do l=$(sysctl -n vm.loadavg | awk '{print $2}'); if awk -v l=$l 'BEGIN{exit !(l<2.0)}'; then echo "$(date +%T) idle: load1=$l"; return; fi; echo "$(date +%T) waiting for idle, load1=$l"; sleep 30; done; }
echo "$(date +%T) start commit $(git rev-parse --short HEAD) (+ uncommitted metal/decode_attn_r17_test.go)"
run() { wait_idle; echo "##### MODEL $1 splits $2"; GOINFER_METAL_R17=1 GOINFER_METAL_R17_MODEL=$HOME/models/$1 GOINFER_METAL_R17_DEPTHS=2048,3900 GOINFER_METAL_R17_SPLITS=$2 \
    go test -tags goinfer_testhooks -count=1 -timeout 30m -v -run '^TestR17AttentionProto$' ./metal/ 2>&1; echo "##### MODEL $1 exit=$?"; }
run qwen2.5-coder-1.5b-instruct-q4_k_m.gguf 0,8,16,24,32,48
run qwen2.5-7b-instruct-q4_k_m.int4.metal.giw 0,16,32
echo "$(date +%T) DONE"
