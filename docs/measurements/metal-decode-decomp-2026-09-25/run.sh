#!/bin/bash
cd /Users/francistownsend-merino/tmcode/goinfer/metal || exit 1
wait_idle() { local w=0; while :; do l=$(sysctl -n vm.loadavg | awk '{print $2}'); awk -v l="$l" 'BEGIN{exit !(l<=2.0)}' && { echo "$(date +%T) idle load1=$l"; return 0; }; [ $w -ge 1800 ] && { echo "NOT IDLE"; return 1; }; [ $((w%60)) -eq 0 ] && echo "$(date +%T) waiting load1=$l"; sleep 10; w=$((w+10)); done; }
echo "$(date +%T) start commit $(git rev-parse --short=8 HEAD) dirty(tracked)=$(git status --porcelain | grep -v '^??' | awk '{print $2}' | tr '\n' ' ')"
for mp in "$HOME/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf" "$HOME/models/qwen2.5-7b-instruct-q4_k_m.int4.metal.giw" "$HOME/models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf"; do
  wait_idle || exit 1
  echo "$(date +%T) ##### $(basename $mp); $(memory_pressure | tail -1)"
  GOINFER_METAL_DDECOMP=1 GOINFER_METAL_DDECOMP_MODEL="$mp" go test -tags goinfer_testhooks -run '^TestMetalDecodeDecomp$' -v -count=1 -timeout 40m . 2>&1
  echo "$(date +%T) ##### $(basename $mp) exit=$?"
done
echo "$(date +%T) DONE"
