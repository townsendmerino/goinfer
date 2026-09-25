#!/bin/bash
w=0; while :; do l=$(sysctl -n vm.loadavg | awk '{print $2}'); awk -v l="$l" 'BEGIN{exit !(l<=2.0)}' && { echo "$(date +%T) idle load1=$l"; break; }; [ $w -ge 1800 ] && { echo "NOT IDLE after 1800s"; exit 1; }; [ $((w%60)) -eq 0 ] && echo "$(date +%T) waiting load1=$l"; sleep 10; w=$((w+10)); done
echo "$(date +%T) start; $(llama-bench --version 2>&1 | head -1); $(memory_pressure | tail -1)"
llama-bench -m $HOME/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf -p 512,3904 -n 0 -r 5 -ngl 99 -o json 2>&1
echo "$(date +%T) exit=$?"
