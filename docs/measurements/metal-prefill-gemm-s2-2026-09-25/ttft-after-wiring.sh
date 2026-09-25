#!/bin/bash
cd /Users/francistownsend-merino/tmcode/goinfer || exit 1
w=0; while :; do l=$(sysctl -n vm.loadavg | awk '{print $2}'); awk -v l="$l" 'BEGIN{exit !(l<=2.0)}' && { echo "$(date +%T) idle load1=$l"; break; }; [ $w -ge 1800 ] && { echo "NOT IDLE"; exit 1; }; [ $((w%60)) -eq 0 ] && echo "$(date +%T) waiting load1=$l"; sleep 10; w=$((w+10)); done
echo "$(date +%T) start commit $(git rev-parse --short=8 HEAD); $(memory_pressure | tail -1)"
export OLLAMA_BIN=/opt/homebrew/bin/ollama OLLAMA_MODELS=$HOME/.ollama/models GOINFER_SERVE_METAL=/Users/francistownsend-merino/goinfer-bench/metal-prefill-ttft-2026-09-25/serve-metal-1fd9d95e
python3 -u scripts/bench_peer_prefill.py /Users/francistownsend-merino/goinfer-bench/metal-prefill-ttft-2026-09-25/h-metal-prefill-r16.json --models 1.5B --depths 512,3900 --n 6 --backend metal 2>&1
echo "$(date +%T) exit=$?"
