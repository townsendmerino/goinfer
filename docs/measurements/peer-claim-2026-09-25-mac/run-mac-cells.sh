#!/bin/bash
# Mac cells g–i of docs/measurements/peer-claim-2026-09-25.md (pre-registered at 411e7fc4).
# Binaries built at 9c592095 (no Go diff from 411e7fc4). Logs and results live here, not /tmp.
set -u
cd /Users/francistownsend-merino/tmcode/goinfer || exit 1
B=$HOME/goinfer-bench/peer-claim-2026-09-25
export OLLAMA_BIN=/opt/homebrew/bin/ollama OLLAMA_MODELS=$HOME/.ollama/models
export LLAMACPP_BIN=/opt/homebrew/bin/llama-server
export GOINFER_SERVE_METAL=$B/serve-metal-9c592095 GOINFER_SERVE_CPU=$B/serve-cpu-9c592095
export BENCH_RUNS=3 BENCH_MAX_LOADAVG=2.0 BENCH_IDLE_WAIT=1800

ts() { date '+%H:%M:%S'; }
# bench_peer.py's preflight REFUSES a busy box instead of waiting, and bench_peer_prefill.py reads
# /proc/loadavg, which macOS lacks. So every harness call is preceded by this wait.
wait_idle() {
  local waited=0
  while :; do
    l1=$(sysctl -n vm.loadavg | awk '{print $2}')
    if awk -v l="$l1" 'BEGIN{exit !(l <= 2.0)}'; then echo "$(ts) idle: load1=$l1"; return 0; fi
    [ "$waited" -ge 1800 ] && { echo "$(ts) NOT IDLE after 1800s (load1=$l1) — stopping"; return 1; }
    [ $((waited % 60)) -eq 0 ] && echo "$(ts) waiting for idle: load1=$l1 (${waited}s)"
    sleep 10; waited=$((waited + 10))
  done
}

echo "$(ts) == cell g (Metal decode 0.5B/1.5B/7B @ 128/2048/3900) start"
wait_idle || exit 1
BENCH_BACKENDS=metal BENCH_DEPTH_BACKEND=metal BENCH_DEPTHS=2048,3900 BENCH_MODELS=0.5B,1.5B,7B \
  BENCH_ENGINES=goinfer,ollama,llamacpp python3 -u scripts/bench_peer.py $B/g-metal-decode.json
echo "$(ts) == cell g exit=$?"

echo "$(ts) == cell i (CPU decode 0.5B/1.5B @ 128) start"
wait_idle || exit 1
BENCH_BACKENDS=cpu BENCH_DEPTHS=none BENCH_MODELS=0.5B,1.5B \
  BENCH_ENGINES=goinfer,ollama,llamacpp python3 -u scripts/bench_peer.py $B/i-cpu-decode.json
echo "$(ts) == cell i exit=$?"

echo "$(ts) == cell h (Metal prefill TTFT 1.5B @ K=512/3900, 6 prompts) start"
if wait_idle; then
  python3 -u scripts/bench_peer_prefill.py $B/h-metal-prefill.json --models 1.5B --depths 512,3900 --n 6 --backend metal
  echo "$(ts) == cell h exit=$?"
fi
echo "$(ts) == DONE"
