#!/bin/bash
# R17 end-to-end (reported, not deciding): Metal greedy decode, goinfer new (7df881f5, attention_fa_blk) vs old
# (dde11d93, its parent: executor fix, legacy attention_fa) vs Ollama, interleaved, cell g's protocol.
set -u
cd /Users/francistownsend-merino/tmcode/goinfer || exit 1
B=$HOME/goinfer-bench/r17-e2e-2026-09-25
export OLLAMA_BIN=/opt/homebrew/bin/ollama OLLAMA_MODELS=$HOME/.ollama/models
export GOINFER_SERVE_METAL=$B/serve-metal-7df881f5 GOINFER_SERVE_METAL_OLD=$B/serve-metal-dde11d93
export BENCH_RUNS=3 BENCH_MAX_LOADAVG=2.0 BENCH_IDLE_WAIT=1800
ts() { date '+%H:%M:%S'; }
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
echo "$(ts) == R17 e2e (Metal decode 1.5B/7B @ 128/2048/3900; goinfer, goinfer_old, ollama) start; commit $(git rev-parse --short HEAD)"
wait_idle || exit 1
BENCH_BACKENDS=metal BENCH_DEPTH_BACKEND=metal BENCH_DEPTHS=2048,3900 BENCH_MODELS=1.5B,7B \
  BENCH_ENGINES=goinfer,goinfer_old,ollama python3 -u scripts/bench_peer.py $B/r17-e2e-metal-decode.json
echo "$(ts) == exit=$?"
echo "$(ts) == DONE"
