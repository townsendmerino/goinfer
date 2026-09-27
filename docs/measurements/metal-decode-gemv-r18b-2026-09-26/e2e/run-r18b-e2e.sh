#!/bin/bash
# R18b end-to-end (reported, not deciding): Metal greedy decode, goinfer new (R18b masked half-staged GEMV form wired) vs old
# (its parent 9bafd1f3, R18's integer rows kernels) vs Ollama, interleaved, R17's e2e protocol (run-r17-e2e.sh).
set -u
cd /Users/francistownsend-merino/tmcode/goinfer || exit 1
B=$HOME/goinfer-bench/r18b-e2e-2026-09-26
export OLLAMA_BIN=/opt/homebrew/bin/ollama OLLAMA_MODELS=$HOME/.ollama/models
export GOINFER_SERVE_METAL=$B/serve-metal-NEW GOINFER_SERVE_METAL_OLD=$B/serve-metal-OLD
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
echo "$(ts) == R18b e2e (Metal decode 1.5B/7B @ 128/2048/3900; goinfer, goinfer_old, ollama) start; commit $(git rev-parse --short HEAD); new=$(readlink $GOINFER_SERVE_METAL) old=$(readlink $GOINFER_SERVE_METAL_OLD)"
wait_idle || exit 1
BENCH_BACKENDS=metal BENCH_DEPTH_BACKEND=metal BENCH_DEPTHS=2048,3900 BENCH_MODELS=1.5B,7B \
  BENCH_ENGINES=goinfer,goinfer_old,ollama python3 -u scripts/bench_peer.py $B/r18b-e2e-metal-decode.json
echo "$(ts) == exit=$?"
echo "$(ts) == DONE"
