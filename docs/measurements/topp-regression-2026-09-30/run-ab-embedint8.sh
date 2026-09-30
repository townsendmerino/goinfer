#!/bin/bash
set -u
W=$HOME/goinfer-bench/topp-regression-2026-09-30
cd "$HOME/goinfer-bench/peer-sweep-2026-09-29/wt-754f12d3" || exit 1
for i in $(seq 1 60); do awk '{exit !($1<0.8)}' /proc/loadavg && break; sleep 10; done
export GOINFER_SERVE_CUDA=$HOME/goinfer-bench/peer-sweep-2026-09-29/serve-cuda-754f12d3 GOINFER_SERVE_CUDA_OLD=$HOME/bench-peer-claim/serve-cuda-411e7fc4
export BENCH_RUNS=3 BENCH_ENGINES=goinfer,goinfer_old BENCH_BACKENDS=cuda BENCH_DEPTHS=none BENCH_MODELS=0.5B
export BENCH_CONFIGS=temp0.8_topp0.95 BENCH_SERVE_LOG_DIR=$W/serve-logs-b BENCH_GOINFER_ARGS="-embed-int4=false"
echo "$(date '+%H:%M:%S') B (-embed-int4=false on both)"; python3 -u scripts/bench_peer.py "$W/b-embedint8.json"; echo "B exit=$?"; echo DONE
