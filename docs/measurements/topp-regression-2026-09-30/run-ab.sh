#!/bin/bash
# Top-p sampling regression on CUDA (found in docs/measurements/peer-sweep-2026-09-29.md, finding 4): goinfer-vs-goinfer,
# one cell (0.5B, depth 128, t=0.8 top-p 0.95, plus greedy as the control), old 411e7fc4 against new 754f12d3.
# A: both binaries at their own defaults. B: both with -embed-int4=false (isolates the 2026-09-28 default flip).
set -u
W=$HOME/goinfer-bench/topp-regression-2026-09-30; mkdir -p "$W"
cd "$HOME/goinfer-bench/peer-sweep-2026-09-29/wt-754f12d3" || exit 1
export GOINFER_SERVE_CUDA=$HOME/goinfer-bench/peer-sweep-2026-09-29/serve-cuda-754f12d3
export GOINFER_SERVE_CUDA_OLD=$HOME/bench-peer-claim/serve-cuda-411e7fc4
export BENCH_RUNS=3 BENCH_ENGINES=goinfer,goinfer_old BENCH_BACKENDS=cuda BENCH_DEPTHS=none BENCH_MODELS=0.5B
export BENCH_CONFIGS=greedy,temp0.8_topp0.95 BENCH_SERVE_LOG_DIR=$W/serve-logs
echo "$(date '+%H:%M:%S') A (defaults)"; python3 -u scripts/bench_peer.py "$W/a-defaults.json"; echo "A exit=$?"
echo "$(date '+%H:%M:%S') B (-embed-int4=false on both)"; BENCH_GOINFER_ARGS="-embed-int4=false" python3 -u scripts/bench_peer.py "$W/b-embedint8.json"; echo "B exit=$?"
echo "$(date '+%H:%M:%S') DONE"
