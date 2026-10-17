#!/usr/bin/env bash
# The eight CUDA sampled peer cells of docs/benchmarks.md ("Sampled configurations, depth 128, CUDA") plus the greedy depth-128 phase A the harness always plans,
# on main 63038121, through bench_peer.py's current sampled configs. Phase A and the ninth sampled cell (0.5B temp0.8_topk40) are controls, not part of the eight.
set -eo pipefail
cd /home/francis/wt/serve-main
export GOINFER_SERVE_CUDA=$HOME/goinfer-bench/serve-cuda-63038121
export OLLAMA_BIN=$HOME/ollama-0325/bin/ollama OLLAMA_MODELS=$HOME/ollama-0325/models
export BENCH_BACKENDS=cuda BENCH_ENGINES=goinfer,ollama BENCH_DEPTHS=none BENCH_MODELS=0.5B,gemma3-1b,phi3-mini
export BENCH_CONFIGS=temp1.0_notrunc,temp0.8_topp0.95,temp0.8_topk40 BENCH_LOCK_WAIT=900
OUT=$HOME/goinfer-bench/peer-sampled-2026-10-17
python3 -u scripts/bench_peer.py $OUT/results.json 2>&1 | tee $OUT/run.log
echo "RUN-DONE rc=${PIPESTATUS[0]} $(date +%T)" | tee -a $OUT/run.log
