#!/usr/bin/env bash
# The CUDA cells once more with one shared truncated preset on every arm (temperature 1.0, top_k 40, top_p 0.9), on the same main build as peer-sampled-2026-10-17.
# Phase A (greedy depth 128, every model, both engines) is planned unconditionally by the harness and runs again here, a second same-session control.
set -eo pipefail
cd /home/francis/wt/peer-trunc
export GOINFER_SERVE_CUDA=$HOME/goinfer-bench/serve-cuda-63038121
export OLLAMA_BIN=$HOME/ollama-0325/bin/ollama OLLAMA_MODELS=$HOME/ollama-0325/models
export BENCH_BACKENDS=cuda BENCH_ENGINES=goinfer,ollama BENCH_DEPTHS=none BENCH_MODELS=0.5B,gemma3-1b,phi3-mini
export BENCH_CONFIGS=temp1.0_topk40_topp0.9 BENCH_LOCK_WAIT=900
OUT=$HOME/goinfer-bench/peer-sampled-trunc-2026-10-17
python3 -u scripts/bench_peer.py $OUT/results.json 2>&1 | tee $OUT/run.log
echo "RUN-DONE rc=${PIPESTATUS[0]} $(date +%T)" | tee -a $OUT/run.log
