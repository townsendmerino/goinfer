#!/usr/bin/env bash
# Peer comparison deciding a CUDA q4k default for Q4_K GGUFs (pre-registered d5077650).
set -u
cd /home/francis/mycode/goinfer
R=/home/francis/goinfer-logs/q4k-peer-2026-09-26
export GOINFER_SERVE_CUDA=/home/francis/bench-peer-claim/serve-cuda-d5077650 GOINFER_SERVE_CPU=/home/francis/bench-peer-claim/serve-cpu-d5077650 GOINFER_SERVE=/home/francis/bench-peer-claim/serve-cuda-d5077650
export OLLAMA_BIN=$HOME/ollama-0325/bin/ollama OLLAMA_MODELS=$HOME/ollama-0325/models
unset LD_LIBRARY_PATH
export LLAMACPP_BIN=$HOME/mycode/peers/llama.cpp/build/bin/llama-server
export BENCH_RUNS=3 BENCH_ENGINES=goinfer,ollama,llamacpp BENCH_MODELS=1.5B,7B BENCH_BACKENDS=cuda BENCH_DEPTHS=2048,3900
echo "=== $(date -u +%FT%TZ) START S1 q4k"
BENCH_QUANT_OVERRIDE=M35=int4mix,phi3-mini=int8int8,1.5B=q4k,7B=q4k python3 scripts/bench_peer.py $R/s1-q4k.json; echo "=== $(date -u +%FT%TZ) END S1 rc=$?"
echo "=== $(date -u +%FT%TZ) START S2 int4"
python3 scripts/bench_peer.py $R/s2-int4.json; echo "=== $(date -u +%FT%TZ) END S2 rc=$?"
echo "=== $(date -u +%FT%TZ) ALL DONE"
