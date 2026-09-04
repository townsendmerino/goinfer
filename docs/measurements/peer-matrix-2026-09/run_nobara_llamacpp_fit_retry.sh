#!/usr/bin/env bash
# Retry llama-server on M35/M26 without a forced -ngl, so its own --fit (on by default) can
# place layers automatically instead of the earlier run's explicit -ngl 99 defeating it.
set -euo pipefail
cd /home/francis/mycode/goinfer
export PATH=$HOME/.local/bin:/usr/local/go/bin:$HOME/go/bin:$PATH
export OLLAMA_BIN=/home/francis/ollama-0325/bin/ollama
export OLLAMA_MODELS=/home/francis/ollama-0325/models
export GOINFER_SERVE_CUDA=/home/francis/bench-current/serve-cuda-abcdd1fe
export LLAMACPP_BIN=/home/francis/mycode/peers/llama.cpp/build/bin/llama-server
export BENCH_MODELS=M35,M26
export BENCH_BACKENDS=cuda
export BENCH_ENGINES=llamacpp
export BENCH_DEPTHS=none
export BENCH_RUNS=5
exec python3 scripts/bench_peer.py docs/measurements/peer-matrix-2026-09/nobara-w1-llamacpp-fit-retry_2026-09-04.json
