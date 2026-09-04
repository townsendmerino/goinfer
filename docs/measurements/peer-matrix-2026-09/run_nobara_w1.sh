#!/usr/bin/env bash
# W1 (decode tok/s, depth 128) for D7/M35/M26 on CUDA, goinfer vs ollama vs llama-server.
# docs/task-peer-benchmarks.md tier-1 minimum pass. Resumable: re-run and it skips completed cells.
set -euo pipefail
cd /home/francis/mycode/goinfer
export PATH=$HOME/.local/bin:/usr/local/go/bin:$HOME/go/bin:$PATH
export OLLAMA_BIN=/home/francis/ollama-0325/bin/ollama
export OLLAMA_MODELS=/home/francis/ollama-0325/models
export GOINFER_SERVE_CUDA=/home/francis/bench-current/serve-cuda-abcdd1fe
export LLAMACPP_BIN=/home/francis/mycode/peers/llama.cpp/build/bin/llama-server
export BENCH_MODELS=7B,M35,M26
export BENCH_BACKENDS=cuda
export BENCH_ENGINES=goinfer,ollama,llamacpp
export BENCH_DEPTHS=none
export BENCH_RUNS=5
exec python3 scripts/bench_peer.py docs/measurements/peer-matrix-2026-09/nobara-w1-d7-m35-m26_2026-09-04.json
