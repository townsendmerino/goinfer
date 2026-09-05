#!/usr/bin/env bash
# G20 (gpt-oss-20b, MXFP4) tier-2 cell (docs/task-peer-benchmarks.md §5). W1-style decode tok/s,
# depth 128, cuda, n=5, goinfer vs ollama vs llama-server. Same GGUF for all three engines
# (unlike M35/M26): goinfer needs -moe-cache-experts (added via MOE_MODELS membership,
# scripts/bench_peer.py), llama.cpp needs -ngl dropped for its own --fit (same MOE_MODELS set,
# 20B/~3.6B-active MoE exceeds the 8GB card). Resumable: re-run and it skips completed cells.
set -euo pipefail
cd /home/francis/mycode/goinfer
export PATH=$HOME/.local/bin:/usr/local/go/bin:$HOME/go/bin:$PATH
export OLLAMA_BIN=/home/francis/ollama-0325/bin/ollama
export OLLAMA_MODELS=/home/francis/ollama-0325/models
export GOINFER_SERVE_CUDA=/home/francis/bench-current/serve-cuda-abcdd1fe
export LLAMACPP_BIN=/home/francis/mycode/peers/llama.cpp/build/bin/llama-server
export BENCH_MODELS=G20
export BENCH_BACKENDS=cuda
export BENCH_ENGINES=goinfer,ollama,llamacpp
export BENCH_DEPTHS=none
export BENCH_RUNS=5
exec python3 scripts/bench_peer.py docs/measurements/peer-matrix-2026-09/nobara-g20_2026-09-04.json
