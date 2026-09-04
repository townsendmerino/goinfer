#!/usr/bin/env bash
# W3-style depth-8k decode (task doc §3, W3). D7 only for this pass (BENCH_RUNS left at the
# harness's own deep-cell default of 2, NOT the usual n=5 -- an 8k prefill measured ~5-6 min for
# a single completion on this box, so n=5 x 2 completions would triple the cost for a workload
# that isn't the primary comparison; the deep-cell default (ncomp=2, nruns=2, ngen=128) is what
# scripts/bench_peer.py's own gen_params() already reduces to specifically for this reason.
# Reuses the ORIGINAL results file so the already-done Phase A (depth-128) cells for 7B are
# skipped and only the new depth-8000 Phase B cells run.
set -euo pipefail
cd /home/francis/mycode/goinfer
export PATH=$HOME/.local/bin:/usr/local/go/bin:$HOME/go/bin:$PATH
export OLLAMA_BIN=/home/francis/ollama-0325/bin/ollama
export OLLAMA_MODELS=/home/francis/ollama-0325/models
export GOINFER_SERVE_CUDA=/home/francis/bench-current/serve-cuda-abcdd1fe
export LLAMACPP_BIN=/home/francis/mycode/peers/llama.cpp/build/bin/llama-server
export BENCH_MODELS=7B
export BENCH_BACKENDS=cuda
export BENCH_ENGINES=goinfer,ollama,llamacpp
export BENCH_DEPTHS=8000
export BENCH_DEEP_CTX=8192
exec python3 scripts/bench_peer.py docs/measurements/peer-matrix-2026-09/nobara-w1-d7-m35-m26_2026-09-04.json
