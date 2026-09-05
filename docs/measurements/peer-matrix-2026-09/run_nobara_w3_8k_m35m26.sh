#!/usr/bin/env bash
# W3-style depth-8000 decode for M35/M26 (task doc S3, W3; the 770e98e8 follow-up explicitly
# flagged as owed). Same reduced-cost reasoning as run_nobara_w3_8k.sh's D7 cell (BENCH_RUNS left
# at the harness's own deep-cell default of ncomp=2, nruns=2, ngen=128) -- confirmed live
# 2026-09-05 with a single-completion smoke test before committing to this: goinfer/M35 at depth
# 8000 cost 264.8s for ONE completion (128 gen tokens over an 8012-token prompt), consistent with
# gpt-oss/gemma4moe's SEQUENTIAL prefill path (one forward per prompt token, not batched) -- so a
# full ncomp=2 x nruns=2 goinfer cell costs roughly 4x that per model, several times the D7 cell's
# cost. Both M35 (qwen35moe) and M26 (gemma4moe) report the sequential prefill path via /v1/models'
# equivalent server log line, not just M26 -- noted here since it affects wall-time interpretation.
# Reuses the ORIGINAL M35/M26 results file so already-done Phase A (depth-128) cells are skipped
# and only the new depth-8000 Phase B cells run.
set -euo pipefail
cd /home/francis/mycode/goinfer
export PATH=$HOME/.local/bin:/usr/local/go/bin:$HOME/go/bin:$PATH
export OLLAMA_BIN=/home/francis/ollama-0325/bin/ollama
export OLLAMA_MODELS=/home/francis/ollama-0325/models
export GOINFER_SERVE_CUDA=/home/francis/bench-current/serve-cuda-abcdd1fe
export LLAMACPP_BIN=/home/francis/mycode/peers/llama.cpp/build/bin/llama-server
export BENCH_MODELS=M35,M26
export BENCH_BACKENDS=cuda
export BENCH_ENGINES=goinfer,ollama,llamacpp
export BENCH_DEPTHS=8000
export BENCH_DEEP_CTX=8192
exec python3 scripts/bench_peer.py docs/measurements/peer-matrix-2026-09/nobara-w1-d7-m35-m26_2026-09-04.json
