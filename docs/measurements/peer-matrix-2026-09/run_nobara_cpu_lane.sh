#!/usr/bin/env bash
# CPU lane for S (Qwen2.5-Coder-1.5B), docs/task-peer-benchmarks.md tier-2 (S2): "the pure-Go
# story". W1-style decode tok/s, depth 128, CPU backend, n=5, goinfer vs ollama (num_gpu=0) vs
# llama.cpp (-ngl 0) -- run_cell's existing backend=="cpu" switch already forces all three onto
# CPU (V-08), no harness changes needed. goinfer's CPU serve binary is freshly built at HEAD
# (770e98e8) via `go build ./cmd/serve` (no build tag -- backendtag_guard_*.go only fires under
# -tags cuda/gpu/metal; CPU is cmd/serve's default). go-llama/goccy (the Go-lane peer named in
# the task doc's peer table) is NOT installed on this box -- skipped, noted in the report, not
# attempted here (a bigger, riskier lift than this pass's scope). Resumable.
set -euo pipefail
cd /home/francis/mycode/goinfer
export PATH=$HOME/.local/bin:/usr/local/go/bin:$HOME/go/bin:$PATH
export OLLAMA_BIN=/home/francis/ollama-0325/bin/ollama
export OLLAMA_MODELS=/home/francis/ollama-0325/models
export GOINFER_SERVE_CPU=/home/francis/bench-current/serve-cpu-770e98e8
export LLAMACPP_BIN=/home/francis/mycode/peers/llama.cpp/build/bin/llama-server
export BENCH_MODELS=1.5B
export BENCH_BACKENDS=cpu
export BENCH_ENGINES=goinfer,ollama,llamacpp
export BENCH_DEPTHS=none
export BENCH_RUNS=5
exec python3 scripts/bench_peer.py docs/measurements/peer-matrix-2026-09/nobara-s-cpu-lane_2026-09-04.json
