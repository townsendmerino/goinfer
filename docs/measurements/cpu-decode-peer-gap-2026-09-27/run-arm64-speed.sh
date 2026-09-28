#!/bin/bash
# L1 build gate 6, arm64 CPU decode speed on the MacBook (M1 Pro, 16 GB), written 2026-09-28 before any timing.
# goinfer = merge commit 5c85f7c0 (PR #5, f16 int4 scales, aikit v1.50.0), goinfer_old = its first parent 3cd62e6d
# (aikit v1.49.0, f32 scales), interleaved against Ollama 0.32.5 (CPU-forced, num_gpu=0). CPU only, 0.5B / 1.5B / 7B,
# depth 128, 3 runs per cell, essay-v2 prompts: the same harness settings as nobara's gate 5 (run-gate5.sh).
#
# REPORTED, not a ship band (the task: "the owner decides the default"). Read as new ÷ old per model. Per the task, if
# arm64 comes out slower (new ÷ old < 1.00 on a model beyond the cells' own spread) the follow-up is a NEON widen
# (FCVTL) of row4's per-quad scale widening, which is in Go on arm64 (only amd64 has the fused asm kernel).
#
# Disclosed deviation: BENCH_MAX_LOADAVG=2.5, not the script's 1.0 or the 2.0 used by peer-claim-2026-09-25. This
# box's ambient 1-min load with nothing heavy running measured 2.16-2.58 just before launch (VSCode, this Claude
# Code session, the desktop). BENCH_IDLE_WAIT=1800 so the per-cell gate waits out a post-cell load spike instead of
# refusing mid-sweep. Both goinfer builds and Ollama see the same gate.
set -u
cd /Users/francistownsend-merino/tmcode/goinfer
L=$HOME/goinfer-bench/l1-mac
export GOINFER_SERVE_CPU=$L/serve-cpu-5c85f7c0 GOINFER_SERVE_CPU_OLD=$L/serve-cpu-3cd62e6d
export OLLAMA_BIN=/opt/homebrew/bin/ollama OLLAMA_MODELS=$HOME/.ollama/models
export BENCH_RUNS=3 BENCH_ENGINES=goinfer,goinfer_old,ollama BENCH_BACKENDS=cpu BENCH_DEPTHS=none BENCH_MODELS=0.5B,1.5B,7B
export BENCH_MAX_LOADAVG=2.5 BENCH_IDLE_WAIT=1800
for i in $(seq 1 90); do l=$(sysctl -n vm.loadavg | awk '{print $2}'); awk -v l="$l" 'BEGIN{exit !(l<=2.5)}' && break; echo "=== waiting for idle: loadavg $l"; sleep 20; done
echo "=== $(date '+%F %T %Z') / $(date -u '+%T UTC') START L1 gate 6 arm64 served"
sysctl vm.swapusage
python3 scripts/bench_peer.py $L/arm64-speed-served.json; rc=$?
sysctl vm.swapusage
echo "=== $(date '+%F %T %Z') / $(date -u '+%T UTC') END rc=$rc"
