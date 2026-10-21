#!/bin/bash
# One half of the pre-registered shared-preset CUDA cells (docs/measurements/peer-notrunc-2026-10-17.md, "The shared truncated preset"): temp0.8_topk40_topp0.9, depth 128,
# 3 runs, goinfer / Ollama / llama.cpp, at the like-for-like harness (e8e8c95d). $1 = 1: 0.5B + gemma3-1b; $1 = 2: phi3-mini (default quant q4k, against its Q4_K_M peers).
# Each half also carries the harness's unskippable greedy depth-128 phase A (same-session controls).
set -u
H=$1; W=$HOME/goinfer-bench/peer-sharedtrunc-2026-10-17; R=/home/francis/wt/peer-trunc; REV=e8e8c95d
ts() { date -u +%FT%TZ; }
[ "$(git -C $R rev-parse --short=8 HEAD)" = "$REV" ] || { echo "worktree is not at $REV"; exit 1; }
export GOINFER_SERVE_CUDA=$W/serve-cuda-$REV GOINFER_SERVE=$W/serve-cuda-$REV
export OLLAMA_BIN=$HOME/ollama-0325/bin/ollama OLLAMA_MODELS=$HOME/ollama-0325/models
unset LD_LIBRARY_PATH
export LLAMACPP_BIN=$HOME/mycode/peers/llama.cpp/build/bin/llama-server
export BENCH_RUNS=3 BENCH_ENGINES=goinfer,ollama,llamacpp BENCH_SERVE_LOG_DIR=$W/serve-logs-half$H BENCH_LOCK_WAIT=900
case $H in 1) MODELS=0.5B,gemma3-1b;; 2) MODELS=phi3-mini;; *) exit 2;; esac
cd $R || exit 1
for i in $(seq 1 90); do l=$(cut -d' ' -f1 /proc/loadavg); awk -v l="$l" 'BEGIN{exit !(l<0.8)}' && break; echo "=== waiting for idle: loadavg $l"; sleep 20; done
echo "=== $(ts) START half $H ($MODELS)"; t0=$(date +%s)
BENCH_MODELS=$MODELS BENCH_BACKENDS=cuda BENCH_DEPTHS=none BENCH_CONFIGS=temp0.8_topk40_topp0.9 python3 -u scripts/bench_peer.py "$W/half$H.json"
rc=$?
echo "=== $(ts) END half $H rc=$rc ($(( $(date +%s) - t0 ))s)"
mkdir -p "$W/g$H" && cp "$W/half$H.json" "$W/g$H/f-sampled.json" && python3 "$W/grade.py" "$W/g$H" | tee "$W/grade-half$H.jsonl"
echo "HALF-DONE $H rc=$rc $(date +%T)"
exit $rc
