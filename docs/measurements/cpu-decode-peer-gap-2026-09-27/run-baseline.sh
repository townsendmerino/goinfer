#!/usr/bin/env bash
# CPU decode vs Ollama, Phase 0 step 1: re-baseline (exploratory, not a graded claim).
# goinfer @157196bb (HEAD) and goinfer_old @411e7fc4 (the 2026-09-25 peer sweep's build) interleaved
# against Ollama v0.32.5, CPU only, 0.5B / 1.5B / 7B at depth 128, 3 runs each, essay-v2 prompts.
set -u
cd /home/francis/mycode/goinfer
D=/home/francis/goinfer-bench/cpu-decode-2026-09-27
export GOINFER_SERVE_CPU=$D/serve-cpu-157196bb GOINFER_SERVE_CPU_OLD=/home/francis/bench-peer-claim/serve-cpu-411e7fc4
export OLLAMA_BIN=$HOME/ollama-0325/bin/ollama OLLAMA_MODELS=$HOME/ollama-0325/models
unset LD_LIBRARY_PATH
export BENCH_RUNS=3 BENCH_ENGINES=goinfer,goinfer_old,ollama BENCH_BACKENDS=cpu BENCH_DEPTHS=none BENCH_MODELS=0.5B,1.5B,7B
for i in $(seq 1 90); do l=$(cut -d' ' -f1 /proc/loadavg); awk -v l="$l" 'BEGIN{exit !(l<0.8)}' && break; echo "=== waiting for idle: loadavg $l"; sleep 20; done
echo "=== $(date '+%F %T %Z') START baseline"
python3 scripts/bench_peer.py $D/baseline.json; rc=$?
echo "=== $(date '+%F %T %Z') END rc=$rc"
