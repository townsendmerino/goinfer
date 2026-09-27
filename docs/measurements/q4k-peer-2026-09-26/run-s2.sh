#!/usr/bin/env bash
# S2 re-run: the first attempt was REFUSED at startup (loadavg 1.03 from S1's own servers; bench_peer exits 0 on refusal).
set -u
cd /home/francis/mycode/goinfer
R=/home/francis/goinfer-logs/q4k-peer-2026-09-26
B=$HOME/bench-peer-claim
export GOINFER_SERVE_CUDA=$B/serve-cuda-d5077650 GOINFER_SERVE_CPU=$B/serve-cpu-d5077650 GOINFER_SERVE=$B/serve-cuda-d5077650
export OLLAMA_BIN=$HOME/ollama-0325/bin/ollama OLLAMA_MODELS=$HOME/ollama-0325/models
unset LD_LIBRARY_PATH
export LLAMACPP_BIN=$HOME/mycode/peers/llama.cpp/build/bin/llama-server
export BENCH_RUNS=3 BENCH_ENGINES=goinfer,ollama,llamacpp BENCH_MODELS=1.5B,7B BENCH_BACKENDS=cuda BENCH_DEPTHS=2048,3900
for i in $(seq 1 90); do l=$(cut -d' ' -f1 /proc/loadavg); awk -v l="$l" 'BEGIN{exit !(l<0.8)}' && break; echo "=== waiting for idle: loadavg $l"; sleep 20; done
echo "=== $(date -u +%FT%TZ) START S2 int4"
python3 scripts/bench_peer.py $R/s2-int4.json; echo "=== $(date -u +%FT%TZ) END S2 rc=$?"
test -s $R/s2-int4.json && echo "=== results file present" || echo "=== NO RESULTS FILE"
echo "=== $(date -u +%FT%TZ) ALL DONE"
