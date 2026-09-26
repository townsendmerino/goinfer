#!/usr/bin/env bash
# R6 phase 3 (f16 resident KV) BASELINE, before any kernel work: phi3-mini on CUDA, same weights (p3m-local).
# Run A = today's default (int8int8 + per-32) at the deepest standard depth whose f32 KV fits (2048).
# Run B = the band's original configuration (int4, per-row) at 2048 and 3900: the pre-registered cell.
set -u
cd /home/francis/mycode/goinfer
R=/home/francis/goinfer-logs/f16kv-baseline-2026-09-26
export GOINFER_SERVE_CUDA=/home/francis/bench-peer-claim/serve-cuda-cad1973b GOINFER_SERVE_CPU=/home/francis/bench-peer-claim/serve-cpu-cad1973b GOINFER_SERVE=/home/francis/bench-peer-claim/serve-cuda-cad1973b
export OLLAMA_BIN=$HOME/ollama-0325/bin/ollama OLLAMA_MODELS=$HOME/ollama-0325/models
unset LD_LIBRARY_PATH
export LLAMACPP_BIN=$HOME/mycode/peers/llama.cpp/build/bin/llama-server
export BENCH_RUNS=3 BENCH_ENGINES=goinfer,ollama,llamacpp BENCH_MODELS=phi3-mini BENCH_BACKENDS=cuda
waitidle() { for i in $(seq 1 90); do l=$(cut -d' ' -f1 /proc/loadavg); awk -v l="$l" 'BEGIN{exit !(l<0.8)}' && return; echo "=== waiting for idle: loadavg $l"; sleep 20; done; }
step() { waitidle; echo "=== $(date -u +%FT%TZ) START $1"; t0=$(date +%s); }
done_() { echo "=== $(date -u +%FT%TZ) END $1 rc=$2 ($(( $(date +%s) - t0 ))s)"; }
step "A int8int8+per-32 @2048, BENCH_CTX=3072"
BENCH_CTX=3072 BENCH_QUANT_OVERRIDE=M35=int4mix,phi3-mini=int8int8 BENCH_DEPTHS=2048 \
  python3 scripts/bench_peer.py $R/a-int8int8-g32.json; done_ A $?
step "B int4 per-row @2048/3900 (the band's original configuration)"
BENCH_QUANT_OVERRIDE=M35=int4mix,phi3-mini=int4 BENCH_DEPTHS=2048,3900 \
  python3 scripts/bench_peer.py $R/b-int4.json; done_ B $?
echo "=== $(date -u +%FT%TZ) ALL DONE"
