#!/usr/bin/env bash
set -eo pipefail
cd /home/francis/goinfer-bench/rf2-confirm/wt-new
export PATH=/usr/local/go/bin:$HOME/go/bin:$PATH
python3 docs/measurements/kv4-cuda-2026-10-21/procmem_sampler.py /home/francis/goinfer-logs/rf2-confirm-2026-11-09/procmem-rerun.jsonl serve-cuda-old serve-cuda-new &
S=$!; trap 'kill $S 2>/dev/null || true' EXIT
export GOINFER_SERVE_CUDA=/home/francis/goinfer-bench/rf2-confirm/bins/serve-cuda-new GOINFER_SERVE_CUDA_OLD=/home/francis/goinfer-bench/rf2-confirm/bins/serve-cuda-old
export BENCH_ENGINES=goinfer,goinfer_old BENCH_BACKENDS=cuda BENCH_ABBA=1 BENCH_REQUIRE_BACKEND=1 BENCH_SERVE_LOG_DIR=/home/francis/goinfer-logs/rf2-confirm-2026-11-09/serve-logs-rerun
for _ in $(seq 90); do awk '{exit !($1 < 0.9)}' /proc/loadavg && break; sleep 10; done
echo "$(date +%T) rerun start load $(cut -d' ' -f1 /proc/loadavg)"
env BENCH_MODELS=7B BENCH_DEPTHS=128 python3 scripts/bench_peer.py /home/francis/goinfer-logs/rf2-confirm-2026-11-09/decode-qwen-rerun-7b128.json 2>&1 | tee /home/francis/goinfer-logs/rf2-confirm-2026-11-09/decode-qwen-rerun-7b128.log
echo RERUN-DONE
