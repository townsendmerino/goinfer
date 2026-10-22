#!/usr/bin/env bash
# KV4's G2 MC3 cells (kv4-cuda-2026-10-21.md §2): bench_w7_plain.py at 4 clients on the 1.5B, serve defaults, 6 turns x 128 greedy tokens per
# client, a fresh server per cell, cells old new new old old new, as MC3's own grading ran them (concurrency-mc3-cuda-2026-09-27.md).
#   run-mc3.sh <out dir>        (the caller holds the timing lock)
set -u
OUT=${1:?out dir}; mkdir -p "$OUT"
REPO=/home/francis/goinfer-bench/kv4-2026-10-21
BINS=/home/francis/goinfer-bench/kv4-bins
M15=$HOME/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf
cd "$REPO" || exit 1
idle() { for _ in $(seq 90); do awk '{exit !($1 < 2.0)}' /proc/loadavg && return 0; sleep 10; done; echo "box not idle after 15 min"; exit 1; }
for k in old4_1 new4_1 new4_2 old4_2 old4_3 new4_3; do
  case $k in old*) bin=$BINS/serve-cuda-old;; new*) bin=$BINS/serve-cuda-new;; esac
  idle
  echo "$(date +%T) cell $k load $(cut -d' ' -f1 /proc/loadavg)"
  BENCH_W7_MODEL=$M15 GOINFER_SERVE_CPU=$bin python3 -u scripts/bench_w7_plain.py "$OUT/w7.json" --clients 4 --engines goinfer \
    --backend cuda --key $k --fixed-nonce --server-log "$OUT/w7-servers.log" || { echo "cell $k FAILED"; exit 1; }
done
echo RUN-MC3-DONE
