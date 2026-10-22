#!/usr/bin/env bash
# A diagnostic for KV4's MC3 cell, not a gate: bench_w7_plain.py at 4 clients on the 1.5B, old/new alternating for PAIRS pairs, with the
# serve flags in ARGS and the new binary in NEWBIN.   mc3_probe.sh <out json> ; ARGS, NEWBIN, PAIRS from the environment
set -u
OUT=$1; PAIRS=${PAIRS:-3}; NEWBIN=${NEWBIN:-/home/francis/goinfer-bench/kv4-bins/serve-cuda-new}
cd /home/francis/goinfer-bench/kv4-2026-10-21
for i in $(seq $PAIRS); do
  for k in old new; do
    bin=/home/francis/goinfer-bench/kv4-bins/serve-cuda-old; [ $k = new ] && bin=$NEWBIN
    BENCH_W7_MODEL=$HOME/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf GOINFER_SERVE_CPU=$bin python3 -u scripts/bench_w7_plain.py "$OUT" \
      --clients ${CLIENTS:-4} --engines goinfer --backend cuda --key ${k}4_$i --fixed-nonce --serve-args "${ARGS:-}" 2>&1 | grep aggregate
  done
done
