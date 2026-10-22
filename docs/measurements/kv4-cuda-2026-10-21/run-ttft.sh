#!/usr/bin/env bash
# KV4's G2 time-to-first-token cells: ttft_ab.py on the 1.5B and the 7B, old/new pinned binaries, under the timing lock (exclusive).
#   run-ttft.sh <out dir>
set -euo pipefail
OUT=${1:?out dir}; mkdir -p "$OUT"
D=/home/francis/goinfer-bench/kv4-2026-10-21/docs/measurements/kv4-cuda-2026-10-21
BINS=/home/francis/goinfer-bench/kv4-bins
for mk in 1.5B 7B; do
  TTFT_MODEL=$mk python3 /home/francis/goinfer-bench/kv4-2026-10-21/scripts/timing_lock.py run --wait 1800 --label kv4-ttft-$mk -- \
    python3 $D/ttft_ab.py $BINS/serve-cuda-old $BINS/serve-cuda-new "$OUT/ttft-$mk.json" 2>&1 | tee "$OUT/ttft-$mk.log"
done
echo RUN-TTFT-DONE
