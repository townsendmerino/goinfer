#!/usr/bin/env bash
# KV4's G2 and G4 on one build: decode at depth (run-speed.sh), time to first token (run-ttft.sh), and the MC3 cells (run-mc3.sh, under
# the timing lock), in that order.   run-final.sh <out dir>
set -u
OUT=${1:?out dir}; mkdir -p "$OUT"
D=/home/francis/goinfer-bench/kv4-2026-10-21/docs/measurements/kv4-cuda-2026-10-21
bash $D/run-speed.sh "$OUT/speed" > "$OUT/speed.log" 2>&1
bash $D/run-ttft.sh "$OUT/ttft" > "$OUT/ttft.log" 2>&1
python3 /home/francis/goinfer-bench/kv4-2026-10-21/scripts/timing_lock.py run --wait 1800 --label kv4-mc3 -- bash $D/run-mc3.sh "$OUT/mc3" > "$OUT/mc3.log" 2>&1
echo RUN-FINAL-DONE
