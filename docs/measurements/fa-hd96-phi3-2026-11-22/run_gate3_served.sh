#!/bin/bash
# Gate 3's served half for the hd-96 lane (README beside this file), by day, under the exclusive timing lock:
#   python3 scripts/timing_lock.py run --label hd96-gate3-served -- bash docs/measurements/fa-hd96-phi3-2026-11-22/run_gate3_served.sh
# One serve binary serves both arms. goinfer adds -flash-decode-hd96, and goinfer_old adds -flash-decode-hd96=false after it.
# Phi-3 mini at its default (q4k), depths 128 and 3,900, two runs a cell, one counterbalanced quad (BENCH_ABBA=1).
# OP1-o2's run-cell.sh with the model, the flag and the decision depth changed. Grades with grade_cell.py.
set -uo pipefail
BIN=${BIN:?set BIN to the serve binary}
OUT=${OUT:-$HOME/goinfer-bench/fa-hd96-run/served-$(date +%H%M)}
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=${REPO:-$(git -C "$HERE" rev-parse --show-toplevel)}
mkdir -p "$OUT/serve-logs" || exit 2
[ -x "$BIN" ] || { echo "FATAL: $BIN is missing"; exit 2; }
[ -e "$HOME/models/phi3-mini-4k-gguf/Phi-3-mini-4k-instruct-q4.gguf" ] || { echo "FATAL: the checkpoint is missing"; exit 2; }
{ echo "started: $(date '+%F %T %Z')"; echo "binary: $BIN sha256 $(sha256sum "$BIN" | cut -c1-16)"
  echo "gpu: $(nvidia-smi --query-gpu=name,driver_version,memory.used --format=csv,noheader)"; echo "load: $(cat /proc/loadavg)"
  echo "mem: $(free -g | sed -n 2p)"; } | tee "$OUT/provenance.txt"
for _ in $(seq 1 90); do awk '{exit !($1 < 0.9)}' /proc/loadavg && break; sleep 10; done
awk '{exit !($1 < 1.0)}' /proc/loadavg || { echo "!! the box did not go idle: $(cat /proc/loadavg)"; exit 2; }
python3 "$HERE/probe_lane_cost.py" "$BIN" "$OUT" || { echo "!! the arms did not launch their own attention kernels (or a probe failed): the cell does not run"; exit 2; }
for _ in $(seq 1 90); do awk '{exit !($1 < 0.9)}' /proc/loadavg && break; sleep 10; done   # the probe's loads, settled before bench_peer's idle preflight
rc=0
(cd "$REPO" && env BENCH_ENGINES=goinfer,goinfer_old BENCH_BACKENDS=cuda BENCH_DEPTH_BACKEND=cuda BENCH_MODELS=phi3-mini BENCH_DEPTHS=3900 \
  BENCH_QUANT_OVERRIDE="phi3-mini=default" BENCH_RUNS=2 BENCH_ABBA=1 BENCH_SERVE_LOG_DIR="$OUT/serve-logs" BENCH_LOAD_TIMEOUT=240 \
  GOINFER_SERVE_CUDA="$BIN" GOINFER_SERVE_CUDA_OLD="$BIN" BENCH_GOINFER_ARGS="-flash-decode-hd96" BENCH_GOINFER_OLD_ARGS="-flash-decode-hd96=false" \
  python3 -u scripts/bench_peer.py "$OUT/decode-cuda-all.json") 2>&1 | tee "$OUT/decode-cuda.log"
[ "${PIPESTATUS[0]}" = 0 ] && [ -s "$OUT/decode-cuda-all.json" ] || { echo "!! the run failed or wrote no results (bench_peer exits 0 when it refuses a busy box)"; rc=1; }
grep -l "flash-decode lane unavailable" "$OUT"/serve-logs/*.log 2>/dev/null && { echo "VOID: a serve log reports the lane unavailable"; rc=1; }
python3 "$HERE/grade_cell.py" "$OUT" 2>&1 | tee "$OUT/verdict.txt"
[ "${PIPESTATUS[0]}" = 0 ] || rc=1
echo "finished: $(date '+%F %T %Z') rc=$rc" | tee -a "$OUT/provenance.txt"
exit $rc
