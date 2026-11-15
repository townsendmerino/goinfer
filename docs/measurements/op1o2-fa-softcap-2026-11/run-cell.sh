#!/bin/bash
# OP1-o2's timed cell (README, "The timed cell" and its 2026-11-15 addendum), by day, under the timing lock:
#   python3 scripts/timing_lock.py run --label op1o2-cell -- bash docs/measurements/op1o2-fa-softcap-2026-11/run-cell.sh
# One serve binary serves both arms; the opted-in arm adds -flash-decode-softcap. Gemma 2 2B, depths 128 and 2,048, two runs a cell,
# one counterbalanced quad (BENCH_ABBA=1). Grades with grade_cell.py.
set -uo pipefail
BIN=${BIN:-$HOME/goinfer-bench/op1o2-2026-11/bin/serve-cuda}
OUT=${OUT:-$HOME/goinfer-bench/op1o2-2026-11/cell-$(date +%H%M)}
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=${REPO:-$(git -C "$HERE" rev-parse --show-toplevel)}
mkdir -p "$OUT/serve-logs" || exit 2
[ -x "$BIN" ] || { echo "FATAL: $BIN is missing"; exit 2; }
[ -e "$HOME/models/gemma-2-2b-it-Q8_0.gguf" ] || { echo "FATAL: the checkpoint is missing"; exit 2; }
{ echo "started: $(date '+%F %T %Z')"; echo "binary: $BIN ($(cat "$BIN.rev" 2>/dev/null))"
  echo "gpu: $(nvidia-smi --query-gpu=name,driver_version,memory.used --format=csv,noheader)"; echo "load: $(cat /proc/loadavg)"
  echo "mem: $(free -g | sed -n 2p)"; } | tee "$OUT/provenance.txt"
for _ in $(seq 1 90); do awk '{exit !($1 < 0.9)}' /proc/loadavg && break; sleep 10; done
awk '{exit !($1 < 1.0)}' /proc/loadavg || { echo "!! the box did not go idle: $(cat /proc/loadavg)"; exit 2; }
python3 "$HERE/probe_arms.py" "$BIN" "$OUT" || { echo "!! the arms did not differ (or a probe failed): the cell does not run"; exit 2; }
rc=0
(cd "$REPO" && env BENCH_ENGINES=goinfer,goinfer_old BENCH_BACKENDS=cuda BENCH_DEPTH_BACKEND=cuda BENCH_MODELS=gemma2-2b BENCH_DEPTHS=2048 \
  BENCH_RUNS=2 BENCH_ABBA=1 BENCH_REQUIRE_BACKEND=1 BENCH_SERVE_LOG_DIR="$OUT/serve-logs" BENCH_LOAD_TIMEOUT=240 \
  GOINFER_SERVE_CUDA="$BIN" GOINFER_SERVE_CUDA_OLD="$BIN" BENCH_GOINFER_ARGS="-flash-decode-softcap" BENCH_GOINFER_OLD_ARGS="-flash-decode-softcap=false" \
  python3 -u scripts/bench_peer.py "$OUT/decode-cuda-all.json") 2>&1 | tee "$OUT/decode-cuda.log"
[ "${PIPESTATUS[0]}" = 0 ] && [ -s "$OUT/decode-cuda-all.json" ] || { echo "!! the run failed or wrote no results (bench_peer exits 0 when it refuses a busy box)"; rc=1; }
grep -l "flash-decode lane unavailable" "$OUT"/serve-logs/*.log 2>/dev/null && { echo "VOID: a serve log reports the lane unavailable"; rc=1; }
python3 "$HERE/grade_cell.py" "$OUT" 2>&1 | tee "$OUT/verdict.txt"
[ "${PIPESTATUS[0]}" = 0 ] || rc=1
echo "finished: $(date '+%F %T %Z') rc=$rc" | tee -a "$OUT/provenance.txt"
exit $rc
