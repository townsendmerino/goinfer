#!/usr/bin/env bash
# G-RF3-S (docs/tasks/task-forward-and-memory-2026-11.md, "RF3: plan and registration"), on nobara, by day, exclusive timing lock.
#   run-g-rf3-s.sh <out dir> <serve-cuda-A> <serve-cuda-B>
# A = the alternation (RF3's merge base), B = RF3. Build each once from its own worktree, as run-g-rf2-s.sh does:
#   (cd <wt>/backend/cuda && GOWORK=<wt>/go.work CGO_ENABLED=0 go build -tags cuda -o <bins>/serve-cuda-A ./cmd/serve), and serve-cuda-B.
# Order: the A/A first (two A-A pairs on the 1.5B stall cell; a TTFT ratio off 1.00 by more than 0.03 voids the read and stops
# here), then per cell four A B B A blocks: each B is paired with the A next to it. Cells: 1.5B and 7B, stall (3 decoders, 3
# newcomers, 2400 words) and solo (no decoders). Score: python3 score_rf3s.py <out dir>.
set -euo pipefail
SELF=$(realpath "$0")
idle() { for _ in $(seq 90); do awk '{exit !($1 < 0.9)}' /proc/loadavg && return 0; sleep 10; done; echo "box not idle after 15 min"; return 1; }
if [ "${1:-}" = __cell ]; then
  shift
  idle || exit 75
  echo "$(date +%T) cell start load $(cut -d' ' -f1 /proc/loadavg)"
  exec "$@"
fi
OUT=$(realpath -m "${1:?out dir}"); A=$(realpath -e "${2:?serve-cuda-A}"); B=$(realpath -e "${3:?serve-cuda-B}")
[ "$(sha256sum < "$A")" != "$(sha256sum < "$B")" ] || { echo "A and B are the same binary"; exit 2; }
REPO=$(cd "$(dirname "$SELF")/../../.." && pwd)
mkdir -p "$OUT"; cd "$REPO"
lock() { python3 scripts/timing_lock.py run --wait 1800 --label "g-rf3-s-$1" -- bash "$SELF" __cell "${@:2}"; }
# run <cell> <model file> <extra stall args> <key> <binary>
run() {
  local cell=$1 model=$2 extra=$3 key=$4 bin=$5
  echo "$(date +%T) $cell $key $(basename "$bin") sha256 $(sha256sum < "$bin" | cut -c1-16)" | tee -a "$OUT/cells.log"
  # shellcheck disable=SC2086
  lock "$cell-$key" env GOINFER_SERVE_CPU=$bin BENCH_W7_MODEL=$HOME/models/$model \
    python3 scripts/bench_prefill_stall.py "$OUT/$cell.json" --key "$key" --serve-args="-backend cuda -require-backend" \
      --prompt-words 2400 $extra --server-log "$OUT/serve-$cell.log" 2>&1 | tee -a "$OUT/$cell.log"
}
M15=qwen2.5-coder-1.5b-instruct-q4_k_m.gguf; M7=qwen2.5-7b-instruct-q4_k_m.gguf
# A/A, same session, 1.5B stall: key aa0-A0 aa0-A1 / aa1-A1 aa1-A0 (the ABBA shape on one binary)
for k in aa0-A0 aa0-B1 aa0-B2 aa0-A3; do run 15-stall-aa $M15 "" $k $A; done
python3 "$(dirname "$SELF")/score_rf3s.py" "$OUT" --aa || { echo "A/A off 1.00 by more than 0.03: VOID, stop, coordinator"; exit 3; }
for spec in "15-stall $M15 " "15-solo $M15 --decoders=0" "7-stall $M7 " "7-solo $M7 --decoders=0"; do
  set -- $spec; cell=$1; model=$2; extra=${3:-}
  for blk in 0 1 2 3; do
    run "$cell" "$model" "$extra" "b$blk-A0" "$A"; run "$cell" "$model" "$extra" "b$blk-B1" "$B"
    run "$cell" "$model" "$extra" "b$blk-B2" "$B"; run "$cell" "$model" "$extra" "b$blk-A3" "$A"
  done
done
echo RUN-G-RF3-S-DONE
