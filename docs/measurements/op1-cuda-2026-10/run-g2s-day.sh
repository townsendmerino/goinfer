#!/bin/bash
# OP1 on CUDA, the speed guard G2-S as a tier 1 run by day (owner, 2026-10-16), under the timing lock:
#   python3 scripts/timing_lock.py run --label op1-g2s-day -- bash docs/measurements/op1-cuda-2026-10/run-g2s-day.sh
# New (the head under test) against base (78e4faf2), served, two arms, no peer arm. One dense model (Qwen2.5-Coder 1.5B) and one
# sliding-window model (Gemma 3 4B, local layers, head dimension 256): decode at depth 128 (bench_peer.py) and time to first token
# (bench_prefill_stall.py, no decoders, three newcomers of about 2,400 words a cell). Blocks are paired, run order mirrored (BENCH_ABBA).
# Why this is enough (the record has the argument): the plain PTX of every attention module is byte for byte base's, so the device code
# of the two arms is the same; what differs is host code that picks a module and appends a cap argument when the model has one.
set -uo pipefail
NEW_WT=${NEW_WT:-/home/francis/wt/op1-cuda}
OLD_WT=${OLD_WT:-/home/francis/wt/op1-cuda-base}
B=${B:-$HOME/goinfer-bench/op1-cuda-2026-10}
OUT=${OUT:-$B/g2s-day-$(date +%H%M)}
ABBA=${ABBA:-2}
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$B/bin" "$OUT/serve-logs" || exit 2
for p in "$HOME/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf" "$HOME/models/gemma-3-4b-it-gguf/gemma-3-4b-it-Q4_K_M.gguf"; do
  [ -e "$p" ] || { echo "FATAL: $p is missing"; exit 2; }
done
build() { (cd "$2/cuda" && CGO_ENABLED=0 go build -tags cuda -o "$B/bin/serve-cuda-$1" ./cmd/serve) || { echo "FATAL: the $1 build failed"; exit 2; }; git -C "$2" rev-parse HEAD > "$B/bin/serve-cuda-$1.rev"; }
build new "$NEW_WT"; build old "$OLD_WT"
NEW=$B/bin/serve-cuda-new; OLD=$B/bin/serve-cuda-old
{ echo "started: $(date '+%F %T %Z')"; echo "new: $(cat "$NEW.rev")"; echo "old: $(cat "$OLD.rev")"; echo "blocks: ABBA=$ABBA"
  echo "gpu: $(nvidia-smi --query-gpu=name,driver_version,memory.used --format=csv,noheader)"; echo "load: $(uptime)"; } | tee "$OUT/provenance.txt"
[ "${DRY:-0}" = 1 ] && exit 0
rc=0
wait_idle() { # the harness refuses to measure above a 1-minute load average of 1.0 (bench_peer.py's idle gate), and the builds above raise it
  for _ in $(seq 1 90); do
    awk '{exit !($1 < 0.9)}' /proc/loadavg && return 0
    sleep 10
  done
  echo "!! the box did not go idle in 15 minutes: $(cat /proc/loadavg)"; return 1
}
wait_idle || exit 2
echo "== decode, depth 128: started $(date '+%T')"
(cd "$NEW_WT" && env BENCH_ENGINES=goinfer,goinfer_old BENCH_BACKENDS=cuda BENCH_DEPTH_BACKEND=cuda BENCH_MODELS=1.5B,gemma3-4b BENCH_DEPTHS=none BENCH_ABBA=$ABBA \
  BENCH_REQUIRE_BACKEND=1 BENCH_SERVE_LOG_DIR="$OUT/serve-logs" BENCH_LOAD_TIMEOUT=240 GOINFER_SERVE_CUDA="$NEW" GOINFER_SERVE_CUDA_OLD="$OLD" \
  python3 -u scripts/bench_peer.py "$OUT/decode-cuda-all.json") 2>&1 | tee "$OUT/decode-cuda.log"
[ "${PIPESTATUS[0]}" = 0 ] && [ -s "$OUT/decode-cuda-all.json" ] || { echo "!! the decode pass failed or wrote no results (bench_peer exits 0 when it refuses a busy box)"; rc=1; }
ttft() { # ttft <model key> <model path>
  wait_idle || { rc=1; return; }
  echo "== time to first token, $1: started $(date '+%T')"
  local cells=()
  for ((i = 1; i <= ABBA * 2; i++)); do
    if (( i % 2 )); then cells+=("old_$i" "new_$i"); else cells+=("new_$i" "old_$i"); fi
  done
  for k in "${cells[@]}"; do
    case $k in old*) bin=$OLD ;; new*) bin=$NEW ;; esac
    echo "$(date '+%T') cell $k"
    (cd "$NEW_WT" && BENCH_W7_MODEL="$2" GOINFER_SERVE_CPU=$bin python3 -u scripts/bench_prefill_stall.py "$OUT/ttft-cuda-$1.json" --key "$k" --decoders 0 --newcomers 3 \
      --prompt-words 2400 --serve-args "-backend cuda" --server-log "$OUT/serve-logs/ttft-$1.log") 2>&1 | tee -a "$OUT/ttft-cuda.log"
    [ "${PIPESTATUS[0]}" = 0 ] || { echo "!! time-to-first-token cell $k failed"; rc=1; }
  done
}
ttft 1.5B "$HOME/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"
ttft gemma3-4b "$HOME/models/gemma-3-4b-it-gguf/gemma-3-4b-it-Q4_K_M.gguf"
python3 "$HERE/grade_speed.py" "$OUT" 2>&1 | tee "$OUT/verdict.txt"
[ "${PIPESTATUS[0]}" = 0 ] || rc=1
echo "finished: $(date '+%F %T %Z') rc=$rc" | tee -a "$OUT/provenance.txt"
exit $rc
