#!/bin/bash
# KV4b's G4 and G5 for one family group (docs/measurements/kv4-cuda-2026-10-21.md §2, G5): the CPU f32-activation reference of held-out
# prompt set B at K=3900 and 8000, then the f32 and f16 arms of cuda.TestKV16GateVsReference against it (the overflow survey, G5, runs
# inside the f32 arm and prints before any score).
#
#   env CELL=<QMOE|...> PHASE=<ref|gate|both> [W=<worktree>] bash <this script>
#
# ref holds the timing lock shared (a CPU load, nothing timed); gate is a correctness run, also shared. Logs go to
# ~/goinfer-bench/kv4-g4/<cell>-<phase>-<date>.log.
set -u
export PATH=$PATH:/usr/local/go/bin
CELL=${CELL:?name the cell}
PHASE=${PHASE:-both}
W=${W:-$(cd "$(dirname "$0")/../../.." && pwd)}
OUT=$HOME/goinfer-bench/kv4-g4
mkdir -p "$OUT"
stamp=$(date +%F-%H%M)
rc=0
if [ "$PHASE" = ref ] || [ "$PHASE" = both ]; then
  log=$OUT/$CELL-ref-$stamp.log
  { echo "started: $(date '+%F %T %Z')"; echo "tree: $W at $(git -C "$W" rev-parse HEAD)"; echo "load: $(uptime)"; free -m | sed -n 1,2p; } > "$log"
  (cd "$W" && python3 scripts/timing_lock.py run --shared --wait ${LOCK_WAIT:-3600} --label "kv4b-g4-ref-$CELL" -- \
    env GOINFER_HEAVY_TESTS=1 GOINFER_PREFILL_GATE_PROMPTS=b GOINFER_CPU_REF_MODELS="$CELL" GOINFER_CPU_REF_WORKERS="${WORKERS:-2}" \
    go test -count=1 -tags goinfer_testhooks -run '^TestPrefillGateReference$' -v -timeout 4h ./decoder/) 2>&1 | tee -a "$log"
  [ "${PIPESTATUS[0]}" = 0 ] && grep -q "^--- PASS: TestPrefillGateReference" "$log" || rc=1
  n=$(ls "$HOME/goinfer-logs/prefill-ref-b/$CELL"-K*-p*.bin 2>/dev/null | wc -l)
  echo "reference files for $CELL: $n (want 20)" | tee -a "$log"
  [ "$n" = 20 ] || rc=1
fi
if [ "$rc" = 0 ] && { [ "$PHASE" = gate ] || [ "$PHASE" = both ]; }; then
  log=$OUT/$CELL-gate-$stamp.log
  { echo "started: $(date '+%F %T %Z')"; echo "tree: $W at $(git -C "$W" rev-parse HEAD)"; nvidia-smi --query-gpu=driver_version,name,memory.used --format=csv,noheader; } > "$log"
  (cd "$W" && python3 scripts/timing_lock.py run --shared --wait 3600 --label "kv4b-g4-gate-$CELL" -- \
    env GOINFER_HEAVY_TESTS=1 GOINFER_PREFILL_GATE_PROMPTS=b GOINFER_CUDA_FLASH_DECODE=16 GOINFER_KV16_GATE_CELLS="$CELL:3900,$CELL:8000" \
    go test -count=1 -tags 'cuda goinfer_testhooks' -run '^TestKV16GateVsReference$' -v -timeout 3h ./cuda/) 2>&1 | tee -a "$log"
  [ "${PIPESTATUS[0]}" = 0 ] || rc=1
  grep -E "^=== KV16 (GATE|OVERFLOW)" "$log"
  [ "$(grep -c '^=== KV16 GATE' "$log")" = 2 ] || rc=1
fi
echo "finished: $(date '+%F %T %Z') rc=$rc"
exit "$rc"
