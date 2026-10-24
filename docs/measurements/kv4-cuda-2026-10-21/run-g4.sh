#!/bin/bash
# KV4b's G4 and G5 for one family group at one depth (docs/measurements/kv4-cuda-2026-10-21.md §2, G5, as amended 2026-10-22): the
# CPU f32-activation reference of held-out prompt set B, then the f32 and f16 arms of cuda.TestKV16GateVsReference against it (the
# overflow survey, G5, runs inside the f32 arm and prints before any score). K=3900 reads all ten prompts, by day; K=8000 reads the
# first four, on the night queue.
#
#   env CELL=<QMOE|G34B|...> DEPTH=<3900|8000> PHASE=<ref|gate|both> [W=<worktree>] [WORKERS=n] [N=n] [LONG_ANYWAY=1] bash <this script>
#
# N is the registered prompt count when a cell's registration differs from its depth's (DeepSeek-V2-Lite's K=3900 reads four).
#
# Short first: a K=8000 cell runs only after the cell's K=3900 gate read PASSES (its latest gate log here); LONG_ANYWAY=1 runs it
# regardless. WORKERS defaults to 1 at K=8000 and 2 at K=3900: concurrent prompts at depth contend for memory bandwidth and finish
# together, so a timeout loses every one in flight (decoder/prefill_ref_gen_test.go:refWorkers). The reference phase does not start a
# prompt that its slowest finished prompt says would end past REF_TIMEOUT, and a rerun resumes from the cache.
#
# Both phases hold the timing lock shared (a CPU or GPU load, nothing timed). Logs go to ~/goinfer-bench/kv4-g4/<cell>-K<depth>-<phase>-<date>.log.
set -u
export PATH=$PATH:/usr/local/go/bin
CELL=${CELL:?name the cell}
DEPTH=${DEPTH:?3900 or 8000}
PHASE=${PHASE:-both}
W=${W:-$(cd "$(dirname "$0")/../../.." && pwd)}
case $DEPTH in 3900) DEFN=10 DEFW=2 ;; 8000) DEFN=4 DEFW=1 ;; *) echo "DEPTH must be 3900 or 8000"; exit 2 ;; esac
N=${N:-$DEFN}
WORKERS=${WORKERS:-$DEFW}
OUT=$HOME/goinfer-bench/kv4-g4
mkdir -p "$OUT"
if [ "$DEPTH" = 8000 ] && [ -z "${LONG_ANYWAY:-}" ]; then
  short=$(ls -t "$OUT/$CELL"-K3900-gate-*.log 2>/dev/null | head -1)
  if [ -z "$short" ] || ! grep -q '^=== KV16 GATE .*=> PASSES' "$short"; then
    echo "short first: $CELL K=3900 has no PASSES gate log (${short:-none}), so K=8000 does not run; LONG_ANYWAY=1 overrides"
    exit 0
  fi
  echo "short first: $CELL K=3900 PASSES in $short"
fi
stamp=$(date +%F-%H%M)
nref() { ls "$HOME/goinfer-logs/prefill-ref-b/$CELL-K$DEPTH"-p*.bin 2>/dev/null | grep -c '\.bin$'; }
rc=0
if [ "$PHASE" = ref ] || [ "$PHASE" = both ]; then
  log=$OUT/$CELL-K$DEPTH-ref-$stamp.log
  { echo "started: $(date '+%F %T %Z')"; echo "tree: $W at $(git -C "$W" rev-parse HEAD)"; echo "load: $(uptime)"; free -m | sed -n 1,2p; } > "$log"
  (cd "$W" && python3 scripts/timing_lock.py run --shared --wait "${LOCK_WAIT:-3600}" --label "kv4b-g4-ref-$CELL-$DEPTH" -- \
    env GOINFER_HEAVY_TESTS=1 GOINFER_PREFILL_GATE_PROMPTS=b GOINFER_CPU_REF_MODELS="$CELL" GOINFER_CPU_REF_KS="$DEPTH" \
    GOINFER_CPU_REF_PROMPTS_AT="$DEPTH:$N" GOINFER_CPU_REF_WORKERS="$WORKERS" \
    go test -count=1 -tags goinfer_testhooks -run '^TestPrefillGateReference$' -v -timeout "${REF_TIMEOUT:-3h}" ./decoder/) 2>&1 | tee -a "$log"
  [ "${PIPESTATUS[0]}" = 0 ] && grep -q "^--- PASS: TestPrefillGateReference" "$log" || rc=1
  echo "reference files for $CELL K=$DEPTH: $(nref) (want at least $N)" | tee -a "$log"
  [ "$(nref)" -ge "$N" ] || rc=1
fi
if [ "$rc" = 0 ] && { [ "$PHASE" = gate ] || [ "$PHASE" = both ]; }; then
  log=$OUT/$CELL-K$DEPTH-gate-$stamp.log
  { echo "started: $(date '+%F %T %Z')"; echo "tree: $W at $(git -C "$W" rev-parse HEAD)"; nvidia-smi --query-gpu=driver_version,name,memory.used --format=csv,noheader; } > "$log"
  (cd "$W" && python3 scripts/timing_lock.py run --shared --wait "${LOCK_WAIT:-3600}" --label "kv4b-g4-gate-$CELL-$DEPTH" -- \
    env GOINFER_HEAVY_TESTS=1 GOINFER_PREFILL_GATE_PROMPTS=b GOINFER_CUDA_FLASH_DECODE=16 GOINFER_KV4B_GATE_K="$DEPTH" GOINFER_KV4B_GATE_N="$N" \
    go test -count=1 -tags 'cuda goinfer_testhooks' -run "^TestKV4bGate_$CELL\$" -v -timeout 3h ./cuda/) 2>&1 | tee -a "$log"
  [ "${PIPESTATUS[0]}" = 0 ] || rc=1
  grep -E "^=== KV16 (GATE|OVERFLOW)" "$log"
  [ "$(grep -c '^=== KV16 GATE' "$log")" = 1 ] || rc=1
fi
echo "finished: $(date '+%F %T %Z') rc=$rc"
exit "$rc"
