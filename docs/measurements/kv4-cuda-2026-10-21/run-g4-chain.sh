#!/bin/bash
# KV4b's G4 for every family group (kv4-cuda-2026-10-21.md §2, G5), as two loops: the CPU references one after another, and the GPU
# gates one after another, each gate starting once its cell's twenty reference files exist. run-g4.sh does one phase of one cell.
#
#   env REF_CELLS="G34B CR7B ..." GATE_CELLS="QMOE G34B ..." [W=<worktree>] bash <this script> refs|gates
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
W=${W:-$(cd "$HERE/../../.." && pwd)}
REFS=$HOME/goinfer-logs/prefill-ref-b
nref() { ls "$REFS/$1"-K*-p*.bin 2>/dev/null | wc -l; }
case ${1:?refs or gates} in
  refs)
    for c in ${REF_CELLS:?}; do
      [ "$(nref "$c")" = 20 ] && { echo "$c: reference present"; continue; }
      CELL=$c PHASE=ref LOCK_WAIT=${LOCK_WAIT:-14400} W=$W bash "$HERE/run-g4.sh" || echo "$c: reference FAILED"
    done ;;
  gates)
    for c in ${GATE_CELLS:?}; do
      for _ in $(seq 720); do [ "$(nref "$c")" = 20 ] && break; sleep 60; done # up to 12 hours for its reference
      [ "$(nref "$c")" = 20 ] || { echo "$c: no reference, gate NOT RUN"; continue; }
      CELL=$c PHASE=gate W=$W bash "$HERE/run-g4.sh" || echo "$c: gate did not pass or did not run (see its log)"
    done ;;
esac
echo "chain $1 finished: $(date '+%F %T %Z')"
