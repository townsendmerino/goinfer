#!/bin/bash
# KV4b's G4 for several family groups at one depth (kv4-cuda-2026-10-21.md §2, G5, as amended 2026-10-22), as two loops: the CPU
# references one after another, and the GPU gates one after another, each gate starting once its cell's references exist. run-g4.sh
# does one phase of one cell at one depth.
#
#   env DEPTH=<3900|8000> REF_CELLS="G34B CR7B ..." GATE_CELLS="G34B ..." [W=<worktree>] bash <this script> refs|gates
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
W=${W:-$(cd "$HERE/../../.." && pwd)}
DEPTH=${DEPTH:?3900 or 8000}
case $DEPTH in 3900) N=10 ;; 8000) N=4 ;; *) echo "DEPTH must be 3900 or 8000"; exit 2 ;; esac
nref() { ls "$HOME/goinfer-logs/prefill-ref-b/$1-K$DEPTH"-p*.bin 2>/dev/null | grep -c '\.bin$'; }
case ${1:?refs or gates} in
  refs)
    for c in ${REF_CELLS:?}; do
      [ "$(nref "$c")" -ge "$N" ] && { echo "$c K=$DEPTH: reference present"; continue; }
      CELL=$c DEPTH=$DEPTH PHASE=ref W=$W bash "$HERE/run-g4.sh" || echo "$c K=$DEPTH: reference FAILED"
    done ;;
  gates)
    for c in ${GATE_CELLS:?}; do
      for _ in $(seq 720); do [ "$(nref "$c")" -ge "$N" ] && break; sleep 60; done # up to 12 hours for its reference
      [ "$(nref "$c")" -ge "$N" ] || { echo "$c K=$DEPTH: no reference, gate NOT RUN"; continue; }
      CELL=$c DEPTH=$DEPTH PHASE=gate W=$W bash "$HERE/run-g4.sh" || echo "$c K=$DEPTH: gate did not pass or did not run (see its log)"
    done ;;
esac
echo "chain $1 K=$DEPTH finished: $(date '+%F %T %Z')"
