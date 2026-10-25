#!/bin/bash
# KV4b's G4 for several family groups at one depth (kv4-cuda-2026-10-21.md §2, G5, as amended 2026-10-22): run-g4.sh's reference
# phase and then its gate, cell after cell. Nothing is skipped on a count of files: the reference phase finds in the cache only
# references this tree's code would compute (keyed by its source) and computes the rest, and the gate refuses any reference whose
# sidecar names another prompt or source (decoder.PrefillRefCurrentForTest).
#
#   env DEPTH=<3900|8000> CELLS="G34B CR7B ..." [W=<worktree>] bash <this script>
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
W=${W:-$(cd "$HERE/../../.." && pwd)}
DEPTH=${DEPTH:?3900 or 8000}
rc=0
for c in ${CELLS:?}; do
  CELL=$c DEPTH=$DEPTH PHASE=both W=$W bash "$HERE/run-g4.sh" || { echo "$c K=$DEPTH: reference or gate FAILED (see its logs)"; rc=1; }
done
echo "chain K=$DEPTH finished: $(date '+%F %T %Z') rc=$rc"
exit "$rc"
