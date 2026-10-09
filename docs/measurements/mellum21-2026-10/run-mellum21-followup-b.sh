#!/usr/bin/env bash
# Follow-up B: P18's decision measurement on Mellum2.1's weights. Does expert-major still earn its default on the RL checkpoint?
# TestMoEExpertMajor_endToEnd at the registered depth (K=4096, 2 pairs, interleaved with alternating lead, after a discarded warm run), int4 on the CPU
# path, flag on vs off in one process. The rule is P18's own, written before 2.0 ran (docs/queue-performance.md P18): >= 15% FUND, < 8% PARK, between is
# AMBIGUOUS. The 2.0 record: 1,207 s per-row against 277 s expert-major at K=4096. The 2.1 engagement read at K=600 (one pair) was 3.37x.
# Timed, so it needs the box to itself (night.py holds the timing lock). Estimate ~80 min expected (warm 1,207 + 2 x (1,207 + 277) s + load), queued at 130.
#   python3 scripts/night.py add mellum21-fu-b --est 130 --by "nobara session, Mellum2.1 follow-up B" --doc docs/tasks/task-mellum21-2026-10.md -- bash docs/measurements/mellum21-2026-10/run-mellum21-followup-b.sh
source "$(dirname "$0")/_common.sh"
OUT=${1:-$HOME/goinfer-logs/mellum21-followup-b-$(date +%F)}
[ -x "$F/decoder.test" ] || fatal "$F/decoder.test missing"; [ -e "$CK/config.json" ] || fatal "$CK missing"
guard_paths; mkdir -p "$OUT"; SUM="$OUT/summary.txt"; : > "$SUM"; provenance
[ -n "${MELLUM_DRY:-}" ] && { echo "DRY: preconditions hold; plan = endToEnd K=4096 pairs=2"; exit 0; }
step e2e gt TestMoEExpertMajor_endToEnd GOINFER_MOE_BATCH_E2E=1 GOINFER_MOE_BATCH_K=4096 GOINFER_MOE_BATCH_PAIRS=2
grep -E "P18 e2e|pair |ratio" "$OUT/e2e.log" | tee -a "$SUM"; verdict e2e TestMoEExpertMajor_endToEnd
echo "finished: $(date '+%F %T %Z')" | tee -a "$OUT/provenance.txt"; exit 0
