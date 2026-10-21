#!/bin/bash
# Metal KV8a gate M2(b), on the Mac's night queue (README.md here): gate2b.py --backend metal over the 24 pilot pairs. Exit 0 PASS, 1 FAIL,
# 2 VOID or could not run, 75 the start was not clean or the fit guard refused the 9B (night.py serve re-queues it until 06:00).
set -u
. "$(dirname "$0")/common.sh"
OUT=$K/gate2b-$(date +%F-%H%M); mkdir -p "$OUT"
exec > >(tee -a "$OUT/run.log") 2>&1
echo "=== $(date '+%F %T %Z') start; $(pmset -g batt | head -1)"
start_clean 600 || { echo "not clean at the start: re-queued"; exit 75; }
fit_check 32768 || { echo "the fit guard refused the 9B at -ctx 32768: re-queued"; exit 75; }
python3 -u "$(dirname "$0")/gate2b.py" --serve "$KV8A" --model "$MODEL_ARG" --key "$H/ids.key" --bodies "$H/bodies-pilot" --ids "$H/pilot/ids.jsonl" \
	--out "$OUT" --backend metal
rc=$?
echo "=== $(date '+%F %T %Z') end, rc $rc; record $OUT"
exit $rc
