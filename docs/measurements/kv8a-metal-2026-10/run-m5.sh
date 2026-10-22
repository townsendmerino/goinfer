#!/bin/bash
# Metal KV8a gates M5 and M4(b), on the Mac's night queue (README.md here; m5.py), one pass per job: `run-m5.sh <pass 1|2> <record name>`,
# both passes under the same record name; pass 2's job grades both. Exit as m5.py (0 PASS or a pass recorded, 1 FAIL, 3 PARKED, 2 VOID),
# or 75 when the start is not clean or the fit guard refuses the 9B (night.py serve re-queues it until 06:00).
set -u
. "$(dirname "$0")/common.sh"
PASS=${1:?the pass, 1 or 2}; OUT=$K/${2:?a record name shared by both passes}; mkdir -p "$OUT"
exec > >(tee -a "$OUT/run-pass$PASS.log") 2>&1
echo "=== $(date '+%F %T %Z') start; $(pmset -g batt | head -1)"
[ -x "$BEFORE" ] || { echo "VOID: $BEFORE is missing"; exit 2; }
purge_then_check 600 || { echo "not clean at the start: re-queued"; exit 75; }
fit_check 20480 || { echo "the fit guard refused the 9B: re-queued"; exit 75; }
python3 -u "$(dirname "$0")/m5.py" --before "$BEFORE" --kv8a "$KV8A" --model "$MODEL_ARG" --key "$H/ids.key" --bodies "$H/bodies-pilot" \
	--ids "$H/pilot/ids.jsonl" --out "$OUT" --pass "$PASS"
rc=$?
echo "=== $(date '+%F %T %Z') end, rc $rc; record $OUT"
exit $rc
