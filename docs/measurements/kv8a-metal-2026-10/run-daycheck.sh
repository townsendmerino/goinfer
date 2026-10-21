#!/bin/bash
# Metal KV8a, by day (README.md here; Francis 2026-10-21: "lets do 2 by day soon"): the 9B fit check at -ctx 20480 and 32768, then the
# one-pair probe (opencode pair #1, ~7.3k tokens, gate2b.py --limit 1). Queued on night.py, which holds the timing lock, once Francis has
# closed his apps. Exit 0 ran; 3 the fit guard refused the 9B (stop and report; the guard is not changed or bypassed); 2 void or could not run.
set -u
. "$(dirname "$0")/common.sh"
OUT=$K/daycheck-$(date +%F-%H%M); mkdir -p "$OUT"
exec > >(tee -a "$OUT/run.log") 2>&1
echo "=== $(date '+%F %T %Z') start; $(pmset -g batt | head -1)"
for f in "$KV8A" "$MODEL" "$H/ids.key" "$H/pilot/ids.jsonl"; do [ -e "$f" ] || { echo "VOID: $f is missing"; exit 2; }; done
start_clean 300 || { echo "VOID: no clean start in 5 minutes"; exit 2; }
for ctx in 20480 32768; do
	fit_check $ctx || { echo "REFUSED: the fit guard keeps the 9B off Metal at -ctx $ctx with the apps closed"; exit 3; }
done
echo "== probe: opencode pair #1, warm (turn 1 with its first token forbidden, then turn 2) and cold"
python3 -u "$(dirname "$0")/gate2b.py" --serve "$KV8A" --model "$MODEL_ARG" --key "$H/ids.key" --bodies "$H/bodies-pilot" --ids "$H/pilot/ids.jsonl" \
	--out "$OUT/probe" --backend metal --configs h20 --start 1 --limit 1
rc=$?
grep '^request: POST' "$OUT"/probe/serve-*.log | grep -oE '(prompt_tokens|reused|div|ttft|total|ckpt_save_ms|ckpt_restore)=[^ ]+' | paste -sd' ' - | sed 's/ prompt_tokens/\nprompt_tokens/g'
echo "=== $(date '+%F %T %Z') end, rc $rc; the apps may reopen; record $OUT"
exit $rc
