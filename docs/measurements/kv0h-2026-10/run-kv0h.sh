#!/bin/bash
# KV0-H collection + tokenization + analysis for one plan, under the timing lock (the caller holds it):
#   python3 scripts/timing_lock.py run --wait 14400 --label kv0h-<x> -- bash docs/measurements/kv0h-2026-10/run-kv0h.sh <plan.json> <out dir> <bodies dir>
# Bodies and ids stay OUTSIDE the repository (the bodies carry each harness's own prompts); only analyze_h.py's statistics are committed.
set -u
PLAN=$1; OUT=$2; BODIES=$3
K=$(cd "$(dirname "$0")" && pwd); R=$(cd "$K/../../.." && pwd)
mkdir -p "$OUT" "$BODIES"
echo "START $(date '+%F %T %Z') plan $PLAN commit $(git -C "$R" rev-parse --short=8 HEAD) loadavg $(cut -d' ' -f1 /proc/loadavg)"
OTHERS=$(nvidia-smi --query-compute-apps=pid,used_memory,name --format=csv,noheader | grep -v kwin_wayland | grep -c .)
echo "GPU before: $(nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader); other GPU processes: $OTHERS"
[ "$OTHERS" = 0 ] || { echo "VOID: another process holds the card"; nvidia-smi --query-compute-apps=pid,used_memory,name --format=csv,noheader; echo "END VOID"; exit 3; }
python3 -u "$K/run_sessions.py" "$PLAN" "$OUT" "$BODIES"; rc=$?
echo "sessions rc=$rc $(date '+%T')"
MODEL=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['configs'][0]['model'])" "$PLAN")
echo "== tokenize the recorded bodies with $MODEL (one model per plan)"
KEYFILE=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1])).get('ids_key_file',''))" "$PLAN")
NAME=$(python3 -c "import os,sys; print(os.path.splitext(os.path.basename(sys.argv[1]))[0])" "$MODEL")
(cd "$R" && KV0_BODIES="$BODIES" KV0_GGUF="$MODEL" KV0_OUT="$OUT/ids.jsonl" KV0_IDS_KEY_FILE="$KEYFILE" KV0_MODEL_NAME="$NAME" go test -count=1 -run '^TestKV0_tokenizeBodies$' -v ./internal/serveapp/ 2>&1 | tail -4)
for CFG in $(python3 -c "import json,sys; print(' '.join(c['name'] for c in json.load(open(sys.argv[1]))['configs']))" "$PLAN"); do
  [ -f "$OUT/logs/tokenids-$CFG.jsonl" ] || { echo "no token-id log for $CFG"; continue; }
  python3 "$K/analyze_h.py" --sessions "$OUT/sessions.jsonl" --wire "$OUT" --ids "$OUT/ids.jsonl" --tokenids "$OUT/logs/tokenids-$CFG.jsonl" \
    --serve-log "$OUT/logs/serve-$CFG.log" --out "$OUT/stats-h-$CFG.json" --config "$CFG"
done
echo "END $(date '+%F %T %Z') sessions rc=$rc"
