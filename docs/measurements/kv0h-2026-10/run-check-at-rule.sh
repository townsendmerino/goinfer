#!/bin/bash
# KV0-H: apply the Claude Code context rule (calibrate_ctx.py), then run the Claude Code check at the context it lands on, under the lock (the caller holds it):
#   python3 scripts/timing_lock.py run --wait 21600 --label kv0h-check-cc -- bash docs/measurements/kv0h-2026-10/run-check-at-rule.sh
set -u
K=$(cd "$(dirname "$0")" && pwd); R=$(cd "$K/../../.." && pwd)
B=/home/francis/goinfer-bench/kv0h-2026-10; OUT=$B/check-cc-rule; mkdir -p "$OUT"
OTHERS=$(nvidia-smi --query-compute-apps=pid,used_memory,name --format=csv,noheader | grep -v kwin_wayland | grep -c .)
echo "START $(date '+%F %T %Z'); GPU before: $(nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader); other GPU processes: $OTHERS"
[ "$OTHERS" = 0 ] || { echo "VOID: another process holds the card"; echo "END VOID"; exit 3; }
CTX=$(python3 -u "$K/calibrate_ctx.py" /home/francis/serve-cuda-94a6b973 /home/francis/models/Qwen3.5-9B-Q4_K_M.gguf \
  "$B/bodies-pilot/pilot-qwen35-9b-28k-cc-claude-code-chain-r1/0001.json" "$B/ids.key" "$OUT" | tee /dev/stderr | tail -1)
echo "the rule lands on: $CTX"
case "$CTX" in ''|NONE) echo "END no context passes the rule"; exit 2;; esac
python3 - "$CTX" "$K/plan-check-cc.json" "$OUT/plan-check-at-rule.json" <<'PY'
import json, sys
ctx, src, dst = int(sys.argv[1]), sys.argv[2], sys.argv[3]
p = json.load(open(src)); p["configs"][0]["ctx"] = ctx; p["configs"][0]["name"] = f"check-qwen35-9b-{ctx}-cc"
p["configs"][0]["run_timeout"] = 1500
json.dump(p, open(dst, "w"), indent=1)
PY
bash "$K/run-kv0h.sh" "$OUT/plan-check-at-rule.json" "$OUT/run" "$OUT/bodies"
echo "END $(date '+%F %T %Z')"
