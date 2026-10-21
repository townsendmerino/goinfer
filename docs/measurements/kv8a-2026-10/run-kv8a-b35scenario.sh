#!/bin/bash
# B35's end-to-end scenario on the KV8a build, a measurement only (README.md here, "B35's case on the KV8a build"): B35's own gate.py and
# grade.py at their pinned commit (fbae3175), with KV8a's serve in the place of B35's and B35's base build as the base. Queued on night.py
# (the job holds the timing lock) as `bash run-kv8a-b35scenario.sh`. Exit 0 when it ran (whatever grade.py says), 2 when it could not run.
set -u
B=$HOME/goinfer-bench/kv8a; E=$HOME/goinfer-bench/b35-e2e
KV8A=$B/serve-cuda-466aa167; BASE=$E/bin/serve-cuda-base-fa8efaa0; HERE=$E/wt-b35/docs/measurements/b35-e2e-2026-10-20
MODEL=$HOME/models/Qwen3.5-9B-Q4_K_M.gguf
OUT=$B/b35scenario-$(date +%F-%H%M); mkdir -p "$OUT"
exec > >(tee -a "$OUT/run.log") 2>&1
echo "=== $(date '+%F %T %Z') start: KV8a serve $KV8A, base $BASE, harness $(git -C "$E/wt-b35" rev-parse --short=8 HEAD); $(nvidia-smi --query-gpu=name,driver_version,memory.used --format=csv,noheader); $(uptime)"
for f in "$KV8A" "$BASE" "$MODEL" "$HERE/gate.py" "$HERE/grade.py"; do [ -e "$f" ] || { echo "VOID: $f is missing"; exit 2; }; done
[ "$(git -C "$E/wt-b35" rev-parse --short=8 HEAD)" = fbae3175 ] || { echo "VOID: B35's harness is not at fbae3175"; exit 2; }
python3 -u "$HERE/gate.py" --b35 "$KV8A" --base "$BASE" --model "$MODEL" --out "$OUT" || { echo "VOID: gate.py exited $?"; exit 2; }
python3 "$HERE/grade.py" "$OUT/gate.json" | tee "$OUT/grade.txt"
echo "== where each warm turn diverged from what the server held, and what it restored (serve logs of arms A and B)"
for f in "$OUT"/serve-*-A.log "$OUT"/serve-*-B.log; do
	echo "-- $(basename "$f")"
	grep '^request: POST' "$f" | grep -v 'completion_tokens=0 ' | grep -oE '(prompt_tokens|reused|div|ckpt_restore|ckpt_save_ms|ttft)=[^ ]+' | paste -sd' ' | sed 's/ prompt_tokens/\nprompt_tokens/g'
done
echo "=== $(date '+%F %T %Z') end; record $OUT"
exit 0
