#!/usr/bin/env bash
# G-W2 (correctness on the real model, and the ctx-4096 load) and G-W3 (no speed cost) of docs/tasks/task-cuda-windowed-kv-2026-10.md, Mellum2.1 int4mix on
# nobara. Pre-registered in windowed_kv_gate.py's docstring and the task doc. Timed (night.py holds the timing lock).
# Pinned binary: ~/goinfer-bench/windowed-kv/serve-cuda (rev beside it). Estimate ~12 min (5 loads of ~25 s, 6 short + 6 long requests per arm session).
#   python3 scripts/night.py add windowed-kv-gates --est 25 --by "nobara session, CUDA windowed KV" --doc docs/tasks/task-cuda-windowed-kv-2026-10.md -- bash docs/measurements/cuda-windowed-kv-2026-10/run-windowed-kv-gates.sh
F=${F:-$HOME/goinfer-bench/windowed-kv}
source "$(dirname "$0")/../mellum21-2026-10/_common.sh"
OUT=${1:-$HOME/goinfer-logs/windowed-kv-gates-$(date +%F)}
[ -x "$F/serve-cuda" ] || fatal "$F/serve-cuda missing"; [ -e "$CK/config.json" ] || fatal "$CK missing"
ls "$CK".int4*.giw >/dev/null 2>&1 || fatal "the int4mix .giw cache next to $CK is missing (run a serve once, as Gate 2 did)"
guard_paths; mkdir -p "$OUT"; SUM="$OUT/summary.txt"; : > "$SUM"; provenance
curl -fs http://127.0.0.1:18932/v1/models >/dev/null 2>&1 && fatal "something already serves on 18932"
[ -n "${MELLUM_DRY:-}" ] && { echo "DRY: preconditions hold; plan = F W W F sessions at ctx 2048 (3 short x 3 reps, 1 long x 3 reps) then F (must decline) and W at ctx 4096"; exit 0; }
step drive python3 "$SRC/docs/measurements/cuda-windowed-kv-2026-10/windowed_kv_gate.py" "$F/serve-cuda" "$CK" "$OUT"
grep -E "prompt [0-9]:|ratio|session medians|VRAM|G-W2|G-W3|VOID" "$OUT/drive.log" | tee -a "$SUM"
grep -q 'G-W3 OVERALL' "$OUT/drive.log" || echo "drive: NO OVERALL LINE (a failure or VOID of this job)" | tee -a "$SUM"
echo "finished: $(date '+%F %T %Z')" | tee -a "$OUT/provenance.txt"; exit 0
