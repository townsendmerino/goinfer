#!/usr/bin/env bash
# G-W3b pass 2 (docs/completed/task-cuda-windowed-kv-2026-10.md, section 12, pre-registered before it ran): the order-reversed ABBA of G-W3, sessions W F F W at ctx 2048 on the SAME pinned binary
# as pass 1 (rev 4521a62a), no ctx-4096 sessions, then the pooled verdict over both passes (windowed_kv_pool.py). Timed (night.py holds the timing lock). Estimate ~6 min
# (4 loads of ~25 s, 6 short + 6 long requests per session); queue at 15.
#   python3 scripts/night.py add windowed-kv-abba2 --est 15 --by "nobara session, CUDA windowed KV default-on" --doc docs/completed/task-cuda-windowed-kv-2026-10.md -- bash docs/measurements/cuda-windowed-kv-2026-10/run-windowed-kv-abba2.sh
F=${F:-$HOME/goinfer-bench/windowed-kv}
source "$(dirname "$0")/../mellum21-2026-10/_common.sh"
OUT=${1:-$HOME/goinfer-logs/windowed-kv-abba2-$(date +%F)}
PASS1=${PASS1:-$HOME/goinfer-logs/windowed-kv-gates-2026-10-10/rows.json}
[ -x "$F/serve-cuda" ] || fatal "$F/serve-cuda missing"; [ -e "$CK/config.json" ] || fatal "$CK missing"; [ -e "$PASS1" ] || fatal "$PASS1 (pass 1's rows) missing"
[ "$(cat "$F/rev")" = "4521a62a21aa4267e1368bf9848354d643eeafb5" ] || fatal "the pinned binary is not pass 1's (rev $(cat "$F/rev"))"
ls "$CK".int4*.giw >/dev/null 2>&1 || fatal "the int4mix .giw cache next to $CK is missing"
guard_paths; mkdir -p "$OUT"; SUM="$OUT/summary.txt"; : > "$SUM"; provenance
curl -fs http://127.0.0.1:18932/v1/models >/dev/null 2>&1 && fatal "something already serves on 18932"
[ -n "${MELLUM_DRY:-}" ] && { echo "DRY: preconditions hold; plan = W F F W sessions at ctx 2048 (3 short x 3 reps, 1 long x 3 reps), pooled with $PASS1"; exit 0; }
step drive python3 "$SRC/docs/measurements/cuda-windowed-kv-2026-10/windowed_kv_gate.py" "$F/serve-cuda" "$CK" "$OUT" --order=WFFW --no-load
grep -E "prompt [0-9]:|ratio|session medians|VRAM|G-W|VOID" "$OUT/drive.log" | tee -a "$SUM"
grep -q 'G-W3 OVERALL' "$OUT/drive.log" || echo "drive: NO OVERALL LINE (a failure or VOID of this job)" | tee -a "$SUM"
python3 "$SRC/docs/measurements/cuda-windowed-kv-2026-10/windowed_kv_pool.py" "$PASS1" "$OUT/rows.json" 2>&1 | tee "$OUT/pooled.txt" | tee -a "$SUM"
grep -q '^G-W3b ' "$OUT/pooled.txt" || echo "pool: NO VERDICT LINE (a failure of this job)" | tee -a "$SUM"
echo "finished: $(date '+%F %T %Z')" | tee -a "$OUT/provenance.txt"; exit 0
