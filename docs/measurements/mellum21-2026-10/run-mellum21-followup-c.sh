#!/usr/bin/env bash
# Follow-up C: what is the PRIZE for a windowed-KV plan on CUDA? Mellum2.1 int4mix on the 8 GB card is resident only at ctx <= 2048 (it declines at 3072 and
# 4096: the plan prices KV for all 28 layers although 21 are windowed to 1024). At 16384 it runs through expert streaming (C'). This job times decode of the SAME
# model, ctx 2048, fully resident against C' (45 of 64 slots), ABBA by session, 3 prompts x 3 reps, greedy. It sizes the lever; it does not build it.
# Pre-registered reading in resident_vs_cprime.py's docstring and the task doc: ratio >= 1.15 worth building, < 1.05 park, between ambiguous -> parked.
# Timed (night.py holds the timing lock). Estimate ~12 min expected (4 loads of ~25 s + 36 requests), queued at 30.
#   python3 scripts/night.py add mellum21-fu-c --est 30 --by "nobara session, Mellum2.1 follow-up C" --doc docs/tasks/task-mellum21-2026-10.md -- bash docs/measurements/mellum21-2026-10/run-mellum21-followup-c.sh
source "$(dirname "$0")/_common.sh"
OUT=${1:-$HOME/goinfer-logs/mellum21-followup-c-$(date +%F)}
[ -x "$F/serve-cuda" ] || fatal "$F/serve-cuda missing"; [ -e "$CK/config.json" ] || fatal "$CK missing"
ls "$CK".int4*.giw >/dev/null 2>&1 || fatal "the int4mix .giw cache next to $CK is missing (run a serve once, as Gate 2 did)"
guard_paths; mkdir -p "$OUT"; SUM="$OUT/summary.txt"; : > "$SUM"; provenance
curl -fs http://127.0.0.1:18931/v1/models >/dev/null 2>&1 && fatal "something already serves on 18931"
[ -n "${MELLUM_DRY:-}" ] && { echo "DRY: preconditions hold; plan = R S S R sessions, 3 prompts x 3 reps"; exit 0; }
step drive python3 "$SRC/docs/measurements/mellum21-2026-10/resident_vs_cprime.py" "$F/serve-cuda" "$CK" "$OUT"
grep -E "prompt [0-9]:|session medians|VRAM|identical|OVERALL|VOID" "$OUT/drive.log" | tee -a "$SUM"
grep -q 'OVERALL' "$OUT/drive.log" || echo "drive: NO OVERALL LINE (a failure or VOID of this job)" | tee -a "$SUM"
echo "finished: $(date '+%F %T %Z')" | tee -a "$OUT/provenance.txt"; exit 0
