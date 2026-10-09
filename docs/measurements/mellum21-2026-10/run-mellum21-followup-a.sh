#!/usr/bin/env bash
# Follow-up A: WHERE does Mellum2.1's window-golden cosine (0.98600, 2.0: 0.99636) come from? Per-layer residuals, goinfer int8int8 vs HF bf16, at positions
# before and after the 1024 window, one run each. Pre-registered classes W / F / N in scripts/diff_mellum_layers.py and docs/tasks/task-mellum21-2026-10.md.
#   1. dump   TestMellum2_layerDump: the SAME sequential int8int8 path as TestMellum2_windowParity (about 15 min), ForwardCapture at 9 positions
#   2. diff   scripts/diff_mellum_layers.py run: HF bf16 forward over the 1441 ids with a hook on every layer (about 5 min), rotary buffers checked, table + class
# A SKIP is a failure of this job. Estimate ~35 min expected, queued at 70 (window gate 883 s measured; HF pin 195 s for two forwards, measured).
#   python3 scripts/night.py add mellum21-fu-a --est 70 --by "nobara session, Mellum2.1 follow-up A" --doc docs/tasks/task-mellum21-2026-10.md -- bash docs/measurements/mellum21-2026-10/run-mellum21-followup-a.sh
# MELLUM_DRY=1 checks preconditions only.
source "$(dirname "$0")/_common.sh"
OUT=${1:-$HOME/goinfer-logs/mellum21-followup-a-$(date +%F)}
[ -x "$F/decoder.test" ] || fatal "$F/decoder.test missing"; [ -x "$PY" ] || fatal "$PY missing"
for f in "$CK/config.json" "$GOLD/mellum21_window_golden.json"; do [ -e "$f" ] || fatal "$f missing"; done
guard_paths; mkdir -p "$OUT"; SUM="$OUT/summary.txt"; : > "$SUM"; provenance
[ -n "${MELLUM_DRY:-}" ] && { echo "DRY: preconditions hold; plan = dump (goinfer, ~15 min), diff (HF + compare, ~6 min)"; exit 0; }
rm -f "$GOLD/mellum21_window_layers_goinfer.json.gz"
step dump gt TestMellum2_layerDump GOINFER_MELLUM_GOLDEN_PREFIX="$GOLD/mellum21"; grep -E "this run|wrote" "$OUT/dump.log" | tee -a "$SUM"; verdict dump TestMellum2_layerDump
[ -s "$GOLD/mellum21_window_layers_goinfer.json.gz" ] || { echo "RESULT: no dump written"; exit 1; }
step diff "$PY" -I "$SRC/scripts/diff_mellum_layers.py" run "$CK" "$GOLD/mellum21_window_golden.json" "$GOLD/mellum21_window_layers_goinfer.json.gz" "$OUT"
cp "$GOLD/mellum21_window_layers_goinfer.json.gz" "$OUT/" 2>/dev/null
grep -E "CLASS|rotary|inv_freq|d_short|relative L2" "$OUT/diff.log" | tee -a "$SUM"
grep -q '^CLASS ' "$OUT/diff.log" && echo "diff: classified" | tee -a "$SUM" || echo "diff: NO CLASS PRINTED (a failure of this job)" | tee -a "$SUM"
echo "finished: $(date '+%F %T %Z')" | tee -a "$OUT/provenance.txt"; exit 0
