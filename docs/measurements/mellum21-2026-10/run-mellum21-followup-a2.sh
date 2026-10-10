#!/usr/bin/env bash
# Follow-up A2: is the follow-up A crater (positions 1030 and 1300 of the window golden) the window path, or int8 quantization sensitivity?
# The same per-layer dump as A (TestMellum2_layerDump, the same sequential path) with goinfer in f32 (GOINFER_MELLUM_DUMP_QUANT empty), read against the same
# HF bf16 reference by the same script; if the f32 load is declined by the fit guard (12B x 4 bytes is about 48 GB of 62), the registered fallback arm is
# int8 weight-only (f32 activations). The arm that ran is named in the result. Rule (pre-registered in docs/tasks/task-mellum21-2026-10.md, "A2"):
# scripts/diff_mellum_layers.py compare: Q / P / M / X. A SKIP is a failure of this job. Estimate: dump about 25-40 min, HF about 2 min, queue at 90.
#   python3 scripts/night.py add mellum21-fu-a2 --est 90 --by "nobara session, Mellum2.1 follow-up A2" --doc docs/tasks/task-mellum21-2026-10.md -- bash docs/measurements/mellum21-2026-10/run-mellum21-followup-a2.sh
# MELLUM_DRY=1 checks preconditions only.
export F=${F:-$HOME/goinfer-bench/mellum21/a2}      # pinned decoder.test of the commit that adds GOINFER_MELLUM_DUMP_QUANT, and rev
source "$(dirname "$0")/_common.sh"
OUT=${1:-$HOME/goinfer-logs/mellum21-followup-a2-$(date +%F)}
BASEJ=${BASEJ:-$HOME/goinfer-logs/mellum21-followup-a-2026-10-09/layers_cos.json}   # follow-up A's int8int8 reading, the rule's baseline
[ -x "$F/decoder.test" ] || fatal "$F/decoder.test missing"; [ -x "$PY" ] || fatal "$PY missing"
for f in "$CK/config.json" "$GOLD/mellum21_window_golden.json" "$BASEJ"; do [ -e "$f" ] || fatal "$f missing"; done
guard_paths; mkdir -p "$OUT"; SUM="$OUT/summary.txt"; : > "$SUM"; provenance
[ -n "${MELLUM_DRY:-}" ] && { echo "DRY: preconditions hold; plan = dump f32 (else int8), diff, compare against $BASEJ"; exit 0; }
ARM=f32; DUMP="$GOLD/mellum21_window_layers_goinfer_f32.json.gz"; rm -f "$DUMP"
step dump-f32 gt TestMellum2_layerDump GOINFER_MELLUM_GOLDEN_PREFIX="$GOLD/mellum21" GOINFER_MELLUM_DUMP_QUANT=; verdict dump-f32 TestMellum2_layerDump
if [ ! -s "$DUMP" ]; then
  echo "f32 arm produced no dump (declined or failed): $(grep -E 'fit|declin|refus|memory|FAIL' "$OUT/dump-f32.log" | head -3 | cut -c1-200)" | tee -a "$SUM"
  ARM=int8; DUMP="$GOLD/mellum21_window_layers_goinfer_int8.json.gz"; rm -f "$DUMP"
  step dump-int8 gt TestMellum2_layerDump GOINFER_MELLUM_GOLDEN_PREFIX="$GOLD/mellum21" GOINFER_MELLUM_DUMP_QUANT=int8; verdict dump-int8 TestMellum2_layerDump
fi
[ -s "$DUMP" ] || { echo "RESULT: no dump written by either arm"; exit 1; }
echo "ARM: $ARM" | tee -a "$SUM"; grep -E "this run|wrote|differs" "$OUT/dump-$ARM.log" | tee -a "$SUM"
cp "$DUMP" "$OUT/" 2>/dev/null
step diff "$PY" -I "$SRC/scripts/diff_mellum_layers.py" run "$CK" "$GOLD/mellum21_window_golden.json" "$DUMP" "$OUT/arm-$ARM"
grep -E "CLASS|rotary|inv_freq|d_short|relative L2" "$OUT/diff.log" | tee -a "$SUM"
"$PY" -I "$SRC/scripts/diff_mellum_layers.py" compare "$BASEJ" "$OUT/arm-$ARM/layers_cos.json" 2>&1 | tee "$OUT/compare.log" | tee -a "$SUM"
grep -q '^CLASS ' "$OUT/compare.log" && echo "compare: classified ($ARM arm)" | tee -a "$SUM" || echo "compare: NO CLASS PRINTED (a failure of this job)" | tee -a "$SUM"
echo "finished: $(date '+%F %T %Z')" | tee -a "$OUT/provenance.txt"; exit 0
