#!/usr/bin/env bash
# Follow-up A3: A2's registered f32 arm, run with the weights held as bf16. A2 (2026-10-10) fell back to int8 weight-only because the fit guard declined f32
# (48.9 GB against a 40.4 GB budget). The Mellum2.1 checkpoint is all BF16 (5,631 tensors), the new Quant "bf16" keeps every matmul weight as the checkpoint's own
# bits (24.3 GB) and widens a tile of rows to f32 inside the matmul, so the forward is the f32 forward bit for bit (decoder.TestHalf16Weights_bitIdenticalToF32 on
# three fixtures; Qwen3-1.7B bf16 real checkpoint, 16 tokens sequential and batched, bit-identical). There is NO fallback: a failed load is "no reading". The rule is
# A2's, unchanged (docs/tasks/task-mellum21-2026-10.md, "A3"): scripts/diff_mellum_layers.py compare: Q / P / M / X. Estimate: dump about 35 min (1.37 s/token measured
# on 8 tokens), HF 2 min; queue at 60.
#   python3 scripts/night.py add mellum21-fu-a3 --est 60 --by "nobara session, Mellum2.1 follow-up A3" --doc docs/tasks/task-mellum21-2026-10.md -- bash docs/measurements/mellum21-2026-10/run-mellum21-followup-a3.sh
# MELLUM_DRY=1 checks preconditions only.
export F=${F:-$HOME/goinfer-bench/mellum21/a3}      # pinned decoder.test with Quant "bf16" (rev beside it names both local branches' commits)
source "$(dirname "$0")/_common.sh"
OUT=${1:-$HOME/goinfer-logs/mellum21-followup-a3-$(date +%F)}
BASEJ=${BASEJ:-$HOME/goinfer-logs/mellum21-followup-a-2026-10-09/layers_cos.json}   # follow-up A's int8int8 reading, the rule's baseline
[ -x "$F/decoder.test" ] || fatal "$F/decoder.test missing"; [ -x "$PY" ] || fatal "$PY missing"
for f in "$CK/config.json" "$GOLD/mellum21_window_golden.json" "$BASEJ"; do [ -e "$f" ] || fatal "$f missing"; done
guard_paths; mkdir -p "$OUT"; SUM="$OUT/summary.txt"; : > "$SUM"; provenance
[ -n "${MELLUM_DRY:-}" ] && { echo "DRY: preconditions hold; plan = dump bf16 (no fallback), diff, compare against $BASEJ"; exit 0; }
DUMP="$GOLD/mellum21_window_layers_goinfer_bf16.json.gz"; rm -f "$DUMP"
step dump-bf16 gt TestMellum2_layerDump GOINFER_MELLUM_GOLDEN_PREFIX="$GOLD/mellum21" GOINFER_MELLUM_DUMP_QUANT=bf16; verdict dump-bf16 TestMellum2_layerDump
[ -s "$DUMP" ] || { echo "RESULT: no bf16 dump written (no reading; there is no fallback in A3)" | tee -a "$SUM"; exit 1; }
echo "ARM: bf16 weights, f32 activations (the f32 forward)" | tee -a "$SUM"; grep -E "this run|wrote|differs" "$OUT/dump-bf16.log" | tee -a "$SUM"
cp "$DUMP" "$OUT/" 2>/dev/null
step diff "$PY" -I "$SRC/scripts/diff_mellum_layers.py" run "$CK" "$GOLD/mellum21_window_golden.json" "$DUMP" "$OUT/arm-bf16"
grep -E "CLASS|rotary|inv_freq|d_short|relative L2" "$OUT/diff.log" | tee -a "$SUM"
"$PY" -I "$SRC/scripts/diff_mellum_layers.py" compare "$BASEJ" "$OUT/arm-bf16/layers_cos.json" 2>&1 | tee "$OUT/compare.log" | tee -a "$SUM"
grep -q '^CLASS ' "$OUT/compare.log" && echo "compare: classified (bf16 arm)" | tee -a "$SUM" || echo "compare: NO CLASS PRINTED (a failure of this job)" | tee -a "$SUM"
echo "finished: $(date '+%F %T %Z')" | tee -a "$OUT/provenance.txt"; exit 0
