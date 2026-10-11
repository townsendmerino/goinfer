#!/usr/bin/env bash
# I2 (docs/tasks/task-mellum21-2026-10.md, "The third crater, investigated"): the same goinfer dump A3 wrote (the f32 forward), read against HF in FLOAT32 instead of bf16.
# The f32 reference needs about 49 GB resident, so the HF run is wrapped in a user systemd scope with MemoryMax=56G: an overrun kills that one process and nothing else,
# and the job refuses to start with under 54 GB available. No fit-guard is involved (it is a Python/transformers run). Rule: diff_mellum_layers.py i2 (G / D / M / X).
# Estimate: load+convert ~3 min, forward ~5 min, compare seconds; queue at 40. Night tier: it holds 49 GB of a 62 GB box.
#   python3 scripts/night.py add mellum21-i2 --est 40 --by "nobara session, Mellum2.1 third crater" --doc docs/tasks/task-mellum21-2026-10.md -- bash docs/measurements/mellum21-2026-10/run-mellum21-i2.sh
source "$(dirname "$0")/_common.sh"
OUT=${1:-$HOME/goinfer-logs/mellum21-i2-$(date +%F)}
DUMP=${DUMP:-$GOLD/mellum21_window_layers_goinfer_bf16.json.gz}   # A3's dump
BF16J=${BF16J:-$HOME/goinfer-logs/mellum21-followup-a3-2026-10-10/arm-bf16/layers_cos.json}   # the same dump against the bf16 reference
[ -x "$PY" ] || fatal "$PY missing"
for f in "$CK/config.json" "$GOLD/mellum21_window_golden.json" "$DUMP" "$BF16J"; do [ -e "$f" ] || fatal "$f missing"; done
guard_paths; mkdir -p "$OUT"; SUM="$OUT/summary.txt"; : > "$SUM"; provenance
avail=$(awk '/MemAvailable/ {printf "%d", $2/1048576}' /proc/meminfo); echo "MemAvailable ${avail} GB (the f32 reference needs about 49)" | tee -a "$SUM"
[ "$avail" -ge 54 ] || fatal "only ${avail} GB available; the f32 HF reference needs about 49 GB resident"
systemd-run --user --scope -p MemoryMax=56G true >/dev/null 2>&1 || fatal "systemd-run --user --scope with MemoryMax is not available: no memory cap, so no run"
python3 "$SRC/scripts/diff_mellum_layers.py" selftest-i2 || fatal "the i2 classifier's self-test failed"
[ -n "${MELLUM_DRY:-}" ] && { echo "DRY: preconditions hold; plan = HF float32 forward under MemoryMax=56G, i2 against $BF16J"; exit 0; }
step hf-f32 systemd-run --user --scope -q -p MemoryMax=56G "$PY" -I "$SRC/scripts/diff_mellum_layers.py" run "$CK" "$GOLD/mellum21_window_golden.json" "$DUMP" "$OUT/arm-f32ref" float32
grep -E "HF reference dtype|CLASS|relative L2" "$OUT/hf-f32.log" | tee -a "$SUM"
[ -s "$OUT/arm-f32ref/layers_cos.json" ] || { echo "RESULT: no f32-reference table written (the HF run failed or was killed at the memory cap): no reading" | tee -a "$SUM"; exit 1; }
python3 "$SRC/scripts/diff_mellum_layers.py" i2 "$BF16J" "$OUT/arm-f32ref/layers_cos.json" 2>&1 | tee "$OUT/i2.log" | tee -a "$SUM"
grep -q '^CLASS ' "$OUT/i2.log" || echo "i2: NO CLASS PRINTED (a failure of this job)" | tee -a "$SUM"
echo "finished: $(date '+%F %T %Z')" | tee -a "$OUT/provenance.txt"; exit 0
