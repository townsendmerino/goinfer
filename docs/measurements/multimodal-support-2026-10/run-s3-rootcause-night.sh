#!/usr/bin/env bash
# S3's root-cause night job (docs/tasks/task-multimodal-support-2026-10.md, "G-S3c root cause" and "G-S3c, re-registered").
# Everything here needs more free memory than the Mac has while the owner works: a built Metal resident beside the CPU
# arm, or the Gemma 3 4B in-process. Three steps, each independent; one failing does not stop the others.
#
#   A. G-S3c, re-registered: Qwen2.5-VL-3B and Gemma 3 4B, arms metal:cpu, cpu:cpu, cpu:cpu with --embed-int4=false on
#      every arm (the tower on the CPU everywhere; the second CPU run is the determinism control).
#   B. G-S3b's Gemma 3 controls: metal:cpu, metal:cpu, metal:auto, metal:auto at serve's --backend metal defaults (each
#      tower arm twice: does either arm repeat itself byte for byte?).
#   C. Gemma 3 phase 2: the int4 decoder teacher-forced on the CPU tower's, the Metal tower's and three noise-matched
#      feature sets (decoder/gemma3_tower_sensitivity_real_test.go).
#
# Pinned inputs, built by day (the tree may move before tonight): $BIN/serve-metal (s2-towers, rev in serve-metal.rev),
# $BIN/decoder-g3.test (main, rev in decoder-g3.rev), $BIN/g3-feats.json (metal/gemma3_tower_dump_test.go's output).
# Checkpoints from ~/models only. Estimate: A ~8 min, B ~6 min, C ~6 min; queued at 30.
#   python3 scripts/night.py add s3-rootcause --est 30 --by "Claude, S3 root cause" --doc docs/tasks/task-multimodal-support-2026-10.md -- bash docs/measurements/multimodal-support-2026-10/run-s3-rootcause-night.sh
set -uo pipefail
SRC=$(cd "$(dirname "$0")/../../.." && pwd)
BIN=${BIN:-$HOME/goinfer-bench/s3}
OUT=${1:-$HOME/goinfer-bench/s3/night-$(date +%F)}
for f in serve-metal decoder-g3.test g3-feats.json; do [ -e "$BIN/$f" ] || { echo "FATAL: $BIN/$f is missing"; exit 2; }; done
for m in qwen25vl-3b-instruct gemma-3-4b-it; do [ -d "$HOME/models/$m" ] || { echo "FATAL: ~/models/$m is missing"; exit 2; }; done
mkdir -p "$OUT"
{ echo "serve:    $BIN/serve-metal (rev $(cat "$BIN/serve-metal.rev" 2>/dev/null))"
  echo "decoder:  $BIN/decoder-g3.test (rev $(cat "$BIN/decoder-g3.rev" 2>/dev/null))"
  echo "started:  $(date '+%F %T %Z')"; sw_vers 2>/dev/null | tr '\n' ' '; echo; pmset -g batt | head -1; } | tee "$OUT/provenance.txt"
cd "$SRC" || exit 2
SERVED=docs/measurements/multimodal-support-2026-10/run-gs3c-served.sh
rc=0
echo "=== A. G-S3c re-registered $(date '+%T')"
GS3C_EXTRA="--embed-int4=false" bash "$SERVED" "$BIN/serve-metal" "$OUT/A-gs3c" metal:cpu,=cpu:cpu,cpu:cpu \
  "$HOME/models/qwen25vl-3b-instruct" "$HOME/models/gemma-3-4b-it" > "$OUT/A-gs3c.log" 2>&1 || rc=1
grep -E "decode path|IDENTICAL|differing|top-3|near-tie|exited" "$OUT/A-gs3c.log"
echo "=== B. G-S3b Gemma 3 controls $(date '+%T')"
bash "$SERVED" "$BIN/serve-metal" "$OUT/B-gs3b" =metal:cpu,metal:cpu,metal:auto,metal:auto \
  "$HOME/models/gemma-3-4b-it" > "$OUT/B-gs3b.log" 2>&1 || rc=1
grep -E "decode path|IDENTICAL|differing|top-3|near-tie|exited" "$OUT/B-gs3b.log"
echo "=== C. Gemma 3 phase 2 $(date '+%T')"
( cd "$SRC/decoder" && GOINFER_HEAVY_TESTS=1 GOINFER_G3_FEATS="$BIN/g3-feats.json" \
  "$BIN/decoder-g3.test" -test.run '^TestGemma3TowerSensitivity$' -test.v -test.timeout 30m ) > "$OUT/C-g3-sensitivity.log" 2>&1 || rc=1
grep -E "\[g3\]|^--- " "$OUT/C-g3-sensitivity.log"
grep -q -- "--- PASS: TestGemma3TowerSensitivity" "$OUT/C-g3-sensitivity.log" || { echo "!! phase 2 did not PASS (a skip or a failure)"; rc=1; }
echo "finished: $(date '+%F %T %Z') rc=$rc" | tee -a "$OUT/provenance.txt"
exit $rc
