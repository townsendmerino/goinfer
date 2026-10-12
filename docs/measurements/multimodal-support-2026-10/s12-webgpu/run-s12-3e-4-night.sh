#!/usr/bin/env bash
# G-S12.3e-4 (docs/tasks/task-multimodal-support-2026-10.md, S12.3e): the Qwen3.5 0.8B tower and the GLM-OCR tower on
# one image (testdata/glm_ocr/table.png, through serve's preprocessing) on the Mac, the WebGPU tower's encode time over
# the CPU tower's, five interleaved pairs each. Each tower's verdict line decides whether its device tower becomes
# --backend webgpu's default. night.py holds the timing lock for the job.
#
# Pinned: $BIN/gpu-3e4.test, built from the rev in $BIN/rev-3e4 (go test -c -tags 'gpu goinfer_testhooks' ./gpu).
# It runs from the repo's gpu directory because it reads ../testdata. Estimate: 8 minutes (GLM-OCR's CPU tower is about
# 40 s on this image).
#   python3 scripts/night.py add s12-3e-4-webgpu-grid-tower-speed --est 12 --by "Claude (Mac), S12.3e" --doc docs/tasks/task-multimodal-support-2026-10.md -- bash docs/measurements/multimodal-support-2026-10/s12-webgpu/run-s12-3e-4-night.sh
set -uo pipefail
SRC=$(cd "$(dirname "$0")/../../../.." && pwd)
BIN=${BIN:-$HOME/goinfer-bench/s12-webgpu-2026-10-11/bin}
OUT=$HOME/goinfer-logs/s12-webgpu/3e-4-$(date +%F)
[ -x "$BIN/gpu-3e4.test" ] || { echo "FATAL: $BIN/gpu-3e4.test is missing"; exit 2; }
for d in qwen3.5-0.8b glm-ocr; do
  [ -d "$HOME/models/$d" ] || { echo "FATAL: ~/models/$d is missing"; exit 2; }
done
[ -f "$SRC/testdata/glm_ocr/table.png" ] || { echo "FATAL: the test image is missing"; exit 2; }
mkdir -p "$OUT"
{ echo "binary $BIN/gpu-3e4.test rev $(cat "$BIN/rev-3e4" 2>/dev/null)"; date '+started %F %T %Z'; sw_vers | tr '\n' ' '; echo; pmset -g batt | head -1; uptime; } | tee "$OUT/provenance.txt"
rc=0
for tower in qwen3 glm-ocr; do
  ( cd "$SRC/gpu" && GOINFER_TOWER_SPEED_GATE=1 GOINFER_GRID_SPEED_TOWER=$tower "$BIN/gpu-3e4.test" -test.run '^TestGridVisionWebGPU_speedGate$' -test.v -test.timeout 25m ) > "$OUT/g-s12-3e-4-$tower.log" 2>&1 || rc=1
  grep -E 'G-S12.3e-4|^--- ' "$OUT/g-s12-3e-4-$tower.log"
  # A skip is not a pass: the gate must have run and printed a verdict.
  grep -qE "G-S12.3e-4 $tower (SHIP|PARKED|OFF)" "$OUT/g-s12-3e-4-$tower.log" || { echo "!! no verdict line for $tower: the gate did not run"; rc=1; }
  date "+$tower done %T"
done
echo "finished $(date '+%F %T %Z') rc=$rc" | tee -a "$OUT/provenance.txt"
exit $rc
