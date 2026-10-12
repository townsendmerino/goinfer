#!/usr/bin/env bash
# G-S12.3f-4 (docs/tasks/task-multimodal-support-2026-10.md, S12.3f): Qwen2.5-VL 3B's vision tower on one image
# (testdata/glm_ocr/table.png, through serve's preprocessing) on the Mac, the WebGPU tower's encode time over the CPU
# tower's, five interleaved pairs. The verdict line decides whether the device tower becomes --backend webgpu's default
# for Qwen2.5-VL. night.py holds the timing lock for the job.
#
# Pinned: $BIN/gpu-3f4.test, built from the rev in $BIN/rev-3f4 (go test -c -tags 'gpu goinfer_testhooks' ./gpu).
# It runs from the repo's gpu directory because it reads ../testdata. Estimate: 6 minutes (the CPU tower is about 31 s
# on this image).
#   python3 scripts/night.py add s12-3f-4-webgpu-qwen25vl-tower-speed --est 9 --by "Claude (Mac), S12.3f" --doc docs/tasks/task-multimodal-support-2026-10.md -- bash docs/measurements/multimodal-support-2026-10/s12-webgpu/run-s12-3f-4-night.sh
set -uo pipefail
SRC=$(cd "$(dirname "$0")/../../../.." && pwd)
BIN=${BIN:-$HOME/goinfer-bench/s12-webgpu-2026-10-11/bin}
OUT=$HOME/goinfer-logs/s12-webgpu/3f-4-$(date +%F)
[ -x "$BIN/gpu-3f4.test" ] || { echo "FATAL: $BIN/gpu-3f4.test is missing"; exit 2; }
[ -d "$HOME/models/qwen25vl-3b-instruct" ] || { echo "FATAL: ~/models/qwen25vl-3b-instruct is missing"; exit 2; }
[ -f "$SRC/testdata/glm_ocr/table.png" ] || { echo "FATAL: the test image is missing"; exit 2; }
mkdir -p "$OUT"
{ echo "binary $BIN/gpu-3f4.test rev $(cat "$BIN/rev-3f4" 2>/dev/null)"; date '+started %F %T %Z'; sw_vers | tr '\n' ' '; echo; pmset -g batt | head -1; uptime; } | tee "$OUT/provenance.txt"
( cd "$SRC/gpu" && GOINFER_TOWER_SPEED_GATE=1 "$BIN/gpu-3f4.test" -test.run '^TestQwen25VisionWebGPU_speedGate$' -test.v -test.timeout 25m ) > "$OUT/g-s12-3f-4.log" 2>&1
rc=$?
grep -E 'G-S12.3f-4|^--- ' "$OUT/g-s12-3f-4.log"
# A skip is not a pass: the gate must have run and printed a verdict.
grep -qE 'G-S12.3f-4 (SHIP|PARKED|OFF)' "$OUT/g-s12-3f-4.log" || { echo "!! no verdict line: the gate did not run"; rc=1; }
echo "finished $(date '+%F %T %Z') rc=$rc" | tee -a "$OUT/provenance.txt"
exit $rc
