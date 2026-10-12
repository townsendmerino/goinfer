#!/usr/bin/env bash
# G-S12.3d-4 (docs/tasks/task-multimodal-support-2026-10.md, S12.3d): Gemma 4 E2B's vision tower on one image at the
# checkpoint's image budget on the Mac, the WebGPU tower's encode time over the CPU tower's, five interleaved pairs.
# The verdict line decides whether the device tower becomes --backend webgpu's default for Gemma 4. night.py holds the
# timing lock for the job.
#
# Pinned: $BIN/gpu-3d4.test, built from the rev in $BIN/rev-3d4 (go test -c -tags 'gpu goinfer_testhooks' ./gpu).
# It runs from the repo's gpu directory because it reads ../testdata/glm_ocr/table.png. Estimate: 2 minutes.
#   python3 scripts/night.py add s12-3d-4-webgpu-gemma4-tower-speed --est 5 --by "Claude (Mac), S12.3d" --doc docs/tasks/task-multimodal-support-2026-10.md -- bash docs/measurements/multimodal-support-2026-10/s12-webgpu/run-s12-3d-4-night.sh
set -uo pipefail
SRC=$(cd "$(dirname "$0")/../../../.." && pwd)
BIN=${BIN:-$HOME/goinfer-bench/s12-webgpu-2026-10-11/bin}
OUT=$HOME/goinfer-logs/s12-webgpu/3d-4-$(date +%F)
[ -x "$BIN/gpu-3d4.test" ] || { echo "FATAL: $BIN/gpu-3d4.test is missing"; exit 2; }
[ -d "$HOME/models/gemma-4-E2B-unq" ] || { echo "FATAL: ~/models/gemma-4-E2B-unq is missing"; exit 2; }
[ -f "$SRC/testdata/glm_ocr/table.png" ] || { echo "FATAL: the test image is missing"; exit 2; }
mkdir -p "$OUT"
{ echo "binary $BIN/gpu-3d4.test rev $(cat "$BIN/rev-3d4" 2>/dev/null)"; date '+started %F %T %Z'; sw_vers | tr '\n' ' '; echo; pmset -g batt | head -1; uptime; } | tee "$OUT/provenance.txt"
( cd "$SRC/gpu" && GOINFER_TOWER_SPEED_GATE=1 "$BIN/gpu-3d4.test" -test.run '^TestG4VWebGPU_speedGate$' -test.v -test.timeout 20m ) > "$OUT/g-s12-3d-4.log" 2>&1
rc=$?
grep -E 'G-S12.3d-4|^--- ' "$OUT/g-s12-3d-4.log"
# A skip is not a pass: the gate must have run and printed a verdict.
grep -qE 'G-S12.3d-4 (SHIP|PARKED|OFF)' "$OUT/g-s12-3d-4.log" || { echo "!! no verdict line: the gate did not run"; rc=1; }
echo "finished $(date '+%F %T %Z') rc=$rc" | tee -a "$OUT/provenance.txt"
exit $rc
