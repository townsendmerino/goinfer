#!/usr/bin/env bash
# G-S12.3g (docs/tasks/task-multimodal-support-2026-10.md, "S12.3g"): the patch count above which each grid tower's
# WebGPU form is faster than its CPU form, by a sweep over image sizes (TestGridTowersWebGPU_patchSweep), for the
# Qwen3.5 0.8B tower, GLM-OCR's and Qwen2.5-VL 3B's. Runs on either box; night.py holds the timing lock.
#
# Pinned: $BIN/gpu-3g.test, built from the rev in $BIN/gpu-3g.rev (go test -c -tags 'gpu goinfer_testhooks' in gpu/).
# It runs from a gpu directory because it reads ../testdata: GPUDIR, default the gpu directory of this script's checkout.
#   BIN=~/goinfer-bench/s12-3g-2026-10-12/bin bash <this script>
set -uo pipefail
SRC=$(cd "$(dirname "$0")/../../../.." && pwd)
BIN=${BIN:-$HOME/goinfer-bench/s12-3g-2026-10-12/bin}
GPUDIR=${GPUDIR:-$SRC/gpu}
OUT=${OUT:-$HOME/goinfer-bench/s12-3g-2026-10-12/out}
[ -x "$BIN/gpu-3g.test" ] || { echo "FATAL: $BIN/gpu-3g.test is missing"; exit 2; }
for d in qwen3.5-0.8b glm-ocr qwen25vl-3b-instruct; do
  [ -d "$HOME/models/$d" ] || { echo "FATAL: ~/models/$d is missing"; exit 2; }
done
[ -f "$GPUDIR/../testdata/glm_ocr/table.png" ] || { echo "FATAL: the test image is missing under $GPUDIR/../testdata"; exit 2; }
mkdir -p "$OUT"
{ echo "binary $BIN/gpu-3g.test rev $(cat "$BIN/gpu-3g.rev" 2>/dev/null)"; date '+started %F %T %Z'; uname -srm
  nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null; uptime; } | tee "$OUT/provenance.txt"
rc=0
for tower in qwen3 glm-ocr qwen25; do
  ( cd "$GPUDIR" && GOINFER_TOWER_PATCH_SWEEP=1 GOINFER_GRID_SPEED_TOWER=$tower "$BIN/gpu-3g.test" -test.run '^TestGridTowersWebGPU_patchSweep$' -test.v -test.timeout 40m ) > "$OUT/g-s12-3g-$tower.log" 2>&1
  grep -E 'G-S12.3g|^--- ' "$OUT/g-s12-3g-$tower.log"
  # A skip is not a pass: the sweep must have run and printed its threshold line.
  grep -qE "G-S12.3g $tower .*THRESHOLD" "$OUT/g-s12-3g-$tower.log" || { echo "!! no threshold line for $tower: the sweep did not finish"; rc=1; }
  date "+$tower done %T"
done
echo "finished $(date '+%F %T %Z') rc=$rc" | tee -a "$OUT/provenance.txt"
exit $rc
