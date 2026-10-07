#!/usr/bin/env bash
# S2's speed record (docs/tasks/task-multimodal-support-2026-10.md): tower time per image, Metal against the CPU, for the
# Qwen3.5+ and GLM-OCR towers, on the real checkpoints (~/models/qwen3.5-0.8b, ~/models/glm-ocr) and the G-S2b images.
# A record, not a gate. TestGridVisionMetal_real times each image once per tower per pass (CPU, then Metal); three
# passes, so each figure has three readings. Under the timing lock.
# Estimate: ~3.7 min a pass (220 s measured by day), three passes ~11 min; queued at 20.
#
# The test binary is pre-built from the s2-towers branch (BIN/rev names its commit; it needs aikit's unreleased
# s2-tower-exports, so it cannot be built from main):
#   (cd ../goinfer-s2/metal && GOWORK=$PWD/../go.work go test -c -tags goinfer_testhooks -o $BIN/metal.test .)
#   python3 scripts/night.py add s2-tower-speed --est 20 --by "Claude, S2" --doc docs/tasks/task-multimodal-support-2026-10.md -- bash docs/measurements/multimodal-support-2026-10/run-s2-tower-speed.sh
set -uo pipefail
SRC=$(cd "$(dirname "$0")/../../.." && pwd)
BIN=${BIN:-$HOME/goinfer-bench/s2}
OUT=${1:-$HOME/goinfer-bench/s2/run-$(date +%F)}
[ -x "$BIN/metal.test" ] || { echo "FATAL: $BIN/metal.test is missing"; exit 2; }
for m in qwen3.5-0.8b glm-ocr; do [ -d "$HOME/models/$m" ] || { echo "FATAL: ~/models/$m is missing"; exit 2; }; done
mkdir -p "$OUT"
{ echo "binary:   $BIN/metal.test (rev $(cat "$BIN/rev" 2>/dev/null || echo unknown))"; echo "started:  $(date '+%F %T %Z')"
  sw_vers 2>/dev/null | tr '\n' ' '; echo; pmset -g batt | head -1; } | tee "$OUT/provenance.txt"
cd "$SRC/metal" || exit 2
python3 "$SRC/scripts/timing_lock.py" run --label s2-tower-speed -- \
  env GOINFER_HEAVY_TESTS=1 "$BIN/metal.test" -test.run '^TestGridVisionMetal_real$' -test.v -test.count 3 -test.timeout 40m \
  > "$OUT/tower-speed.log" 2>&1
rc=$?
grep -E "S2 real|^--- " "$OUT/tower-speed.log"
grep -q -- "--- PASS: TestGridVisionMetal_real" "$OUT/tower-speed.log" || { echo "!! the tower test did not PASS (a skip or a failure)"; rc=1; }
echo "finished: $(date '+%F %T %Z') rc=$rc" | tee -a "$OUT/provenance.txt"
exit $rc
