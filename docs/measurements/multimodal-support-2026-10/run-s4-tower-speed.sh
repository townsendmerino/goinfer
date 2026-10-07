#!/usr/bin/env bash
# S4's speed record (docs/tasks/task-multimodal-support-2026-10.md): tower time per image, CUDA against the CPU, for every CUDA tower that passed its gates: Gemma 4 E2B
# (TestG4VCUDA_gemma4E2B), Qwen3.5-0.8B and GLM-OCR (TestGridVisionCUDA_real), and Qwen2.5-VL-3B on aikit's gpu/qwencuda (TestS4Towers_real), on the real checkpoints under ~/models
# and the F2a/G-S2b images. A record, not a gate. Each test times every image once per tower per pass (CPU, then CUDA; the CUDA figure includes aikit's host tail); COUNT passes
# (default 3), so each figure has COUNT readings. Under the timing lock.
# Estimate: ~7 min a pass (28 s Gemma 4, 273 s grid towers, ~100 s Qwen2.5-VL, measured by day), three passes ~21 min; queued at 30.
#
# The test binary is pre-built from the s2-towers branch (BIN/rev names its commit; it needs aikit's unreleased s2-tower-exports, so it cannot be built from main):
#   (cd ~/wt/goinfer-s2/cuda && GOWORK=$HOME/wt/goinfer-s2/go.work go test -c -tags 'cuda goinfer_testhooks' -o $BIN/cuda.test . && git -C .. rev-parse --short HEAD > $BIN/rev)
#   python3 scripts/night.py add s4-tower-speed --est 30 --by "Claude, S4" --doc docs/tasks/task-multimodal-support-2026-10.md -- bash docs/measurements/multimodal-support-2026-10/run-s4-tower-speed.sh
set -uo pipefail
SRC=$(cd "$(dirname "$0")/../../.." && pwd)
BIN=${BIN:-$HOME/goinfer-bench/s4}
OUT=${1:-$HOME/goinfer-bench/s4/run-$(date +%F)}
COUNT=${COUNT:-3}
[ -x "$BIN/cuda.test" ] || { echo "FATAL: $BIN/cuda.test is missing"; exit 2; }
for m in gemma-4-E2B-unq qwen3.5-0.8b glm-ocr qwen25vl-3b-instruct; do [ -d "$HOME/models/$m" ] || { echo "FATAL: ~/models/$m is missing"; exit 2; }; done
mkdir -p "$OUT"
{ echo "binary:   $BIN/cuda.test (rev $(cat "$BIN/rev" 2>/dev/null || echo unknown))"; echo "started:  $(date '+%F %T %Z')"; echo "passes:   $COUNT"
  echo "gpu:      $(nvidia-smi --query-gpu=name,driver_version,memory.used --format=csv,noheader)"; echo "load:     $(cat /proc/loadavg)"; } | tee "$OUT/provenance.txt"
# the test binary reads testdata/ relative to the package directory
cd "$SRC/cuda" || exit 2
python3 "$SRC/scripts/timing_lock.py" run --label s4-tower-speed -- \
  env GOINFER_HEAVY_TESTS=1 "$BIN/cuda.test" -test.run '^(TestG4VCUDA_gemma4E2B|TestGridVisionCUDA_real|TestS4Towers_real)$' -test.v -test.count "$COUNT" -test.timeout 90m \
  > "$OUT/tower-speed.log" 2>&1
rc=$?
grep -E "worst soft-token|\[S4 real|\[S4 G-S4q\]|^--- " "$OUT/tower-speed.log" | cut -c1-200
for t in TestG4VCUDA_gemma4E2B TestGridVisionCUDA_real TestS4Towers_real; do
  n=$(grep -c -- "^--- PASS: $t" "$OUT/tower-speed.log")
  [ "$n" -eq "$COUNT" ] || { echo "!! $t passed $n of $COUNT passes (a skip or a failure)"; rc=1; }
done
echo "finished: $(date '+%F %T %Z') rc=$rc" | tee -a "$OUT/provenance.txt"
exit $rc
