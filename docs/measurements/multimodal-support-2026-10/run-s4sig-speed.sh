#!/usr/bin/env bash
# The S4 addendum's speed record (docs/tasks/task-multimodal-support-2026-10.md, "S4 addendum: a float32 SigLIP tower on the CUDA tower base"): Gemma 3's SigLIP tower time per
# image, the float32 CUDA tower against the int8 CUDA tower against the CPU float32 tower, interleaved per round on two of the F2a images, under the timing lock. A record, no bar.
# Pre-built test binary (BIN/cuda.test from the commit in BIN/rev), so tonight's checkout is not what runs:
#   (cd cuda && CGO_ENABLED=0 go test -c -tags cuda -o $BIN/cuda.test .) ; git rev-parse --short HEAD > $BIN/rev
# Estimate ~10 min (3 rounds x 2 images x (CPU ~21 s + CUDA f32 ~18 s + CUDA int8 ~4 s), plus two real-tower loads).
# Queue:  python3 scripts/night.py add s4sig-speed --est 12 --by "nobara session, S4 addendum" --doc docs/tasks/task-multimodal-support-2026-10.md -- bash docs/measurements/multimodal-support-2026-10/run-s4sig-speed.sh
set -uo pipefail
SRC=$(cd "$(dirname "$0")/../../.." && pwd)
BIN=${BIN:-$HOME/goinfer-bench/s4sig}
OUT=${1:-$HOME/goinfer-logs/s4sig/speed-$(date +%F-%H%M)}
for p in "$BIN/cuda.test" "$HOME/models/gemma-3-4b-it/config.json"; do [ -e "$p" ] || { echo "FATAL: $p is missing" >&2; exit 2; }; done
mkdir -p "$OUT"
{ echo "binary: $BIN/cuda.test (rev $(cat "$BIN/rev" 2>/dev/null || echo unknown))"; echo "rounds: ${SIGLIP_SPEED_ROUNDS:-3}"; echo "started: $(date '+%F %T %Z')"
  echo "gpu: $(nvidia-smi --query-gpu=name,driver_version,memory.used --format=csv,noheader)"; echo "load: $(cat /proc/loadavg)"; } | tee "$OUT/provenance.txt"
cd "$SRC/cuda" || exit 2
python3 "$SRC/scripts/timing_lock.py" run --label s4sig-speed -- \
  env GOINFER_HEAVY_TESTS=1 "$BIN/cuda.test" -test.run '^TestSiglipCUDA_speed$' -test.v -test.timeout 25m > "$OUT/speed.log" 2>&1
rc=$?
grep -E "^(--- |PASS|FAIL)|siglip speed" "$OUT/speed.log"
grep -q -- "--- PASS: TestSiglipCUDA_speed" "$OUT/speed.log" || { echo "!! the speed test did not PASS (a skip or a failure)"; rc=1; }
echo "finished: $(date '+%F %T %Z') rc=$rc" | tee -a "$OUT/provenance.txt"
exit $rc
