#!/usr/bin/env bash
# S6's Gemma 4 E4B on Metal (docs/tasks/task-multimodal-support-2026-10.md, "Gemma 4 E4B on Metal, gates registered"):
#   1. the int4 Metal sidecar, built once (prequant -quant int4 -target metal; skipped when it exists);
#   2. G-E4B-1, S1's G3 non-inferiority rule on E4B (metal.test, TestGemma4EModel_realE4BNonInferiority);
#   3. G-E4B-2, served: G4's image request (run-gs3c-served.sh) and G-S5c's audio clips (run-gs5c-served.sh), Metal
#      against the CPU with equal load flags, plus a CPU repeat.
# At night because the sidecar build and E4B's resident both need memory the owner's session leaves short. Each step's
# failure is recorded and the next still runs where it can.
#
# Pinned in $BIN (built from main at the rev in $BIN/rev): prequant, metal.test; serve-metal from s2-towers (rev in
# serve-metal.rev; it has -vision-device). Checkpoint ~/models/gemma-4-E4B-it (local disk). Estimate: sidecar ~10 min,
# G-E4B-1 ~20 min, served ~10 min; queued at 60.
#   python3 scripts/night.py add s6-e4b --est 60 --by "Claude, S6" --doc docs/tasks/task-multimodal-support-2026-10.md -- bash docs/measurements/multimodal-support-2026-10/run-s6-e4b-night.sh
set -uo pipefail
SRC=$(cd "$(dirname "$0")/../../.." && pwd)
BIN=${BIN:-$HOME/goinfer-bench/s6}
OUT=${1:-$HOME/goinfer-bench/s6/e4b-$(date +%F)}
DIR=$HOME/models/gemma-4-E4B-it
GIW=$HOME/models/gemma-4-E4B-it.int4.metal.giw
for f in prequant metal.test serve-metal; do [ -x "$BIN/$f" ] || { echo "FATAL: $BIN/$f is missing"; exit 2; }; done
[ -d "$DIR" ] || { echo "FATAL: $DIR is missing"; exit 2; }
mkdir -p "$OUT"
{ echo "binaries: $BIN (main $(cat "$BIN/rev" 2>/dev/null), serve $(cat "$BIN/serve-metal.rev" 2>/dev/null))"; echo "started:  $(date '+%F %T %Z')"
  sw_vers 2>/dev/null | tr '\n' ' '; echo; pmset -g batt | head -1; df -h ~ | tail -1; } | tee "$OUT/provenance.txt"
rc=0
echo "=== 1. sidecar $(date '+%T')"
if [ -e "$GIW" ]; then
  echo "exists: $GIW"
else
  "$BIN/prequant" -quant int4 -target metal -o "$GIW" "$DIR" > "$OUT/1-prequant.log" 2>&1 || { echo "!! prequant failed"; tail -5 "$OUT/1-prequant.log"; rc=1; }
  tail -3 "$OUT/1-prequant.log"
fi
if [ -e "$GIW" ]; then
  echo "=== 2. G-E4B-1 $(date '+%T')"
  ( cd "$SRC/metal" && GOINFER_HEAVY_TESTS=1 GOINFER_GEMMA4_E4B_GIW="$GIW" "$BIN/metal.test" \
      -test.run '^TestGemma4EModel_realE4BNonInferiority$' -test.v -test.timeout 45m ) > "$OUT/2-g3.log" 2>&1
  grep -E "G-E4B-1|G3|^--- " "$OUT/2-g3.log" | tail -8
  grep -q -- "--- PASS: TestGemma4EModel_realE4BNonInferiority" "$OUT/2-g3.log" || { echo "!! G-E4B-1 did not PASS (a failure or a skip)"; rc=1; }
  echo "=== 3. G-E4B-2 served $(date '+%T')"
  cd "$SRC" || exit 2
  GS3C_EXTRA="--vision $DIR --embed-int4=false" bash docs/measurements/multimodal-support-2026-10/run-gs3c-served.sh \
    "$BIN/serve-metal" "$OUT/3a-image" metal:cpu,=cpu:cpu,cpu:cpu "$GIW" > "$OUT/3a-image.log" 2>&1 || rc=1
  grep -E "decode path|IDENTICAL|differing|top-3|near-tie|exited" "$OUT/3a-image.log"
  bash docs/measurements/multimodal-support-2026-10/run-gs5c-served.sh "$BIN/serve-metal" "$OUT/3b-audio" =cpu,metal,cpu -- \
    --model "$GIW" --vision "$DIR" -vision-device cpu > "$OUT/3b-audio.log" 2>&1 || rc=1
  grep -E "decode path|IDENTICAL|differing|exited" "$OUT/3b-audio.log"
fi
echo "finished: $(date '+%F %T %Z') rc=$rc" | tee -a "$OUT/provenance.txt"
exit $rc
