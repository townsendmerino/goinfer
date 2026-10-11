#!/usr/bin/env bash
# S6's Gemma 4 E4B on Metal as amended 2026-10-11 (docs/tasks/task-multimodal-support-2026-10.md, "Gemma 4 E4B on
# Metal", the amendment): G-E4B-1b and G-E4B-2 on Google's quantization-aware checkpoint, and the plain checkpoint as a
# record. Run by day on the owner's word; nothing here is timed. Every step holds the timing lock.
#
# Inputs, all on local disk:
#   ~/models/gemma-4-e4b-gguf/gemma-4-E4B_q4_0-it.gguf   google/gemma-4-E4B-it-qat-q4_0-gguf, sha256 676c3507...53fbaee
#   ~/models/gemma-4-E4B-qat-unq-towers                   build_qat_towers.py: the vision and audio tensors of
#                                                         google/gemma-4-E4B-it-qat-q4_0-unquantized and its configs
#   ~/models/gemma-4-E4B-it                               the plain checkpoint (record only; skipped when absent)
#   $BIN/{prequant,serve-metal,metal.test}                built from the rev in $BIN/rev
# Estimate: 15 minutes (two sidecar builds, two runs of the test at about 2.5 min, the served arms about 4 min).
set -uo pipefail
SRC=$(cd "$(dirname "$0")/../../../.." && pwd)
BIN=${BIN:-$HOME/goinfer-bench/s6-e4b-metal-2026-10-11/bin}
OUT=${1:-$HOME/goinfer-logs/s6-e4b-metal-$(date +%F)}
M=$HOME/models
E=$M/gemma-4-e4b-gguf/gemma-4-E4B_q4_0-it
V=$M/gemma-4-E4B-qat-unq-towers
P=$M/gemma-4-E4B-it
Q=$M/qwen2.5-coder-1.5b-instruct-q4_k_m
LOCK="python3 $SRC/scripts/timing_lock.py run --label"
for f in "$BIN/prequant" "$BIN/serve-metal" "$BIN/metal.test" "$E.gguf" "$V/model.safetensors" "$Q.gguf"; do
  [ -e "$f" ] || { echo "missing: $f"; exit 2; }
done
mkdir -p "$OUT"; rc=0
{ echo "binaries $BIN rev $(cat "$BIN/rev")"; date '+started %F %T %Z'; sw_vers | tr '\n' ' '; echo; pmset -g batt | head -1; uptime; } > "$OUT/provenance.txt"
[ -e "$Q.int4.metal.giw" ] || "$BIN/prequant" -quant int4 -target metal -o "$Q.int4.metal.giw" "$Q.gguf" | tail -1
[ -e "$E.int4.metal.giw" ] || "$BIN/prequant" -quant int4 -target metal -o "$E.int4.metal.giw" "$E.gguf" > "$OUT/1-prequant-qat.log" 2>&1
g3() { # g3 <log> [env...]: the gate test from its package directory
  local log=$1; shift
  ( cd "$SRC/metal" && env GOINFER_HEAVY_TESTS=1 "$@" $LOCK s6-e4b-g1b -- "$BIN/metal.test" \
      -test.run '^TestGemma4EModel_realE4BNonInferiority$' -test.v -test.timeout 45m ) > "$log" 2>&1
}
echo "G-E4B-1b $(date +%T)"; g3 "$OUT/2-g-e4b-1b.log"
grep -q -- '--- PASS: TestGemma4EModel_realE4BNonInferiority' "$OUT/2-g-e4b-1b.log" || { echo "!! G-E4B-1b did not PASS (a failure or a skip)"; rc=1; }
grep -E 'non-inferiority|^--- ' "$OUT/2-g-e4b-1b.log"
if [ -d "$P" ]; then # the record: the same procedure on the plain checkpoint; its verdict line is not this gate's
  "$BIN/prequant" -quant int4 -target metal -o "$P.int4.metal.giw" "$P" > "$OUT/3-prequant-plain.log" 2>&1
  echo "plain record $(date +%T)"; g3 "$OUT/4-plain-record.log" GOINFER_GEMMA4_E4B_GIW="$P.int4.metal.giw" GOINFER_GEMMA4_E4B_TOK="$P"
  grep -E 'non-inferiority' "$OUT/4-plain-record.log"
  find "$M" -maxdepth 1 -name 'gemma-4-E4B-it.int4.metal.giw*' -delete
fi
cd "$SRC" || exit 2
echo "G-E4B-2 image $(date +%T)"
GS3C_SETTLE=15 GS3C_EXTRA="--vision $V --embed-int4=false" $LOCK s6-e4b-g2-image -- \
  bash docs/measurements/multimodal-support-2026-10/run-gs3c-served.sh "$BIN/serve-metal" "$OUT/5a-image" metal:cpu,=cpu:cpu,cpu:cpu "$E.gguf" > "$OUT/5a-image.log" 2>&1 || rc=1
grep -E 'decode path|IDENTICAL|differing' "$OUT/5a-image.log"
echo "G-E4B-2 audio $(date +%T)"
$LOCK s6-e4b-g2-audio -- bash docs/measurements/multimodal-support-2026-10/run-gs5c-served.sh "$BIN/serve-metal" "$OUT/5b-audio" '=cpu,metal,cpu' -- \
  --model "$E.gguf" --vision "$V" -vision-device cpu --embed-int4=false > "$OUT/5b-audio.log" 2>&1 || rc=1
grep -E 'decode path|IDENTICAL|differing' "$OUT/5b-audio.log"
# The CPU arms build a cpu-arm64 sidecar beside the GGUF (5.7 GB); it is a cache and is removed to give the disk back.
find "$M/gemma-4-e4b-gguf" -maxdepth 1 -name 'gemma-4-E4B_q4_0-it.int4.cpu-arm64.giw*' -delete
echo "finished $(date '+%F %T %Z') rc=$rc" | tee -a "$OUT/provenance.txt"
exit $rc
