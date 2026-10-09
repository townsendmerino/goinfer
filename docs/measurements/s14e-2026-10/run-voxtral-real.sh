#!/usr/bin/env bash
# G-S14e3 (docs/tasks/task-multimodal-support-2026-10.md, S14.4b): the REAL Voxtral Mini 3B, goinfer float32 on the CPU against transformers float32.
#   1. ref   scripts/pin_voxtral_real.py: loads transformers' VoxtralForConditionalGeneration (zero missing keys asserted), transcribes the LibriSpeech clip (asserted), dumps the tower output, projector rows,
#            prompt logits, the greedy generation and the teacher-forced logits for the clip (graded case) and the clip x6 (two windows, record only). mistral_common's dependencies come from a directory of
#            their own on PYTHONPATH (g4venv is untouched).
#   2. gate  TestVoxtralReal_gate from a PINNED realckpt test binary ($BIN/decoder-real.test): the registered bars on the clip, the record-only readings on the two-window case. A SKIP fails this job.
# Not timed (it grades correctness, not speed), so it takes no timing lock of its own; night.py holds one for the whole job anyway.
#   python3 scripts/night.py add voxtral-real --est <min> --by "nobara session, S14.4b" --doc docs/tasks/task-multimodal-support-2026-10.md -- bash docs/measurements/s14e-2026-10/run-voxtral-real.sh
set -uo pipefail
SRC=$(cd "$(dirname "$0")/../../.." && pwd)
BIN=${BIN:-$HOME/goinfer-bench/s14e/real}
MODEL=${MODEL:-$HOME/models/voxtral-mini-3b-2507}
PY=${PY:-$HOME/g4venv/bin/python}
MC=${MC:-$HOME/goinfer-bench/s14e/mc}
OUT=${1:-$HOME/goinfer-logs/s14e/real-$(date +%F)}
fatal() { echo "FATAL: $*" >&2; exit 2; }
case "$MODEL$OUT" in *"/srv/models"*|*"/Volumes/"*) fatal "an archive path";; esac
[ -x "$BIN/decoder-real.test" ] || fatal "$BIN/decoder-real.test missing"; [ -x "$PY" ] || fatal "$PY missing"; [ -d "$MC/pydantic" ] || fatal "$MC (mistral_common's dependencies) missing"
for f in config.json tekken.json model-00001-of-00002.safetensors model-00002-of-00002.safetensors; do [ -e "$MODEL/$f" ] || fatal "$MODEL/$f missing"; done
mkdir -p "$OUT"; SUM="$OUT/summary.txt"; : > "$SUM"
{ echo "binary: $BIN (rev $(cat "$BIN/rev" 2>/dev/null))"; echo "model: $MODEL"; echo "started: $(date '+%F %T %Z')"; echo "load: $(cat /proc/loadavg)"; free -g | sed -n 2p; df -h "$HOME" | tail -1; } | tee "$OUT/provenance.txt"
[ -n "${VOX_DRY:-}" ] && { echo "DRY: preconditions hold; plan = reference (both cases), then the realckpt gate"; exit 0; }
step() { local name=$1; shift; local t0=$SECONDS rc
  echo "[$(date +%T)] START $name"
  ( while sleep 60; do echo "[$(date +%T)] ... $name running $((SECONDS - t0))s; last: $(tail -n1 "$OUT/$name.log" | cut -c1-120)"; done ) &
  local tick=$!
  "$@" > "$OUT/$name.log" 2>&1; rc=$?
  kill "$tick" 2>/dev/null; wait "$tick" 2>/dev/null
  printf '%-8s rc=%-3s %5ss\n' "$name" "$rc" "$((SECONDS - t0))" | tee -a "$SUM"; return $rc; }
cd "$SRC" || fatal "cd $SRC"
step ref env PYTHONPATH="$MC" "$PY" -W ignore scripts/pin_voxtral_real.py --model "$MODEL" --out "$OUT/ref" || { echo "RESULT: the reference failed (the clip was not transcribed, keys were missing, or an error): see $OUT/ref.log" | tee -a "$SUM"; exit 1; }
grep -E "loaded in|missing|libri.*generated" "$OUT/ref.log" | tee -a "$SUM"
cd "$SRC/decoder" || fatal "cd decoder"
step gate env GOINFER_HEAVY_TESTS=1 GOINFER_VOXTRAL_REF="$OUT/ref" GOINFER_VOXTRAL_DIR="$MODEL" "$BIN/decoder-real.test" -test.run '^TestVoxtralReal_gate$' -test.v -test.timeout 90m
grep -E "loaded in|features|tower rows|prefill|free-run|reference text|teacher-forced|RECORD ONLY" "$OUT/gate.log" | cut -c1-300 | tee -a "$SUM"
if grep -q -- '--- PASS: TestVoxtralReal_gate' "$OUT/gate.log" && ! grep -q -- '--- SKIP: TestVoxtralReal_gate' "$OUT/gate.log"; then echo "gate: PASS" | tee -a "$SUM"
elif grep -q -- '--- SKIP: TestVoxtralReal_gate' "$OUT/gate.log"; then echo "gate: SKIPPED (a failure of this job)" | tee -a "$SUM"
else echo "gate: FAIL (see $OUT/gate.log)" | tee -a "$SUM"; fi
echo "finished: $(date '+%F %T %Z')" | tee -a "$OUT/provenance.txt"; exit 0
