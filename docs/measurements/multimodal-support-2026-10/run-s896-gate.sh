#!/usr/bin/env bash
# G-S10k (docs/tasks/task-multimodal-support-2026-10.md, "G-S10k" and its amendment A1): the noise-calibrated gate for the resident DeepStack prefill, 64 units (4 images x 16 prompts).
#   1. run 1  the dump at serve's configuration: OFF (fast CPU attention), ON and the three planted ON arms        -> $OUT/a
#   2. run 2  the same with GOINFER_CPU_FAST_ATTENTION=0 (OFFX) forced along run 1's teacher path               -> $OUT/b
#   3. grade  Hugging Face float32 for each unit (cached) and the registered rule                                -> $OUT/grade.log, $OUT/a/gate-result.json
# The cuda test binary is built once from the tree at start and pinned (the tree may move while it runs). Estimate ~45 min (12-15 s a unit per run for the dumps, ~14 s a unit for HF);
# normally a night job, run by day on the owner's "run it now". A heartbeat is the dump's own line per unit and a ticker every 60 s. Usage: run-s896-gate.sh [out dir]
set -uo pipefail
SRC=$(cd "$(dirname "$0")/../../.." && pwd)
OUT=${1:-$HOME/goinfer-logs/s896g-$(date +%F)}
export PATH=/usr/local/go/bin:$PATH GOTOOLCHAIN=go1.27.2
mkdir -p "$OUT/a" "$OUT/b"
cd "$SRC" || exit 2
{ echo "rev: $(git rev-parse HEAD) (+dirty files: $(git status --short | grep -vc '^??'))"; echo "started: $(date '+%F %T %Z')"; echo "load: $(cat /proc/loadavg)"; nvidia-smi --query-gpu=name,driver_version,memory.used --format=csv,noheader; } | tee "$OUT/provenance.txt"
(cd cuda && go test -c -tags 'cuda goinfer_testhooks' -o "$OUT/cuda.test" .) || { echo "FATAL: build"; exit 2; }
# The ticker is started directly, never inside $(...): a command substitution waits for every holder of its pipe, and a background loop holds it forever (the first launch of this script hung there).
run() { # label dir extra-env...
  local label=$1 dir=$2; shift 2
  ( while sleep 60; do echo "[$(date +%T)] ... $label running, $(ls "$dir" 2>/dev/null | grep -c '\.meta\.json$') unit files in $dir"; done ) &
  local t=$!
  ( cd cuda && env GOINFER_HEAVY_TESTS=1 GOINFER_S896G_DIR="$dir" "$@" "$OUT/cuda.test" -test.run '^TestS896_gateDump$' -test.v -test.timeout 90m ) > "$OUT/$label.log" 2>&1
  local rc=$?; kill "$t" 2>/dev/null; wait "$t" 2>/dev/null
  echo "$label rc=$rc units=$(ls "$dir" | grep -c '\.meta\.json$')"; return $rc
}
run run1 "$OUT/a" GOINFER_S896G_PLANTED=late,notadded,textrows || exit 3
run run2 "$OUT/b" GOINFER_S896G_KNOBS=GOINFER_CPU_FAST_ATTENTION=0 GOINFER_S896G_TEACHER_DIR="$OUT/a" || exit 3
~/g4venv/bin/python -I scripts/s896_gate_grade.py ~/models/qwen3-vl-2b-instruct "$OUT/a" "$OUT/b" ~/goinfer-logs/s10j-r3 2>&1 | grep -v "Loading weights" | tee "$OUT/grade.log"
echo "finished: $(date '+%F %T %Z')" | tee -a "$OUT/provenance.txt"
