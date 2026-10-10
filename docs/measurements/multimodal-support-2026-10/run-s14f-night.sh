#!/usr/bin/env bash
# G-S14f3 (docs/tasks/task-multimodal-support-2026-10.md, "G-S14f", registered before the code): openai/whisper-small, goinfer's pure-Go short-form greedy against transformers 5.15.0 float32, language detected by each,
# on the 73 LibriSpeech dummy clips of G-S14c4. Correctness record, not timed.
#   1. hf       scripts/s14f_hf.py                                  -> hf.json
#   2. goinfer  internal/whisper TestWhisperSmall_clips (realckpt)  -> go.json
#   3. grade    scripts/s14f_grade.py                               -> summary.txt
# Pinned in $BIN: whisper.test (rev beside it) and the python scripts. Checkpoint ~/models/whisper-small and the data ~/goinfer-bench/librispeech-dummy (local NVMe), never the archive.
# Estimate ~20 min (hf ~5, goinfer ~15 at ~12 s a clip on this CPU, measured on 3 clips); queue at 40:
#   python3 scripts/night.py add s14f --est 40 --by "nobara session, S14 G-S14f3" --doc docs/tasks/task-multimodal-support-2026-10.md -- bash <pinned>/run-s14f-night.sh
# S14F_DRY=1 checks the preconditions and prints the plan.
set -uo pipefail
BIN=${BIN:-$HOME/goinfer-bench/s14f}
OUT=${1:-$HOME/goinfer-logs/s14f-$(date +%F)}
DATA=$HOME/goinfer-bench/librispeech-dummy
W=$HOME/models/whisper-small
PY=$HOME/g4venv/bin/python
fatal() { echo "FATAL: $*" >&2; exit 2; }
for f in whisper.test rev s14f_hf.py s14f_grade.py; do [ -e "$BIN/$f" ] || fatal "$BIN/$f is missing"; done
for f in "$W/model.safetensors" "$W/tokenizer.json" "$W/generation_config.json" "$DATA/refs.json" "$PY"; do [ -e "$f" ] || fatal "$f is missing"; done
case "$OUT$W$DATA" in *"/srv/models"*|*"/Volumes/"*) fatal "an archive path";; esac
mkdir -p "$OUT"; SUM="$OUT/summary.txt"; : > "$SUM"
{ echo "binaries: $BIN (rev $(cat "$BIN/rev"))"; echo "started: $(date '+%F %T %Z')"; echo "load: $(cat /proc/loadavg)"; free -g | sed -n 2p; } | tee "$OUT/provenance.txt"
[ -n "${S14F_DRY:-}" ] && { echo "DRY: preconditions hold; plan = transformers arm, goinfer arm, grade"; exit 0; }
step() { local name=$1; shift; local t0=$SECONDS rc; echo "[$(date +%T)] START $name"; "$@" > "$OUT/$name.log" 2>&1; rc=$?; printf '%-9s rc=%-3s %5ss\n' "$name" "$rc" "$((SECONDS - t0))" | tee -a "$SUM"; return $rc; }
step hf timeout 60m "$PY" -I "$BIN/s14f_hf.py" "$W" "$DATA" "$OUT/hf.json" || echo "!! the transformers arm failed" | tee -a "$SUM"
gotest() { ( cd "$BIN" && env GOINFER_HEAVY_TESTS=1 GOINFER_S14F_DIR="$DATA" GOINFER_S14F_OUT="$OUT/go.json" GOINFER_WHISPER_DIR="$W" ./whisper.test -test.run '^TestWhisperSmall_clips$' -test.v -test.timeout 90m ); }
step goinfer gotest || echo "!! the goinfer arm failed" | tee -a "$SUM"
if [ -e "$OUT/hf.json" ] && [ -e "$OUT/go.json" ]; then python3 "$BIN/s14f_grade.py" "$DATA/refs.json" "$OUT/hf.json" "$OUT/go.json" 2>&1 | tee -a "$SUM"
else echo "!! not graded: hf.json or go.json is missing" | tee -a "$SUM"; fi
echo "finished: $(date '+%F %T %Z')" | tee -a "$OUT/provenance.txt"
exit 0
