#!/bin/bash
# G-S12.4b-4a-5 (docs/tasks/task-multimodal-support-2026-10.md): which image-prefill route under WebGPU is nearer the
# model, on the fidelity protocol against the float32 reference. Qwen2.5-VL 3B, the 94 registered prompts pooled
# (MODE=both), on the box whose WebGPU backend the decision is about.
# Runs with nobody watching: the test binary is pinned in $B/bin (its revision in <binary>.rev) and runs from
# $B/fidelity-root/gpu, beside a pinned copy of the images it reads as ../testdata. night.py holds the timing lock.
# A restarted run skips the prompts already in $OUT/positions-live.jsonl.
set -u
B=${B:-$HOME/goinfer-bench/s12-4b-2026-10-12}
ROOT=$B/fidelity-root
BIN=$B/bin/gpu-fidelity.test
MODEL=${GOINFER_QWEN25VL_DIR:-$HOME/models/qwen25vl-3b-instruct}
OUT=${OUT:-$B/fidelity-gate}
[ -x "$BIN" ] || { echo "FATAL: $BIN is missing"; exit 2; }
for f in "$ROOT/gpu" "$ROOT/testdata/glm_ocr/invoices/inv15.png" "$ROOT/testdata/gemma3_preprocess_image.png" "$MODEL/tokenizer.json"; do
  [ -e "$f" ] || { echo "FATAL: $f is missing"; exit 2; }
done
case "$MODEL" in /Volumes/*|/srv/models*) echo "FATAL: $MODEL is the archive"; exit 2;; esac
mkdir -p "$OUT" || exit 2
{
  echo "started: $(date '+%F %T %Z')"; echo "host: $(uname -srm)"
  echo "binary: $(cat "$BIN.rev" 2>/dev/null)"; echo "model: $MODEL"
  command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
  echo "load: $(uptime)"
} >> "$OUT/provenance.txt"
cd "$ROOT/gpu" || exit 2
GOINFER_S12_FIDELITY=${MODE:-both} GOINFER_S12_FIDELITY_OUT=$OUT GOINFER_QWEN25VL_DIR=$MODEL \
  "$BIN" -test.run 'TestS12MRoPEFidelity$' -test.v -test.timeout 170m 2>&1 | tee -a "$OUT/run.log"
rc=${PIPESTATUS[0]}
if ! grep -q '^VERDICT ' "$OUT/run.log"; then
  echo "no VERDICT line: the run did not grade (a skip is not a pass)" | tee -a "$OUT/run.log"
  rc=1
fi
echo "finished: $(date '+%F %T %Z') rc=$rc" >> "$OUT/provenance.txt"
exit "$rc"
