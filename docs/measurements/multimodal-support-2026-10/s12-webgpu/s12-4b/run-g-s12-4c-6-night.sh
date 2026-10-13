#!/bin/bash
# G-S12.4c-6 (docs/tasks/task-multimodal-support-2026-10.md, "S12.4c"): which attention kernel leaves Gemma 3 4B nearer
# the float32 model under WebGPU, the per-dimension kernel a wide head used to get or the key-split kernel it gets now.
#   1. the pilot with a planted defect in the candidate: its verdict must be OFF, or the instrument is not used;
#   2. the gate: 64 prompts, 32 positions each, both arms against the float32 CPU reference.
# Runs with nobody watching from a pinned test binary; night.py holds the timing lock. GPUDIR is a gpu directory of a
# checkout (the binary reads nothing from it, but runs there as the other pinned test binaries do).
#
#   B=~/goinfer-bench/s12-4c-6-2026-10-12 GPUDIR=~/wt/s12-4b/gpu bash $B/run-g-s12-4c-6-night.sh
set -u
B=${B:-$HOME/goinfer-bench/s12-4c-6-2026-10-12}
GPUDIR=${GPUDIR:-$HOME/wt/s12-4b/gpu}
MODEL=${MODEL:-$HOME/models/gemma-3-4b-it}
BIN=$B/bin/gpu-4c6.test
[ -x "$BIN" ] || { echo "FATAL: $BIN is missing"; exit 2; }
[ -d "$MODEL" ] || { echo "FATAL: $MODEL is missing"; exit 2; }
case "$MODEL" in /Volumes/*|/srv/models*) echo "FATAL: $MODEL is the archive"; exit 2;; esac
mkdir -p "$B/pilot-defect" "$B/gate" || exit 2
{
  echo "started: $(date '+%F %T %Z')"; echo "host: $(uname -srm)"
  echo "binary: $(cat "$BIN.rev" 2>/dev/null)"; echo "model: $MODEL"
  nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null; echo "load: $(uptime)"
} > "$B/gate/provenance.txt"
cd "$GPUDIR" || exit 2
GOINFER_GEMMA3_DIR=$MODEL GOINFER_S12_ATTN_FIDELITY=pilot GOINFER_S12_ATTN_FIDELITY_DEFECT=1 GOINFER_S12_ATTN_FIDELITY_OUT=$B/pilot-defect \
  "$BIN" -test.run '^TestS12AttnKernelFidelity$' -test.v -test.timeout 40m > "$B/pilot-defect/run.log" 2>&1
grep -E '^VERDICT|^--- ' "$B/pilot-defect/run.log"
if ! grep -q '^VERDICT OFF' "$B/pilot-defect/verdict.txt" 2>/dev/null; then
  echo "!! the planted defect did not read OFF: the instrument cannot go red, so the gate is not run"
  echo "finished: $(date '+%F %T %Z') rc=1 (instrument)" >> "$B/gate/provenance.txt"
  exit 1
fi
GOINFER_GEMMA3_DIR=$MODEL GOINFER_S12_ATTN_FIDELITY=gate GOINFER_S12_ATTN_FIDELITY_OUT=$B/gate \
  "$BIN" -test.run '^TestS12AttnKernelFidelity$' -test.v -test.timeout 170m > "$B/gate/run.log" 2>&1
rc=$?
tail -8 "$B/gate/run.log" | cut -c1-400
grep -q '^VERDICT ' "$B/gate/verdict.txt" 2>/dev/null || { echo "!! no verdict: the gate did not finish"; rc=1; }
echo "finished: $(date '+%F %T %Z') rc=$rc" >> "$B/gate/provenance.txt"
exit "$rc"
