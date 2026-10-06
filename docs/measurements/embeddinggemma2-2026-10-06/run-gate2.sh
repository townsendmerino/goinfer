#!/usr/bin/env bash
# Night job (Mac): Gate 2 of docs/tasks/task-embeddinggemma2.md, pre-registered there ("Gate 2: pre-registered
# 2026-10-06, before it runs") in the commit this job is pinned to. (1) The sentence-transformers reference on the real
# google/embeddinggemma-2 (48 parity texts, and the Matryoshka truncation reading over the repo's own docs at REV),
# (2) goinfer's TestReal_parity against it. CPU, float32, both sides. Not a timed measurement: no throughput figure.
#
# Pinned (the tree may move before tonight): worktree ~/goinfer-bench/embeddinggemma2-2026-10-06/wt-REV, the realckpt
# test binary built from it, and a venv with transformers 5.19.0, sentence-transformers 6.1.0, torch 2.14.0 and
# torchvision 0.29.1. The checkpoint is ~/models/embeddinggemma-2 (revision 914f7f89, sha256 197a3296...), local disk.
# Queued with:
#   python3 scripts/night.py add embeddinggemma2-gate2 --est 30 --by "Claude (Mac session), EmbeddingGemma 2 Gate 2" \
#     --doc docs/tasks/task-embeddinggemma2.md -- bash docs/measurements/embeddinggemma2-2026-10-06/run-gate2.sh
set -uo pipefail
REV=2ec0cdc9
B=$HOME/goinfer-bench/embeddinggemma2-2026-10-06
WT=$B/wt-$REV
BIN=$B/eg2-$REV.test
PY=$B/venv/bin/python
MODEL=$HOME/models/embeddinggemma-2
LOG=$HOME/goinfer-logs/embeddinggemma2-2026-10-06
mkdir -p "$LOG"
for f in "$BIN" "$PY"; do [ -x "$f" ] || { echo "FATAL: missing $f"; exit 2; }; done
[ -f "$WT/scripts/pin_embeddinggemma2_real.py" ] || { echo "FATAL: missing worktree $WT"; exit 2; }
[ -f "$MODEL/model.safetensors" ] || { echo "FATAL: missing $MODEL/model.safetensors"; exit 2; }
case "$MODEL" in /Volumes/*|/srv/models*) echo "FATAL: $MODEL is the archive, not the bench set"; exit 2;; esac
cd "$WT" || exit 2
{
  echo "started: $(date '+%F %T %Z'); rev $(git rev-parse HEAD); checkpoint revision $(cat "$MODEL/REVISION" 2>/dev/null)"
  echo "weights sha256 $(shasum -a 256 "$MODEL/model.safetensors" | cut -c1-16)"
  echo "$("$PY" -c 'import transformers, sentence_transformers, torch; print("transformers", transformers.__version__, "sentence-transformers", sentence_transformers.__version__, "torch", torch.__version__)')"
} | tee "$LOG/provenance.txt"
echo "=== step 1: reference ($(date '+%T'))" | tee -a "$LOG/provenance.txt"
"$PY" -u scripts/pin_embeddinggemma2_real.py --model "$MODEL" --rev "$REV" --golden "$LOG/golden.json" --truncation "$LOG/truncation.json" 2>&1 | tee "$LOG/reference.log"
rc1=${PIPESTATUS[0]}
rc2=99
if [ "$rc1" = 0 ]; then
  echo "=== step 2: goinfer parity ($(date '+%T'))" | tee -a "$LOG/provenance.txt"
  cd "$WT/embeddinggemma2" || exit 2
  GOINFER_EG2_DIR="$MODEL" GOINFER_EG2_GOLDEN="$LOG/golden.json" "$BIN" -test.run '^TestReal_parity$' -test.v -test.count=1 -test.timeout 30m 2>&1 | tee "$LOG/parity.log"
  rc2=${PIPESTATUS[0]}
fi
echo "finished: $(date '+%F %T %Z'), reference exit $rc1, parity exit $rc2" | tee -a "$LOG/provenance.txt"
grep -E 'RESULT|FAIL' "$LOG/reference.log" "$LOG/parity.log" 2>/dev/null | cut -c1-240
[ "$rc1" = 0 ] && [ "$rc2" = 0 ]
