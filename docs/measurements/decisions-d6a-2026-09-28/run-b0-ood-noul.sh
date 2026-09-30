#!/bin/bash
# D6a gap check (decisions-d6a-2026-09-28.md, "The 14-point gap"): the transformers f32 reference Route A (B0, bare-v1,
# base Qwen3.5-9B, no adapter) on the OOD sample's 400 noul rows, so the counterfactual's estimate can be replaced by a
# measurement. Unattended: the pin script is read from REV by `git show` (the tree may move before tonight), the venv
# is D0's (transformers 5.16.1, peft 0.21.0, torch CPU; the script refuses any other version), output is appended and
# resumable. ~106k prompt tokens at D0's measured 45 ms/token on this box: ~80 min plus the f32 load.
#   REV=<commit> bash run-b0-ood-noul.sh
set -euo pipefail
REV=${REV:?}
REPO=${REPO:-$HOME/mycode/goinfer}
B=${B:-$HOME/goinfer-bench/decisions-d6a}
PY=${PY:-$HOME/d0venv/bin/python3}
OUT=${OUT:-$B/b0-ood-noul-f32.jsonl}
work=$(mktemp -d)
git -C "$REPO" show "$REV:scripts/pin_decisions_d0.py" > "$work/pin_decisions_d0.py"
echo "[$(date '+%H:%M:%S')] pin script from $REV; rows $B/samples-amended/eval-sample.jsonl; out $OUT"
JEV_MODEL_PATH=$HOME/models/JEV-9B "$PY" "$work/pin_decisions_d0.py" b0rows \
  --rows "$B/samples-amended/eval-sample.jsonl" --kind noul --out "$OUT" ${LIMIT:+--limit "$LIMIT"}
echo "[$(date '+%H:%M:%S')] done: $(wc -l < "$OUT") rows in $OUT"
