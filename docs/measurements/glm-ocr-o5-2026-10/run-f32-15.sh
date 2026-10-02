#!/usr/bin/env bash
# GLM-OCR O5, the f32 half of the 15-invoice accuracy report (synthetic invoices, NOT real scans). Queued for the night
# (docs/tasks/task-glm-ocr-2026-10.md O5; root CLAUDE.md "Run budget"): by day only a 3-document f32 subset (documents 2, 8, 10) ran.
#
# Self-contained and unattended: it uses a PINNED CPU serve binary and a PINNED copy of the eval script and the test data,
# all staged under $D (nothing is read from the working tree, which may move before tonight), starts its own server, runs
# the 15 documents at f32 on the CPU backend, and writes durable results under $D. No prompts, no Claude session needed.
#
# Estimate: MEASURED by day on three of these documents (exploratory, load from a concurrent test run): 79-114 s a document at f32
# on the CPU backend (the f32 vision tower ~45 s, the f32 prefill of ~1,780 ids, 230-430 decode tokens), so 15 documents are about
# 25-30 min, queued as 40. Staging (by whoever queues it): see $D/STAGED.txt; re-stage if the tree moved and the run should
# include the newer serve.
set -euo pipefail
D="$HOME/goinfer-logs/glm-ocr-o5"
BIN="$D/serve-cpu"
EVAL="$D/scripts/o5_eval_invoices.py"
PORT=18082
OUT="$D/f32-15-$(date +%F)"
mkdir -p "$OUT"
for f in "$BIN" "$EVAL" "$D/testdata/glm_ocr/invoices/labels.json" "$D/testdata/glm_ocr/invoice.schema.json" "$HOME/models/glm-ocr/model.safetensors"; do
  [ -e "$f" ] || { echo "run-f32-15: missing $f (stage it first: see $D/STAGED.txt)" >&2; exit 1; }
done
cp "$D/STAGED.txt" "$OUT/STAGED.txt"
"$BIN" --model "glm=$HOME/models/glm-ocr" --backend cpu --quant f32 --ctx 8192 --addr "127.0.0.1:$PORT" >"$OUT/serve.log" 2>&1 &
SP=$!
trap 'kill "$SP" 2>/dev/null || true' EXIT
for _ in $(seq 1 120); do
  curl -sf "http://127.0.0.1:$PORT/health" >/dev/null && break
  kill -0 "$SP" 2>/dev/null || { echo "run-f32-15: the server exited; see $OUT/serve.log" >&2; exit 1; }
  sleep 2
done
python3 "$EVAL" --url "http://127.0.0.1:$PORT" --tag f32-cpu --out "$OUT" --max-tokens 2500 2>&1 | tee "$OUT/eval.log"
# An unattended run must not report success without its output (a refused or crashed run exits 0 elsewhere in this repo).
[ -s "$OUT/results_f32-cpu.json" ] || { echo "run-f32-15: no results file" >&2; exit 1; }
python3 - "$OUT/results_f32-cpu.json" <<'PY'
import json, sys
r = json.load(open(sys.argv[1]))["results"]
if len(r) != 15:
    sys.exit(f"run-f32-15: {len(r)} of 15 documents finished")
PY
echo "run-f32-15: done; results in $OUT"
