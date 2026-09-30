#!/usr/bin/env bash
# P8a — the 9B leg of G0 (tower) and G2 (tokens vs HF). Night-queue job: nobody watching, no prompts.
#
#   python3 scripts/night.py add p8a-g2-9b --est 120 --by "vscode-claude, P8a" \
#       --doc docs/measurements/p8a-qwen35-vl-2026-09/preregistration.md \
#       -- bash docs/measurements/p8a-qwen35-vl-2026-09/run-g2-9b.sh
#
# What it does, in order (any failing step stops the job non-zero; the log is the record):
#   1. download Qwen/Qwen3.5-9B at a PINNED revision into ~/models/qwen3.5-9b (bf16, 4 shards, ~19 GB,
#      resumable); skipped if the checkpoint is already complete;
#   2. write the G2 images + pixel_values for this checkpoint's geometry (identical to the 0.8B's);
#   3. HF f32 references (tower golden for G0, generation goldens for G2) — ~36 GB of the box's 62 GB RAM;
#   4. aikit G0 real tower test against the 9B golden (bars: preregistration.md G0 + A1);
#   5. goinfer G2 on the 9B (CPU f32, ~36 GB; HF's process has exited by then, so they never coexist).
# Each phase logs entering/leaving with its duration. Estimate: download 5-10 min, HF references ~15 min,
# goinfer G2 ~40-60 min (a 9B f32 CPU decode step reads ~36 GB) => ~90-120 min. Bars are the pre-registered
# ones; a divergence with an HF top1-top2 gap < 0.02 PARKS the gate (teacher-forced follow-up), it is not a pass.
set -euo pipefail
REV=c202236235762e1c871ad0ccb60c8ee5ba337b9a          # Qwen/Qwen3.5-9B main at 2026-09-30
REPO=$(cd "$(dirname "$0")/../../.." && pwd)
AIKIT=${AIKIT:-$HOME/mycode/aikit/aikit}
CK=$HOME/models/qwen3.5-9b
G2=$HOME/models/qwen35vl_g2_9b
TG=$HOME/models/qwen35vl_tower_golden_9b
PY=$HOME/g4venv/bin/python
t0=$(date +%s)
phase() { echo "[$(date +%H:%M:%S)] ($(( $(date +%s) - t0 ))s elapsed) $*"; }
phase "1/5 download Qwen/Qwen3.5-9B @ $REV -> $CK"
$HOME/g4venv/bin/hf download Qwen/Qwen3.5-9B --revision "$REV" --local-dir "$CK" >/dev/null
for f in config.json preprocessor_config.json model.safetensors.index.json tokenizer.json chat_template.jinja; do
  [ -s "$CK/$f" ] || { echo "MISSING $CK/$f after download"; exit 2; }
done
[ "$(ls "$CK"/model.safetensors-0000*-of-00004.safetensors | wc -l)" = 4 ] || { echo "9B shards incomplete"; exit 2; }
cd "$REPO/scripts"
phase "2/5 images + pixel_values"
$PY pin_qwen35_vl_preprocess.py --g2 "$G2"
phase "3/5 HF f32 references (tower golden, then generation goldens)"
(cd "$AIKIT" && $PY scripts/oracle/pin_qwen35_vision.py --real "$CK" --out "$TG/qwen35vl_tower_golden.json.gz")
$PY pin_qwen35_vl_real.py "$CK" "$G2"
phase "4/5 aikit G0 real tower (9B)"
(cd "$AIKIT" && AIKIT_QWEN35_08B="$CK" AIKIT_QWEN35VL_TOWER_GOLDEN="$TG/qwen35vl_tower_golden.json.gz" \
   go test ./vision/ -run TestQwen3VisionEncoder_realParity -v -timeout 30m -count=1)
phase "5/5 goinfer G2 (9B, CPU f32)"
cd "$REPO"
GOINFER_HEAVY_TESTS=1 go test -tags realckpt ./decoder/ -run 'TestQwen35VLReal_G2_9B' -v -timeout 150m -count=1
phase "done"
