#!/bin/bash
# D13's last bullet, Clef 27B on nobara's CPU (docs/measurements/decisions-d13-clef27b-2026-10-03.md, pre-registered before this ran).
# Unattended, for nobara's night queue: the reference at bf16 on every 5th D10 record, then goinfer int8int8 on the same 30, then the grade. Every input is
# sha256-checked, the model is read from ~/models (never the archive), both phases resume (rows already written are skipped), nothing needs a Claude
# session. Exits non-zero, loudly, on every refusal.
#   bash docs/measurements/decisions-d13-clef27b-2026-10/run-clef27b.sh
set -uo pipefail
REPO=${REPO:?the goinfer tree at the registered commit (a pinned worktree)}
B=$HOME/goinfer-bench/decisions-d13-clef27b
MODEL=$HOME/models/clef
PY=$HOME/d0venv/bin/python
BIN=$HOME/goinfer-bench/decisions-d13/clef-fidelity.test
BIN_SHA=ca71f9e8dffbc51046f5fc52afea9acb9f26de0bb61612646dcd914c7e651437
RECORDS_SHA=5942ccb997012d69c37945ef8cf3a2dd1974f62009ef14d9a668f5d4c49f9b7c
export CLEF_REPO=Cloudflare/clef CLEF_REV=2f3de3dd85f379784083b0814d997ab627200f0c
export CLEF_MODULE_SHA256=0e304cf7c6500e8bb59bef7e2afd2c6373f82596dfb3b57d1aa93c175e2dc3a3
export CLEF_HEAD_SHA256=a010ac04f078e699988e4049cbea5e62c962393f59fec366640b64e8d69a4953
die() { echo "$(date +%T) REFUSED: $*"; exit 1; }
sum() { sha256sum "$1" | cut -d' ' -f1; }
[ -s "$MODEL/joint_head.safetensors" ] || die "no Clef 27B at $MODEL (rsync -a /srv/models/clef/ ~/models/clef/)"
[ "$(sum "$MODEL/joint_head.safetensors")" = "$CLEF_HEAD_SHA256" ] || die "the head's sha256 differs from the registered one"
[ "$(sum "$MODEL/joint_schema_model.py")" = "$CLEF_MODULE_SHA256" ] || die "the module's sha256 differs from the registered one"
[ -x "$BIN" ] && [ "$(sum "$BIN")" = "$BIN_SHA" ] || die "the harness binary $BIN is missing or differs from the registered one"
[ "$(sum "$REPO/testdata/decisions/clef/records.jsonl")" = "$RECORDS_SHA" ] || die "records.jsonl differs from the registered one"
[ -x "$PY" ] || die "no reference venv at $PY"
mkdir -p "$B/out"
cp "$REPO/testdata/decisions/clef/records.jsonl" "$B/out/records.jsonl"
echo "$(date '+%F %T %Z') == start; load $(cut -d' ' -f1-3 /proc/loadavg); free $(free -g | awk '/Mem/{print $7}') GB; $(nproc) threads"

echo "$(date +%T) == 1. the reference, bf16 (f32-GEMM emulation), every 5th record"
(cd "$REPO" && CLEF_MODEL="$MODEL" D10_OUT="$B/out" "$PY" scripts/pin_clef_d10.py model --dtype bf16 --every 5) || die "the reference failed"
n=$(wc -l < "$B/out/probs_bf16.jsonl"); [ "$n" -eq 30 ] || die "the reference wrote $n of 30 rows"

echo "$(date +%T) == 2. goinfer int8int8, every 5th record"
(cd "$REPO/internal/clef" && CLEF_FIDELITY_ARM=int8int8 CLEF_FIDELITY_OUT="$B/out/probs_goinfer_int8int8.jsonl" CLEF_FIDELITY_EVERY=5 \
  CLEF_MODEL_DIR="$MODEL" CLEF_HEAD_DIR="$MODEL" CLEF_FIDELITY_WALL=170 "$BIN" -test.run 'TestFidelityArm_run$' -test.v -test.timeout 3h) \
  || die "the goinfer arm failed"
n=$(wc -l < "$B/out/probs_goinfer_int8int8.jsonl"); [ "$n" -eq 30 ] || die "the goinfer arm wrote $n of 30 rows"

echo "$(date +%T) == 3. the grade"
python3 "$REPO/docs/measurements/decisions-d13-clef-fidelity-2026-10/grade.py" --ref "$B/out/probs_bf16.jsonl" --ref-every 5 \
  --arms int8int8="$B/out/probs_goinfer_int8int8.jsonl":5 | tee "$B/out/grade.txt"
echo "$(date '+%F %T %Z') == DONE; rows and grade in $B/out"
