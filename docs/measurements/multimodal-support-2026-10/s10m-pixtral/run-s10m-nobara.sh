#!/usr/bin/env bash
# S10, Ministral 3 (Pixtral): G-S10m-a/b/c on nobara (docs/tasks/task-multimodal-support-2026-10.md, "S10, Ministral 3
# (Pixtral)"). From a goinfer worktree whose go.work uses the aikit branch carrying the Pixtral tower:
#   1. goinfer writes its ids for the full-model prompt (decoder/pixtral_real_test.go, step ids);
#   2. transformers writes every reference (scripts/pin_pixtral_real.py, ~/.venv-vl, float32, one load, ~17 GB);
#   3. goinfer grades a and b (multimodal/pixtral_real_test.go) and c (step compare).
# Each step's log goes to $OUT. Correctness gates, not timed: no timing lock.
set -euo pipefail
export PATH=$PATH:/usr/local/go/bin
REPO=${REPO:-$HOME/wt/goinfer-s10px}
MODEL=${MODEL:-$HOME/models/ministral3-3b-bf16}
OUT=${OUT:-$HOME/goinfer-logs/pixtral-real}
case "$MODEL" in /Volumes/*|/srv/models*) echo "$MODEL is the archive (CLAUDE.md)"; exit 2;; esac
mkdir -p "$OUT"
cd "$REPO"
stamp() { echo "[$(date '+%H:%M:%S')] $*"; }
export GOINFER_HEAVY_TESTS=1 GOINFER_MINISTRAL3="$MODEL" GOINFER_S10M_OUT="$OUT" GOINFER_PIXTRAL_REAL_ARTIFACTS="$OUT"

stamp "1/4 ids"
GOINFER_S10M_STEP=ids go test -tags realckpt ./decoder/ -run '^TestPixtralFull$' -count=1 -v -timeout 10m 2>&1 | tee "$OUT/step1-ids.log"
stamp "2/4 transformers references"
~/.venv-vl/bin/python scripts/pin_pixtral_real.py --model "$MODEL" --images testdata --out "$OUT" 2>&1 | tee "$OUT/step2-hf.log"
# a/b failing must not stop c: each grading step's status is kept, and the script exits non-zero if either failed.
stamp "3/4 G-S10m-a and b"
rc_ab=0; go test ./multimodal/ -run '^TestPixtralReal_' -count=1 -v -timeout 30m 2>&1 | tee "$OUT/step3-ab.log" || rc_ab=$?
stamp "4/4 G-S10m-c"
rc_c=0; GOINFER_S10M_STEP=compare go test -tags realckpt ./decoder/ -run '^TestPixtralFull$' -count=1 -v -timeout 60m 2>&1 | tee "$OUT/step4-c.log" || rc_c=$?
stamp "done: a/b rc=$rc_ab, c rc=$rc_c"
[ "$rc_ab" -eq 0 ] && [ "$rc_c" -eq 0 ]
