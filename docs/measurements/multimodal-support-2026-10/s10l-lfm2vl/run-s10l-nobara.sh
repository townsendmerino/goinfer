#!/usr/bin/env bash
# S10, LFM2.5-VL: G-S10l-a/b/c on nobara (docs/tasks/task-multimodal-support-2026-10.md, "S10, LFM2.5-VL"). From a goinfer
# worktree whose go.work uses the aikit branch carrying the SigLIP2 NaFlex tower:
#   1. goinfer writes its ids for the full-model prompt (decoder/lfm2vl_real_test.go, step ids);
#   2. transformers writes every reference (scripts/pin_lfm2vl_real.py, ~/.venv-vl, float32, one load, ~8 GB);
#   3. goinfer grades a and b (multimodal/lfm2vl_real_test.go) and c (step compare).
# Each step's log goes to $OUT. Correctness gates, not timed: no timing lock.
set -euo pipefail
export PATH=$PATH:/usr/local/go/bin
REPO=${REPO:-$HOME/wt/goinfer-s10l}
MODEL=${MODEL:-$HOME/models/lfm25-vl-1.6b}
OUT=${OUT:-$HOME/goinfer-logs/lfm2vl-real}
case "$MODEL" in /Volumes/*|/srv/models*) echo "$MODEL is the archive (CLAUDE.md)"; exit 2;; esac
mkdir -p "$OUT"
cd "$REPO"
stamp() { echo "[$(date '+%H:%M:%S')] $*"; }
export GOINFER_HEAVY_TESTS=1 GOINFER_LFM2VL="$MODEL" GOINFER_S10L_OUT="$OUT" GOINFER_LFM2VL_REAL_ARTIFACTS="$OUT"

stamp "1/4 ids"
GOINFER_S10L_STEP=ids go test -tags realckpt ./decoder/ -run '^TestLfm2VLFull$' -count=1 -v -timeout 10m 2>&1 | tee "$OUT/step1-ids.log"
stamp "2/4 transformers references"
~/.venv-vl/bin/python scripts/pin_lfm2vl_real.py --model "$MODEL" --images testdata --out "$OUT" 2>&1 | tee "$OUT/step2-hf.log"
# a/b failing must not stop c: each grading step's status is kept, and the script exits non-zero if either failed.
stamp "3/4 G-S10l-a and b"
rc_ab=0; go test ./multimodal/ -run '^TestLfm2VLReal_' -count=1 -v -timeout 30m 2>&1 | tee "$OUT/step3-ab.log" || rc_ab=$?
stamp "4/4 G-S10l-c"
rc_c=0; GOINFER_S10L_STEP=compare go test -tags realckpt ./decoder/ -run '^TestLfm2VLFull$' -count=1 -v -timeout 60m 2>&1 | tee "$OUT/step4-c.log" || rc_c=$?
stamp "done: a/b rc=$rc_ab, c rc=$rc_c"
[ "$rc_ab" -eq 0 ] && [ "$rc_c" -eq 0 ]
