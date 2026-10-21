#!/usr/bin/env bash
# Mellum2.1 prefill FFN share + the server's plan at 16k and 32k (README.md here). About 10 minutes; hold the timing lock:
#   python3 scripts/timing_lock.py run --wait 14400 --label mellum21-ffn-share -- bash docs/measurements/mellum21-prefill-ffn-share-2026-10-19/run-ffn-share.sh
set -uo pipefail
cd "$(git rev-parse --show-toplevel)"
OUT=${OUT:-$HOME/goinfer-logs/mellum21-ffn-share-$(date +%F)}
SERVE=${SERVE:-$HOME/serve-cuda-b2b5e816}
MODEL=$HOME/models/mellum2.1-thinking.int4.e4h.cuda.v17.giw
mkdir -p "$OUT"
echo "start $(date '+%F %T %Z') commit $(git rev-parse --short=8 HEAD) serve $SERVE"
echo "== plan at 16k and 32k, expert cache =="
for ctx in 16384 32768; do
  python3 docs/measurements/mellum21-prefill-ffn-share-2026-10-19/plan_banner.py "$SERVE" "$MODEL" $ctx -moe-cache-experts 2>&1 | tee "$OUT/plan-ctx$ctx.txt"
done
echo "== prefill phases, 512 (warm) then 4096 cold =="
GOINFER_HEAVY_TESTS=1 GOINFER_DECOMP_MODEL=mellum2.1-thinking.int4.e4h.cuda.v17.giw GOINFER_DECOMP_GIW=1 GOINFER_DECOMP_MOECACHE=1 GOINFER_DECOMP_CTX=8192 \
  GOINFER_DECOMP_K=512,4096 GOINFER_DECOMP_RUNS=1 \
  go test -count=1 -timeout 25m -tags 'cuda goinfer_testhooks' -run '^TestPrefillDecomp$' -v ./cuda/ 2>&1 | tee "$OUT/decomp.log"
echo "finish $(date '+%F %T %Z')"
