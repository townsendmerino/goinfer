#!/usr/bin/env bash
# Mellum2.1 Gate 1, part 2 (docs/tasks/task-mellum21-2026-10.md): the goinfer side of the parity gates against the 2.1 checkpoint and the goldens part 1 pinned from its real Hugging Face weights.
#   0. tokenizer  tokenizer.json byte-identical to the staged copy that TestByteLevel_mellum2GoldenParity passed on (21 passes, 2026-10-09): the golden carries over
#   1. logit      TestMellum2_logitParity  (chat golden, thinking off; argmax exact, sample-256 cosine >= 0.98 as 2.0's; 2.0 measured 0.99955)
#   2. window     TestMellum2_windowParity (1,441 tokens past the 1,024 window; same bars; 2.0 measured 0.99636)
#   3. identity   TestMoEExpertMajor_bitIdentical, P18's `!=` on every logit through the real forward at K=600 (past the expert-major chunk boundary), flag on against off
#   4. engagement TestMoEExpertMajor_endToEnd at K=600, one pair: NOT the P18 decision measurement (that is a timed FUND/PARK rule at K=4096, 1,200 s a forward) but the check that expert-major ran
#                 in the "on" arm and not in the "off" arm, on 2.1's weights. Timed, so this job holds the timing lock (night.py does) and nothing else runs on the box.
# A SKIP is a failure of this job (no checkpoint, no golden, a heavy gate not opted in). Pinned: $BIN/decoder.test built from the tree at queue time ($BIN/rev, $BIN/tree.diff).
# Estimate ~100 min (three loads of the 12B at int8int8/int4 ~6-10 min each; the 1,441-token window prefill ~15-30 min; K=600 forwards ~4-7 min each pair); queue at 150:
#   python3 scripts/night.py add mellum21-gates --est 150 --by "nobara session, Mellum2.1 Gate 1" --doc docs/tasks/task-mellum21-2026-10.md -- env BIN=$HOME/goinfer-bench/mellum21 bash docs/measurements/mellum21-2026-10/run-mellum21-gates.sh
# MELLUM_DRY=1 checks preconditions and prints the plan.
set -uo pipefail
SRC=$(cd "$(dirname "$0")/../../.." && pwd)
BIN=${BIN:-$HOME/goinfer-bench/mellum21}
CK=${CK:-$HOME/models/mellum2.1-thinking}
GOLD=${GOLD:-$HOME/goinfer-logs/mellum21-gold}
OUT=${1:-$HOME/goinfer-logs/mellum21-gates-$(date +%F)}
STAGED=$HOME/goinfer-bench/mellum21/tok21
fatal() { echo "FATAL: $*" >&2; exit 2; }
[ -x "$BIN/decoder.test" ] || fatal "$BIN/decoder.test is missing"
for f in "$CK/config.json" "$CK/model-00005-of-00005.safetensors" "$GOLD/mellum21_forward_golden.json" "$GOLD/mellum21_window_golden.json" "$STAGED/tokenizer.json"; do [ -e "$f" ] || fatal "$f is missing (part 1 must have run)"; done
case "$CK$GOLD$OUT" in *"/srv/models"*|*"/Volumes/"*) fatal "an archive path";; esac
mkdir -p "$OUT"
SUM="$OUT/summary.txt"; : > "$SUM"
{ echo "binary: $BIN (tree $(cat "$BIN/rev" 2>/dev/null))"; echo "checkpoint: $CK; goldens: $GOLD"; echo "started: $(date '+%F %T %Z')"; echo "load: $(cat /proc/loadavg)"; free -g | sed -n 2p; df -h "$HOME" | tail -1; } | tee "$OUT/provenance.txt"
[ -n "${MELLUM_DRY:-}" ] && { echo "DRY: preconditions hold; plan = tokenizer identity, logit, window, bit-identity K=600, engagement K=600"; exit 0; }
step() { local name=$1; shift; local t0=$SECONDS rc
  echo "[$(date +%T)] START $name"
  ( while sleep 60; do echo "[$(date +%T)] ... $name running $((SECONDS - t0))s; last: $(tail -n1 "$OUT/$name.log" | cut -c1-120)"; done ) &
  local tick=$!
  "$@" > "$OUT/$name.log" 2>&1; rc=$?
  kill "$tick" 2>/dev/null; wait "$tick" 2>/dev/null
  printf '%-12s rc=%-3s %5ss\n' "$name" "$rc" "$((SECONDS - t0))" | tee -a "$SUM"; return $rc; }
gt() { local run=$1; shift; ( cd "$SRC/decoder" && env GOINFER_HEAVY_TESTS=1 GOINFER_MELLUM_CKPT="$CK" "$@" "$BIN/decoder.test" -test.run "^${run}\$" -test.v -test.timeout 150m ); }
# a gate reads "PASS" only if its own PASS line is in its log and it printed no SKIP: a skipped heavy test returns 0
verdict() { local name=$1 test=$2
  if grep -q -- "--- PASS: $test " "$OUT/$name.log" && ! grep -q -- "--- SKIP: $test " "$OUT/$name.log"; then echo "$name: PASS" | tee -a "$SUM"
  elif grep -q -- "--- SKIP: $test " "$OUT/$name.log"; then echo "$name: SKIPPED (a failure of this job): $(grep -m1 -A1 -- "=== RUN   $test" "$OUT/$name.log" | tail -1 | cut -c1-160)" | tee -a "$SUM"
  else echo "$name: FAIL (see $OUT/$name.log)" | tee -a "$SUM"; fi; }

if cmp -s "$CK/tokenizer.json" "$STAGED/tokenizer.json"; then echo "tokenizer: tokenizer.json byte-identical to the copy TestByteLevel_mellum2GoldenParity passed on (sha256 $(sha256sum "$CK/tokenizer.json" | cut -c1-16))" | tee -a "$SUM"; else echo "tokenizer: DIFFERS from the tested copy: the byte-level golden must be re-run before anything else counts" | tee -a "$SUM"; fi
step logit  gt TestMellum2_logitParity  GOINFER_MELLUM_GOLDEN_PREFIX="$GOLD/mellum21"; grep -E "argmax:|sample-256 cosine" "$OUT/logit.log" | tee -a "$SUM"; verdict logit TestMellum2_logitParity
step window gt TestMellum2_windowParity GOINFER_MELLUM_GOLDEN_PREFIX="$GOLD/mellum21"; grep -E "argmax:|sample-256 cosine" "$OUT/window.log" | tee -a "$SUM"; verdict window TestMellum2_windowParity
step identity gt TestMoEExpertMajor_bitIdentical; verdict identity TestMoEExpertMajor_bitIdentical
step engagement gt TestMoEExpertMajor_endToEnd GOINFER_MOE_BATCH_E2E=1 GOINFER_MOE_BATCH_K=600 GOINFER_MOE_BATCH_PAIRS=1; grep -E "P18 e2e|pair |ratio" "$OUT/engagement.log" | tee -a "$SUM"; verdict engagement TestMoEExpertMajor_endToEnd
echo "finished: $(date '+%F %T %Z')" | tee -a "$OUT/provenance.txt"
exit 0
