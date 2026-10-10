#!/usr/bin/env bash
# Follow-up B2: B's FUND (2.80x) was measured on a 97-token cycle repeated to K=4096, which sends the same tokens to the same experts every cycle: the best case for grouping
# rows by expert. B2 repeats it on real text (the first 4096 tokens of docs/ARCHITECTURE.md + docs/flags.md, the checkpoint's own tokenizer), and checks bit-identity at that depth and input.
# Steps, in one process each (decoder.test pinned beside the source tree it was built from, F below):
#   1. identity   TestMoEExpertMajor_bitIdentical  K=4096 real text   (expert-major on vs off, every logit `!=`)           ~7 min
#   2. e2e_real   TestMoEExpertMajor_endToEnd      K=4096 real text, 2 pairs                                              ~20 min   <- GRADED
#   3. e2e_period TestMoEExpertMajor_endToEnd      K=4096 periodic ids, 2 pairs                                           ~20 min   reference for attribution, not graded
# Timed (night.py holds the timing lock). Estimate ~50 min; queued at 80. Pre-registration: docs/tasks/task-mellum21-2026-10.md, "B2".
#   python3 scripts/night.py add mellum21-fu-b2 --est 80 --by "nobara session, Mellum2.1 follow-up B2" --doc docs/tasks/task-mellum21-2026-10.md -- bash <pinned src>/docs/measurements/mellum21-2026-10/run-mellum21-followup-b2.sh
F=${F:-$HOME/goinfer-bench/mellum21/b2}
source "$(dirname "$0")/_common.sh"
OUT=${1:-$HOME/goinfer-logs/mellum21-followup-b2-$(date +%F)}
[ -x "$F/decoder.test" ] || fatal "$F/decoder.test missing"; [ -e "$CK/config.json" ] || fatal "$CK missing"
[ -e "$SRC/docs/ARCHITECTURE.md" ] && [ -e "$SRC/docs/flags.md" ] || fatal "the two text files the real-text input reads are missing under $SRC/docs"
guard_paths; mkdir -p "$OUT"; SUM="$OUT/summary.txt"; : > "$SUM"; provenance
[ -n "${MELLUM_DRY:-}" ] && { echo "DRY: preconditions hold; plan = identity K=4096, e2e real K=4096 pairs=2, e2e periodic K=4096 pairs=2"; exit 0; }
step identity gt TestMoEExpertMajor_bitIdentical GOINFER_MOE_BATCH_K=4096
grep -E "input:|expert-major chunks|NOT bit-identical|PASS|FAIL|SKIP" "$OUT/identity.log" | tee -a "$SUM"; verdict identity TestMoEExpertMajor_bitIdentical
step e2e_real gt TestMoEExpertMajor_endToEnd GOINFER_MOE_BATCH_E2E=1 GOINFER_MOE_BATCH_K=4096 GOINFER_MOE_BATCH_PAIRS=2
grep -E "P18 e2e|input:|host:|pair |ratio" "$OUT/e2e_real.log" | tee -a "$SUM"; verdict e2e_real TestMoEExpertMajor_endToEnd
step e2e_period gt TestMoEExpertMajor_endToEnd GOINFER_MOE_BATCH_E2E=1 GOINFER_MOE_BATCH_K=4096 GOINFER_MOE_BATCH_PAIRS=2 GOINFER_MOE_BATCH_IDS=periodic
grep -E "P18 e2e|input:|host:|pair |ratio" "$OUT/e2e_period.log" | tee -a "$SUM"; verdict e2e_period TestMoEExpertMajor_endToEnd
echo "finished: $(date '+%F %T %Z')" | tee -a "$OUT/provenance.txt"; exit 0
