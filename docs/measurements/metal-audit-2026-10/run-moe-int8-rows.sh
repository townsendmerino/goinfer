#!/usr/bin/env bash
# Night job: the int8 expert GEMVs' rows form on a resident native int8 MoE (docs/tasks/task-metal-audit-2026-10.md,
# "Int8 MoE rows form", pre-registered 2026-10-04 before it runs). TestDB04R_expertRowsAB at GOINFER_DB04R_QUANT=int8int8
# on the Qwen1.5-MoE 4-layer slice, resident (the memory guard declines it by day: 5.43 GB against 3.0 GB live): the
# one-row gemv_w8a8_moe(_wacc) against gemv_w8a8_moe_rows / _wacc_rows on one resident, 32 tokens, 7 reps alternated,
# GPU time a token, logits bit-identical first (the test fails otherwise). Rule (the owner's bar): stays on at a median
# one-row / rows >= 1.02; parked at 1.00-1.02; off below 1.00.
#
# Pinned: ~/goinfer-bench/metal-audit-2026-10/metal-tagged-REV.test, run from the worktree's metal/ (../testdata).
# Queued with:
#   python3 scripts/night.py add metal-moe-int8-rows --est 5 --priority 60 --by "Claude (Mac session), int8 MoE rows" \
#     --doc docs/tasks/task-metal-audit-2026-10.md -- bash docs/measurements/metal-audit-2026-10/run-moe-int8-rows.sh
set -uo pipefail
REV=e7a8bef4
BASE=$HOME/goinfer-bench/metal-audit-2026-10
BIN=$BASE/metal-tagged-$REV.test
WT=$BASE/wt-int8rows-$REV
LOG=$HOME/goinfer-logs/metal-audit-2026-10/moe-int8-rows
mkdir -p "$LOG"
[ -x "$BIN" ] || { echo "missing $BIN"; exit 1; }
[ -d "$WT/metal" ] || { echo "missing worktree $WT"; exit 1; }
echo "started: $(date '+%F %T %Z'); rev $REV (sha256 $(shasum -a 256 "$BIN" | cut -c1-16)); swap $(sysctl -n vm.swapusage)" | tee "$LOG/provenance.txt"
cd "$WT/metal"
GOINFER_DB04R=1 GOINFER_DB04R_QUANT=int8int8 "$BIN" -test.run '^TestDB04R_expertRowsAB$' -test.v -test.timeout 20m > "$LOG/ab.log" 2>&1
rc=$?
grep -E 'd-b04r|FAIL|differ' "$LOG/ab.log" | tee "$LOG/results.txt"
echo "finished: $(date '+%F %T %Z'), exit $rc" | tee -a "$LOG/provenance.txt"
exit $rc
