#!/bin/bash
# OP6-MP fidelity gate, a night job (night.py; it holds the timing lock, so this script takes none).
#   bash run-fidelity.sh <name> [K list]   K list empty = the registered cells (512,2048); e.g. "512" fills the cache only.
# Runs the pre-built test binary BIN (pinned at the commit that built it) from the branch's backend/metal, where the
# prompt files resolve. The CPU reference is cached in CACHE, so a later job reuses what an earlier one computed.
# Exit: 0 PASS, 1 FAIL or no verdict, 75 the test skipped (a guard declined the load; night.py re-queues it).
set -u
NAME=${1:?name}
KS=${2:-}
D=$HOME/goinfer-logs/op6mp-2026-11-20
BIN=${BIN:-$D/metal.test}
CACHE=$D/ref-cache
LOG=$D/$NAME.log
export TMPDIR=$HOME/goinfer-work/tmp/mac-perf8
mkdir -p "$TMPDIR" "$CACHE"
cd "$HOME/tmcode/perf8-wt/backend/metal" || exit 1
{
  echo "START $(date) binary $(stat -f '%Sm' -t '%F %T' "$BIN") K=${KS:-registered}"
  uptime
} | tee -a "$LOG"
GATE=1
[ -n "${SMOKE:-}" ] && GATE=   # a by-day check of the plumbing: the test skips at once
env ${GATE:+GOINFER_OP6MP_FIDELITY=1} OP6MP_REF_CACHE="$CACHE" OP6MP_POSITIONS_LOG="$D/$NAME-positions.jsonl" \
  ${KS:+OP6MP_FIDELITY_K=$KS} \
  caffeinate -i "$BIN" -test.run '^TestOP6MP_q4kPassFidelity$' -test.v -test.count=1 -test.timeout 240m 2>&1 | tee -a "$LOG"
RC=${PIPESTATUS[0]}
echo "EXIT $RC $(date)" | tee -a "$LOG"
if grep -q -- '--- SKIP: TestOP6MP_q4kPassFidelity' "$LOG"; then exit 75; fi
if [ "$RC" -ne 0 ] || ! grep -q -- '--- PASS: TestOP6MP_q4kPassFidelity' "$LOG"; then exit 1; fi
exit 0
