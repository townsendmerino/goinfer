#!/usr/bin/env bash
# Night job: D-B04's grade (docs/tasks/task-metal-audit-2026-10.md, "D-B04: built, pending its grade"), pre-registered
# there on 2026-10-03 before any graded run. TestDB04AB on the Qwen3.5-9B hybrid: decode with R18's rows form at the
# DeltaNet and shared-expert GEMV sites against off, one resident, 7 reps, at depths 128 and 1024.
#
# Runs a tagged test binary built at REV, from the metal package directory:
#   (cd ~/tmcode/goinfer && go test -c -tags goinfer_testhooks -o ~/goinfer-bench/metal-audit-2026-10/metal-tagged-<REV>.test ./metal/)
# Queued with:
#   python3 scripts/night.py add metal-audit-db04 --est 15 --by "Claude (Mac session), Metal audit D-B04 grade" \
#     --doc docs/tasks/task-metal-audit-2026-10.md -- bash docs/measurements/metal-audit-2026-10/run-db04-grade.sh
# The runner holds the timing lock. Logs: ~/goinfer-logs/metal-audit-2026-10/db04/.
set -uo pipefail
REV=b37b8158
BASE=$HOME/goinfer-bench/metal-audit-2026-10
BIN=$BASE/metal-tagged-$REV.test
REPO=$HOME/tmcode/goinfer
LOG=$HOME/goinfer-logs/metal-audit-2026-10/db04
M9=$HOME/models/Qwen3.5-9B-Q4_K_M.int4.metal.giw
mkdir -p "$LOG"
[ -x "$BIN" ] || { echo "missing test binary $BIN"; exit 1; }
[ -f "$M9" ] || { echo "missing model $M9"; exit 1; }
{
  echo "rev:      $REV"
  echo "binary:   $BIN (sha256 $(shasum -a 256 "$BIN" | cut -c1-16))"
  echo "started:  $(date '+%F %T %Z')"
  echo "machine:  $(sysctl -n machdep.cpu.brand_string), $(( $(sysctl -n hw.memsize) / 1073741824 )) GB, macOS $(sw_vers -productVersion)"
  echo "load:     $(uptime | sed 's/.*load averages*: //')"
  echo "therm:    $(pmset -g therm 2>/dev/null | tr '\n' ' ')"
} | tee "$LOG/provenance.txt"
(cd "$REPO/metal" && GOINFER_METAL_DB04=1 GOINFER_AUDIT_REPS=7 GOINFER_AUDIT_MODEL="$M9" \
  "$BIN" -test.v -test.count=1 -test.timeout 45m -test.run '^TestDB04AB$') > "$LOG/9b.log" 2>&1
rc=$?
echo "9b rc=$rc" | tee -a "$LOG/provenance.txt"
grep -q -- '--- SKIP' "$LOG/9b.log" && { echo "SKIPPED (a skip is not a pass)" | tee -a "$LOG/provenance.txt"; rc=1; }
grep -h 'METRIC\|depth .*: token' "$LOG/9b.log" | tee "$LOG/results.txt"
echo "finished: $(date '+%F %T %Z')" | tee -a "$LOG/provenance.txt"
exit $rc
