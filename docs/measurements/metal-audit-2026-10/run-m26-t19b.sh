#!/usr/bin/env bash
# Night job: T1.9b of docs/tasks/task-metal-audit-2026-10.md, re-run alone, first in the night (owner, 2026-10-02).
# On 2026-10-01 it ran 30 s after two other M26 processes and the resident guard declined the 64-slot build on
# live-available memory (7.72 GB against 6.65 GB), so it never reached Metal. The rule is the M26 table's T1.9b row,
# unchanged; the binary is the one that night ran (f56b40ec, which still has the scale cache), so R11(c) is tested on
# the code the audit read. The guard is not bypassed: if it declines again, that is recorded and the run is ungraded.
#
#   python3 scripts/night.py add metal-audit-t19b --est 10 --priority 10 --by "Claude, Metal audit T1.9b re-run" \
#     --doc docs/tasks/task-metal-audit-2026-10.md -- bash ~/goinfer-bench/metal-audit-2026-10/run-m26-t19b.sh
# Logs: ~/goinfer-logs/metal-audit-2026-10/m26-t19b/.
set -uo pipefail
REV=f56b40ec
BASE=$HOME/goinfer-bench/metal-audit-2026-10
BIN=$BASE/metal-tagged-$REV.test
KW=$BASE/swap_killwatch.sh
LOG=$HOME/goinfer-logs/metal-audit-2026-10/m26-t19b
M26=$HOME/models/gemma4-26b-int4-v14st.metal.giw
mkdir -p "$LOG"
[ -x "$BIN" ] || { echo "missing test binary $BIN"; exit 1; }
[ -f "$KW" ] || { echo "missing kill-watch $KW"; exit 1; }
[ -f "$M26" ] || { echo "missing model $M26"; exit 1; }
cd "$BASE"
{
  echo "rev:      $REV"
  echo "binary:   $BIN (sha256 $(shasum -a 256 "$BIN" | cut -c1-16))"
  echo "model:    $M26 ($(stat -f %z "$M26") bytes)"
  echo "started:  $(date '+%F %T %Z')"
  echo "machine:  $(sysctl -n machdep.cpu.brand_string), $(( $(sysctl -n hw.memsize) / 1073741824 )) GB, macOS $(sw_vers -productVersion)"
  echo "load:     $(uptime | sed 's/.*load averages*: //')"
  echo "swap:     $(sysctl -n vm.swapusage)"
  echo "memory:   $(memory_pressure | tail -1)"
} | tee "$LOG/provenance.txt"

env GOINFER_METAL_AUDIT_M26=1 GOINFER_AUDIT_MODEL="$M26" GOINFER_AUDIT_SLOTS=64 GOINFER_AUDIT_HOLD= \
  "$BIN" -test.v -test.count=1 -test.timeout 15m -test.run '^TestAuditM26_pagedProbe$' > "$LOG/t1.9b-64slots.log" 2>&1 &
pid=$!
KILL_DELTA_MB=1024 TICK_MB=80 TICK_N=2 POLL_S=1 bash "$KW" "$pid" "$LOG/t1.9b-64slots.killwatch.log" > /dev/null &
kw=$!
wait "$pid"
rc=$?
wait "$kw" 2>/dev/null
echo "t1.9b-64slots rc=$rc; swap now $(sysctl -n vm.swapusage)" | tee -a "$LOG/provenance.txt"
grep -q 'KILLING' "$LOG/t1.9b-64slots.killwatch.log" 2>/dev/null && echo "t1.9b-64slots: the kill-watch fired" | tee -a "$LOG/provenance.txt"
grep -q 'BuildResident declined' "$LOG/t1.9b-64slots.log" && echo "t1.9b-64slots: the resident guard declined again (ungraded)" | tee -a "$LOG/provenance.txt"
grep -h 'RESULT\|built in\|BuildResident declined' "$LOG/t1.9b-64slots.log" | cut -c1-400 | tee "$LOG/results.txt"
echo "finished: $(date '+%F %T %Z')" | tee -a "$LOG/provenance.txt"
