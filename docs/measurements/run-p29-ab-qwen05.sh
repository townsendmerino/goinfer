#!/bin/bash
# P29's tier 1 A/B on the 0.5B q4_k_m forced to one resident KV slot (docs/queue-performance.md, P29, "A/B on the 0.5B",
# registered 2026-10-18), a Mac night job. The lift branch is pinned at 88a08574 in its own worktree. The test skips when a
# guard declines the load; that is a VOID here (exit 2), never an ok.
set -u
W=/Users/francistownsend-merino/goinfer-bench/p29/wt-88a08574
LOG=$HOME/goinfer-bench/p29/ab-qwen05-night-$(date +%F-%H%M).log
cd "$W" || exit 1
[ "$(git rev-parse --short=8 HEAD)" = 88a08574 ] || { echo "worktree is not at 88a08574"; exit 1; }
echo "=== $(date '+%F %T %Z') / $(date -u '+%T UTC') START (rev 88a08574, the 0.5B at one slot)" | tee "$LOG"
vm_stat | head -3 >> "$LOG"; sysctl -n vm.swapusage >> "$LOG"; uptime >> "$LOG"
GOINFER_P29_AB=1 GOWORK=$W/go.work go test -count=1 -timeout 20m -run '^TestOneSlotStep_qwen05PrefillAB$' -v ./metal/ 2>&1 | tee -a "$LOG"
rc=${PIPESTATUS[0]}
echo "=== $(date '+%F %T %Z') END rc=$rc; swap $(sysctl -n vm.swapusage)" | tee -a "$LOG"
if grep -q -- '--- SKIP' "$LOG"; then echo "VOID-refused: the A/B skipped (a guard declined the load); see $LOG"; exit 2; fi
exit $rc
