#!/usr/bin/env bash
# Night job: 4b's grade, the exact layer-major paged prefill on M26 (docs/tasks/task-m26-mac-2026-10.md, "4b",
# pre-registered 2026-10-04 before any graded run). TestG4LayerMajor_M26AB: a fresh prose prompt through the sequential
# loop and through prefillG4Paged at M = 128 and 512, one warm-up then 5 reps per arm alternated, one process, logits
# and K/V compared every rep; the auto-sizer's slot count. Rule (amended by the owner 2026-10-04, before it ran): stays
# on at a median sequential / layer-major >= 1.02 at both M with every rep's K/V equal; parked at 1.00-1.02; off below
# 1.00. The pinned binary calls prefillG4Paged directly, so it grades the path whatever its switch says.
# Amended 2026-10-04, before it ran: re-pinned d79269eb -> a845af18, so the sequential arm is today's production
# (async phase 2, the routed-expert rows form) and the ratio does not flatter the layer-major path.
#
# Pinned (the tree may move before tonight):
#   test binary ~/goinfer-bench/m26-mac-2026-10/metal-tagged-a845af18.test
#   worktree    ~/goinfer-bench/m26-mac-2026-10/wt-a845af18 (the binary runs from its metal/ for ../testdata)
# Queued with:
#   python3 scripts/night.py add m26-g4lm-grade --est 20 --by "Claude (Mac session), M26 4b grade" \
#     --doc docs/tasks/task-m26-mac-2026-10.md -- bash docs/measurements/m26-mac-2026-10/run-g4lm-grade.sh
# Estimate: about 12 minutes (M = 512 sequential at ~90 ms a token is ~46 s a rep, 6 reps of both arms; M = 128 about
# 2 minutes; the load about 10 s). The night runner holds the timing lock; swap_killwatch.sh guards M26's memory.
set -uo pipefail
REV=a845af18
B=$HOME/goinfer-bench/m26-mac-2026-10
BIN=$B/metal-tagged-$REV.test
WT=$B/wt-$REV
M26=$HOME/models/gemma4-26b-int4-v14st.metal.giw
LOG=$HOME/goinfer-logs/m26-mac/g4lm-grade
mkdir -p "$LOG"
[ -x "$BIN" ] || { echo "missing $BIN"; exit 1; }
[ -d "$WT/metal" ] || { echo "missing worktree $WT"; exit 1; }
[ -f "$M26" ] || { echo "missing model $M26"; exit 1; }
{
  echo "rev:     $REV"
  echo "binary:  $BIN (sha256 $(shasum -a 256 "$BIN" | cut -c1-16))"
  echo "model:   $M26 ($(stat -f %z "$M26") bytes)"
  echo "started: $(date '+%F %T %Z')"
  echo "load:    $(uptime | sed 's/.*load averages*: //')"
  echo "swap:    $(sysctl -n vm.swapusage)"
} | tee "$LOG/provenance.txt"
cd "$WT/metal"
GOINFER_G4LM_M26=1 GOINFER_AUDIT_MODEL=$M26 GOINFER_G4LM_MS=128,512 \
  "$BIN" -test.run '^TestG4LayerMajor_M26AB$' -test.v -test.timeout 60m > "$LOG/grade.log" 2>&1 &
P=$!
KILL_DELTA_MB=1024 POLL_S=1 TICK_MB=80 bash "$WT/scripts/swap_killwatch.sh" "$P" "$LOG/killwatch.log" > /dev/null 2>&1 &
wait "$P"
rc=$?
grep -E 'built|RESULT|FAIL|differ|panic' "$LOG/grade.log" | tee "$LOG/results.txt"
echo "finished: $(date '+%F %T %Z'), exit $rc" | tee -a "$LOG/provenance.txt"
exit $rc
