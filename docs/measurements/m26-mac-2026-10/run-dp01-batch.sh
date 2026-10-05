#!/usr/bin/env bash
# Night job: D-P01's speed read, the batched expert phase 2 on M26 (docs/tasks/task-m26-mac-2026-10.md, "D-P01",
# pre-registered 2026-10-05 before any graded run). TestG4LayerMajor_M26AB with GOINFER_G4LM_BATCHAB=1: the layer-major
# pass with the per-row phase 2 against the same pass batched (g4ExpertBatchOn), on one resident, M = 128 and 512, one
# warm-up then 5 reps per arm alternated, logits and K/V compared every rep; the auto-sizer's slot count. Rule (the
# owner's bar): on at a median per-row / batched >= 1.02 at both M with every rep's logits and K/V equal; parked at
# 1.00-1.02 at either; off below 1.00 at either. The day attempt (2026-10-05 08:42) was killed by the kill-watch at
# M = 128 and is not graded; this is the read, unchanged.
#
# Pinned (the tree may move before tonight):
#   test binary ~/goinfer-bench/m26-mac-2026-10/metal-tagged-f7497c2c.test
#   worktree    ~/goinfer-bench/m26-mac-2026-10/wt-f7497c2c (the binary runs from its metal/ for ../testdata)
# Queued with:
#   python3 scripts/night.py add m26-dp01-batch --est 20 --priority 60 --by "Claude (Mac session), M26 D-P01" \
#     --doc docs/tasks/task-m26-mac-2026-10.md -- bash docs/measurements/m26-mac-2026-10/run-dp01-batch.sh
# Estimate: about 12 minutes (layer-major at M = 512 is about 29-36 s an arm, 6 reps of both; M = 128 about 2 minutes;
# the load about 10 s). Priority 60 runs it after the short-prompt TTFT job, which needs no M26 memory. The night runner
# holds the timing lock; swap_killwatch.sh guards M26's memory.
set -uo pipefail
REV=f7497c2c
B=$HOME/goinfer-bench/m26-mac-2026-10
BIN=$B/metal-tagged-$REV.test
WT=$B/wt-$REV
M26=$HOME/models/gemma4-26b-int4-v14st.metal.giw
LOG=$HOME/goinfer-logs/m26-mac/dp01-batch
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
GOINFER_G4LM_M26=1 GOINFER_AUDIT_MODEL=$M26 GOINFER_G4LM_MS=128,512 GOINFER_G4LM_BATCHAB=1 \
  "$BIN" -test.run '^TestG4LayerMajor_M26AB$' -test.v -test.timeout 60m > "$LOG/grade.log" 2>&1 &
P=$!
KILL_DELTA_MB=1024 POLL_S=1 TICK_MB=80 bash "$WT/scripts/swap_killwatch.sh" "$P" "$LOG/killwatch.log" > /dev/null 2>&1 &
wait "$P"
rc=$?
grep -E 'built|RESULT|FAIL|differ|panic' "$LOG/grade.log" | tee "$LOG/results.txt"
echo "finished: $(date '+%F %T %Z'), exit $rc" | tee -a "$LOG/provenance.txt"
exit $rc
