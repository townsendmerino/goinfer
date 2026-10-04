#!/usr/bin/env bash
# Night job: MoE int8's gates X3 and X4 (docs/tasks/task-metal-int8-2026-10.md, "Slice 4: MoE int8"), pre-registered
# before the run: X3 is TestW8Native_F3amended_closerToF32 on the Qwen1.5-MoE 4-layer slice at int8int8 (native MoE
# path), the f32 reference read from the file nobara wrote (sha256 edccc9374170749a...); X4 is D-G01's gate
# (TestDG01_expertMajorMoEPrefill) at GOINFER_DG01_QUANT=int8int8, its bars unchanged. X3 moved here from the day: the
# Metal memory guard declined the native int8 slice with the owner's apps open.
#
# Runs a test binary built at REV from a clean tree, from a worktree pinned at REV (the tests read ../testdata):
#   go test -c -tags goinfer_testhooks -o ~/goinfer-bench/metal-int8-2026-10/metal-<REV>.test ./metal/
#   git worktree add --detach ~/goinfer-bench/metal-int8-2026-10/wt-<REV> <REV>
# Queued with:
#   python3 scripts/night.py add metal-int8-moe-x34 --est 20 --by "Claude, Metal MoE int8 X3/X4" \
#     --doc docs/tasks/task-metal-int8-2026-10.md -- bash docs/measurements/metal-int8-2026-10/run-moe-x34.sh
# Logs: ~/goinfer-logs/metal-int8-2026-10/moe-x34/.
set -uo pipefail
REV=ac8622e1
BASE=$HOME/goinfer-bench/metal-int8-2026-10
BIN=$BASE/metal-$REV.test
WT=$BASE/wt-$REV
REF=$HOME/goinfer-logs/metal-int8-2026-10/f3ref-moe-slice-amd64.gob.gz
SLICE=$HOME/models/qwen15-moe-a27b-l4slice
LOG=$HOME/goinfer-logs/metal-int8-2026-10/moe-x34
mkdir -p "$LOG"
[ -x "$BIN" ] || { echo "missing test binary $BIN"; exit 1; }
[ -d "$WT/testdata/prefill-gate-prose-a" ] || { echo "missing worktree $WT"; exit 1; }
[ -f "$SLICE/config.json" ] || { echo "missing $SLICE"; exit 1; }
[ "$(shasum -a 256 "$REF" | cut -c1-16)" = "edccc9374170749a" ] || { echo "reference $REF missing or not the one nobara wrote"; exit 1; }
cd "$WT/metal"
{
  echo "rev:      $REV"
  echo "binary:   $BIN (sha256 $(shasum -a 256 "$BIN" | cut -c1-16))"
  echo "ref:      $REF (sha256 $(shasum -a 256 "$REF" | cut -c1-16))"
  echo "started:  $(date '+%F %T %Z')"
  echo "load:     $(uptime | sed 's/.*load averages*: //')"
} | tee "$LOG/provenance.txt"
rc=0
echo "== X3 — $(date '+%T')" | tee -a "$LOG/provenance.txt"
env GOINFER_HEAVY_TESTS=1 GOINFER_W8_GATE_MODEL=$SLICE GOINFER_W8_F3_REF_IN=$REF "$BIN" -test.v -test.count=1 -test.timeout 40m \
  -test.run '^TestW8Native_F3amended_closerToF32$' 2>&1 | tee "$LOG/x3.log"
s=${PIPESTATUS[0]}; echo "x3 exit $s" | tee -a "$LOG/provenance.txt"; [ "$s" -eq 0 ] || rc=1
echo "== X4 — $(date '+%T')" | tee -a "$LOG/provenance.txt"
env GOINFER_DG01=1 GOINFER_DG01_QUANT=int8int8 GOINFER_DG01_MODEL=$SLICE "$BIN" -test.v -test.count=1 -test.timeout 40m \
  -test.run '^TestDG01_expertMajorMoEPrefill$' 2>&1 | tee "$LOG/x4.log"
s=${PIPESTATUS[0]}; echo "x4 exit $s" | tee -a "$LOG/provenance.txt"; [ "$s" -eq 0 ] || rc=1
echo "finished: $(date '+%F %T %Z')" | tee -a "$LOG/provenance.txt"
exit $rc
