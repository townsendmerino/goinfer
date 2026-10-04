#!/usr/bin/env bash
# Night job: D-B01's grade (docs/tasks/task-metal-audit-2026-10.md, "D-B01: built, off by default; pre-registration"),
# pre-registered there on 2026-10-03 before any graded run. From tagged test binaries pinned at REV, each run from its
# package directory (the tests read ../testdata relative to it):
#   1. preconditions: TestDeltaNetSeqKernels_matchDecodeBitwise, TestDB01_chunkedPrefillMatchesWhole and (the 2026-10-04
#      amendment) TestDB01_prefillFromZeroResetsState (any red stops it)
#   2. CPU f32 references for Qwen3.5-0.8B (cell Q35), set A, K = 256, 512, 1024
#   3. the §3.2 pooled fidelity gate on Q35 (a model that does not ship exits non-zero: a verdict, recorded)
#   4. TestAuditDB01_prefillTiming on the 0.8B (K = 128, 512, 2048, 7 reps), then the 9B (K = 128, 512, 5 reps,
#      reported; the fit guard decides whether it loads, and a refusal is recorded, not bypassed)
#
# Binaries, built at REV:
#   (cd ~/tmcode/goinfer && go test -c -tags goinfer_testhooks -o ~/goinfer-bench/metal-audit-2026-10/decoder-tagged-<REV>.test ./decoder/ \
#     && go test -c -tags goinfer_testhooks -o ~/goinfer-bench/metal-audit-2026-10/metal-tagged-<REV>.test ./metal/)
# Queued with:
#   python3 scripts/night.py add metal-audit-db01 --est 40 --by "Claude (Mac session), Metal audit D-B01 grade" \
#     --doc docs/tasks/task-metal-audit-2026-10.md -- bash docs/measurements/metal-audit-2026-10/run-db01-grade.sh
# Logs: ~/goinfer-logs/metal-audit-2026-10/db01/; the readings in results.txt.
set -uo pipefail
REV=f50afe17
BASE=$HOME/goinfer-bench/metal-audit-2026-10
DBIN=$BASE/decoder-tagged-$REV.test
MBIN=$BASE/metal-tagged-$REV.test
REPO=$HOME/tmcode/goinfer
LOG=$HOME/goinfer-logs/metal-audit-2026-10/db01
Q35=$HOME/models/qwen3.5-0.8b
M9=$HOME/models/Qwen3.5-9B-Q4_K_M.int4.metal.giw
REF=$HOME/goinfer-logs/prefill-ref
mkdir -p "$LOG"
for f in "$DBIN" "$MBIN"; do [ -x "$f" ] || { echo "missing test binary $f"; exit 1; }; done
[ -f "$Q35/config.json" ] || { echo "missing $Q35"; exit 1; }
[ -d "$REPO/testdata/prefill-gate-prose-a" ] || { echo "missing the set-A prose seeds under $REPO/testdata"; exit 1; }
FAILED=0
{
  echo "rev:      $REV"
  echo "binaries: $DBIN (sha256 $(shasum -a 256 "$DBIN" | cut -c1-16)), $MBIN (sha256 $(shasum -a 256 "$MBIN" | cut -c1-16))"
  echo "testdata: $REPO/testdata at $(git -C "$REPO" rev-parse --short=8 HEAD)"
  echo "started:  $(date '+%F %T %Z')"
  echo "machine:  $(sysctl -n machdep.cpu.brand_string), $(( $(sysctl -n hw.memsize) / 1073741824 )) GB, macOS $(sw_vers -productVersion)"
  echo "load:     $(uptime | sed 's/.*load averages*: //')"
  echo "therm:    $(pmset -g therm 2>/dev/null | tr '\n' ' ')"
} | tee "$LOG/provenance.txt"

run() { # <log name> <dir> <binary> <test regexp> <rc counts: 1|0> ENV=VALUE...
  local name=$1 dir=$2 bin=$3 re=$4 strict=$5
  shift 5
  echo "== $name — $(date '+%T')" | tee -a "$LOG/provenance.txt"
  (cd "$dir" && env "$@" "$bin" -test.v -test.count=1 -test.timeout 60m -test.run "$re") > "$LOG/$name.log" 2>&1
  local rc=$?
  echo "$name rc=$rc" | tee -a "$LOG/provenance.txt"
  [ "$strict" = 1 ] && [ $rc -ne 0 ] && FAILED=1
  grep -q -- '--- SKIP' "$LOG/$name.log" && { echo "$name: a test SKIPPED (a skip is not a pass)" | tee -a "$LOG/provenance.txt"; FAILED=1; }
  return $rc
}

run preconditions "$REPO/metal" "$MBIN" '^(TestDeltaNetSeqKernels_matchDecodeBitwise|TestDB01_chunkedPrefillMatchesWhole|TestDB01_prefillFromZeroResetsState)$' 1 GOINFER_DB01_MODEL="$Q35" \
  || { echo "a precondition failed: nothing graded" | tee -a "$LOG/provenance.txt"; exit 1; }
[ $FAILED -eq 0 ] || { echo "a precondition skipped: nothing graded" | tee -a "$LOG/provenance.txt"; exit 1; }

run refs "$REPO/decoder" "$DBIN" '^TestPrefillGateReference$' 1 GOINFER_HEAVY_TESTS=1 GOINFER_CPU_REF_MODELS=Q35 GOINFER_CPU_MODEL_Q35="$Q35"
n=$(ls "$REF"/Q35-K256-p*.bin "$REF"/Q35-K512-p*.bin "$REF"/Q35-K1024-p*.bin 2>/dev/null | grep -c '\.bin$')
echo "Q35 reference files: $n of 30" | tee -a "$LOG/provenance.txt"
[ "$n" -eq 30 ] || { echo "references incomplete: stopping, nothing graded" | tee -a "$LOG/provenance.txt"; exit 1; }

run gate "$REPO/metal" "$MBIN" '^TestPrefillGateVsReference$/^Q35$' 0 \
  GOINFER_HEAVY_TESTS=1 GOINFER_METAL_GATE_MODELS=Q35 GOINFER_METAL_GATE_CONFIRM=0 GOINFER_METAL_MODEL_Q35="$Q35"

run speed-0.8b "$REPO/metal" "$MBIN" '^TestAuditDB01_prefillTiming$' 1 GOINFER_METAL_AUDIT_A=1 GOINFER_AUDIT_MODEL="$Q35"
run speed-9b "$REPO/metal" "$MBIN" '^TestAuditDB01_prefillTiming$' 0 GOINFER_METAL_AUDIT_A=1 GOINFER_AUDIT_MODEL="$M9" \
  GOINFER_DB01_KS=128,512 GOINFER_AUDIT_REPS=5

{
  echo "-- preconditions"; grep -E '^--- (PASS|FAIL)' "$LOG/preconditions.log"
  echo "-- gate"; grep -E 'POOLED decision-set verdict|VOID|reference identity' "$LOG/gate.log" | head -3
  for m in 0.8b 9b; do echo "-- speed $m"; grep -E 'RESULT|load|refus|fit' "$LOG/speed-$m.log" | head -6; done
} | tee "$LOG/results.txt"
echo "finished: $(date '+%F %T %Z')" | tee -a "$LOG/provenance.txt"
[ $FAILED -eq 0 ] || { echo "a step failed: see $LOG"; exit 1; }
