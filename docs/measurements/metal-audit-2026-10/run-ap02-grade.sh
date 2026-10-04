#!/usr/bin/env bash
# Night job: A-P02's grade (docs/tasks/task-metal-audit-2026-10.md, "A-P02: pre-registration"), pre-registered there on
# 2026-10-03 before any graded run. Three steps, each from a pinned tagged test binary run from its package directory
# (the tests read ../testdata relative to it):
#   1. the 1.5B's CPU f32 references, set A, at K = 16, 32, 48 and 64 (decoder TestPrefillGateReference); R3's K = 64
#      files are moved aside first so all four cells come from one generation
#   2. the §3.2 pooled fidelity gate on the 1.5B, set A, confirmation cells off, once per candidate pool and per single
#      cell. A pool that does not ship exits non-zero: that is a verdict, recorded, not a failed step
#   3. TestAuditAP02_shortPromptTiming on the 1.5B and the 7B, 7 reps
#
# Binaries, built at REV:
#   (cd ~/tmcode/goinfer && go test -c -tags goinfer_testhooks -o ~/goinfer-bench/metal-audit-2026-10/decoder-tagged-<REV>.test ./decoder/ \
#     && go test -c -tags goinfer_testhooks -o ~/goinfer-bench/metal-audit-2026-10/metal-tagged-<REV>.test ./metal/)
# Queued with:
#   python3 scripts/night.py add metal-audit-ap02 --est 30 --by "Claude (Mac session), Metal audit A-P02 grade" \
#     --doc docs/tasks/task-metal-audit-2026-10.md -- bash docs/measurements/metal-audit-2026-10/run-ap02-grade.sh
# The runner holds the timing lock. Logs: ~/goinfer-logs/metal-audit-2026-10/ap02/; the readings in results.txt.
set -uo pipefail
REV=f81f3a18
BASE=$HOME/goinfer-bench/metal-audit-2026-10
DBIN=$BASE/decoder-tagged-$REV.test
MBIN=$BASE/metal-tagged-$REV.test
REPO=$HOME/tmcode/goinfer
LOG=$HOME/goinfer-logs/metal-audit-2026-10/ap02
M15=$HOME/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf
M7=$HOME/models/qwen2.5-7b-instruct-q4_k_m.int4.metal.giw
REF=$HOME/goinfer-logs/prefill-ref
mkdir -p "$LOG"
for f in "$DBIN" "$MBIN"; do [ -x "$f" ] || { echo "missing test binary $f"; exit 1; }; done
for f in "$M15" "$M7"; do [ -f "$f" ] || { echo "missing model $f"; exit 1; }; done
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
}

# 1. references
if ls "$REF"/S-K64-p*.bin >/dev/null 2>&1; then
  mkdir -p "$HOME/goinfer-logs/prefill-ref-stale-2026-09-20-K64"
  mv "$REF"/S-K64-p*.bin "$HOME/goinfer-logs/prefill-ref-stale-2026-09-20-K64/"
fi
run refs "$REPO/decoder" "$DBIN" '^TestPrefillGateReference$' 1 GOINFER_HEAVY_TESTS=1 GOINFER_CPU_REF_MODELS=S GOINFER_CPU_REF_KS=16,32,48,64
n=$(ls "$REF"/S-K16-p*.bin "$REF"/S-K32-p*.bin "$REF"/S-K48-p*.bin "$REF"/S-K64-p*.bin 2>/dev/null | grep -c '\.bin$')
echo "S reference files: $n of 40" | tee -a "$LOG/provenance.txt"
[ "$n" -eq 40 ] || { echo "references incomplete: stopping, nothing graded" | tee -a "$LOG/provenance.txt"; exit 1; }

# 2. fidelity: the three candidate pools, then each new cell alone
for ks in 16,32,48,64 32,48,64 48,64 16 32 48; do
  run "gate-$ks" "$REPO/metal" "$MBIN" '^TestPrefillGateVsReference$/^S$' 0 \
    GOINFER_HEAVY_TESTS=1 GOINFER_METAL_GATE_CONFIRM=0 GOINFER_METAL_GATE_DECISION_KS=$ks GOINFER_METAL_MODEL="$M15"
done

# 3. speed
run speed-1.5b "$REPO/metal" "$MBIN" '^TestAuditAP02_shortPromptTiming$' 1 GOINFER_METAL_AUDIT_A=1 GOINFER_AUDIT_MODEL="$M15"
run speed-7b "$REPO/metal" "$MBIN" '^TestAuditAP02_shortPromptTiming$' 1 GOINFER_METAL_AUDIT_A=1 GOINFER_AUDIT_MODEL="$M7"

{
  for ks in 16,32,48,64 32,48,64 48,64 16 32 48; do
    echo "-- gate K=$ks"; grep -E 'POOLED decision-set verdict|VOID|reference identity' "$LOG/gate-$ks.log" | head -3
  done
  for m in 1.5b 7b; do echo "-- speed $m"; grep -E 'RESULT' "$LOG/speed-$m.log"; done
} | tee "$LOG/results.txt"
echo "finished: $(date '+%F %T %Z')" | tee -a "$LOG/provenance.txt"
[ $FAILED -eq 0 ] || { echo "a step failed: see $LOG"; exit 1; }
