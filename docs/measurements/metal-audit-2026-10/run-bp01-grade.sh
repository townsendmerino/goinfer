#!/usr/bin/env bash
# Night job: B-P01's grade (docs/tasks/task-metal-audit-2026-10.md, "B-P01: built, pending its grade"), pre-registered
# there on 2026-10-03 before any graded run. Four steps, each from a pinned tagged test binary, each run from its
# package directory (the tests read ../testdata relative to it):
#   1. the 0.5B's CPU f32 reference cell Q05 at K = 2048 and 3900, prompt set B (decoder TestPrefillGateReference)
#   2. the fidelity gate on the 0.5B against Q05: K = 3900 candidate and exact-null, K = 2048 candidate
#   3. P1, TestR17KernelAccuracy (arms bp03), 10 prompts at 2048 and 3900
#   4. speed, TestBP01AttnAB, 5 reps
# GOINFER_METAL_BLK64=1 turns the hd = 64 twin on in the metal binary (attnFABlk64On, off in production).
#
# Binaries, built at REV:
#   (cd ~/tmcode/goinfer && go test -c -tags goinfer_testhooks -o ~/goinfer-bench/metal-audit-2026-10/decoder-tagged-<REV>.test ./decoder/ \
#     && go test -c -tags goinfer_testhooks -o ~/goinfer-bench/metal-audit-2026-10/metal-tagged-<REV>.test ./metal/)
# Queued with:
#   python3 scripts/night.py add metal-audit-bp01 --est 75 --by "Claude (Mac session), Metal audit B-P01 grade" \
#     --doc docs/tasks/task-metal-audit-2026-10.md -- bash docs/measurements/metal-audit-2026-10/run-bp01-grade.sh
# The runner holds the timing lock. Logs: ~/goinfer-logs/metal-audit-2026-10/bp01/.
set -uo pipefail
REV=5103bd40
BASE=$HOME/goinfer-bench/metal-audit-2026-10
DBIN=$BASE/decoder-tagged-$REV.test
MBIN=$BASE/metal-tagged-$REV.test
REPO=$HOME/tmcode/goinfer
LOG=$HOME/goinfer-logs/metal-audit-2026-10/bp01
M05=$HOME/models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf
REF=$HOME/goinfer-logs/prefill-ref-b
mkdir -p "$LOG"
for f in "$DBIN" "$MBIN"; do [ -x "$f" ] || { echo "missing test binary $f"; exit 1; }; done
[ -f "$M05" ] || { echo "missing model $M05"; exit 1; }
[ -f "$REPO/testdata/prefill-gate-prose-b/task-attention-decode-cost.md" ] || { echo "missing the prose seeds under $REPO/testdata"; exit 1; }
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

run() { # <log name> <dir> <binary> <test regexp> ENV=VALUE...
  local name=$1 dir=$2 bin=$3 re=$4
  shift 4
  echo "== $name — $(date '+%T')" | tee -a "$LOG/provenance.txt"
  (cd "$dir" && env GOINFER_PREFILL_GATE_PROMPTS=b "$@" "$bin" -test.v -test.count=1 -test.timeout 90m -test.run "$re") > "$LOG/$name.log" 2>&1
  local rc=$?
  echo "$name rc=$rc" | tee -a "$LOG/provenance.txt"
  [ $rc -eq 0 ] || FAILED=1
  grep -q -- '--- SKIP' "$LOG/$name.log" && { echo "$name: a test SKIPPED (a skip is not a pass)" | tee -a "$LOG/provenance.txt"; FAILED=1; }
}

run refs "$REPO/decoder" "$DBIN" '^TestPrefillGateReference$' GOINFER_HEAVY_TESTS=1 GOINFER_CPU_REF_MODELS=Q05 GOINFER_CPU_REF_KS=2048,3900
n=$(ls "$REF"/Q05-K2048-p*.bin "$REF"/Q05-K3900-p*.bin 2>/dev/null | grep -c '\.bin$')
echo "Q05 reference files: $n of 20" | tee -a "$LOG/provenance.txt"
[ "$n" -eq 20 ] || { echo "references incomplete: stopping" | tee -a "$LOG/provenance.txt"; exit 1; }

GATE="GOINFER_METAL_BLK64=1 GOINFER_HEAVY_TESTS=1 GOINFER_METAL_GATE_MODEL=$M05 GOINFER_METAL_GATE_CELL=Q05"
run gate-3900 "$REPO/metal" "$MBIN" '^TestR17_decodeFidelityGate$' $GATE GOINFER_METAL_GATE_K=3900 GOINFER_METAL_R17_CAND=attention_fa
run gate-3900-null "$REPO/metal" "$MBIN" '^TestR17_decodeFidelityGate$' $GATE GOINFER_METAL_GATE_K=3900 GOINFER_METAL_R17_CAND=exact-null
run gate-2048 "$REPO/metal" "$MBIN" '^TestR17_decodeFidelityGate$' $GATE GOINFER_METAL_GATE_K=2048 GOINFER_METAL_R17_CAND=attention_fa
run p1 "$REPO/metal" "$MBIN" '^TestR17KernelAccuracy$' GOINFER_METAL_BLK64=1 GOINFER_METAL_R17=1 GOINFER_METAL_R17_ACC_ARMS=bp03 GOINFER_METAL_R17_PROMPTS=10 GOINFER_METAL_R17_DEPTHS=2048,3900 GOINFER_METAL_R17_MODEL="$M05"
run speed "$REPO/metal" "$MBIN" '^TestBP01AttnAB$' GOINFER_METAL_BP01=1 GOINFER_AUDIT_REPS=5

{
  echo "-- p1"; grep -E '^=== R17 kernel accuracy|^  (exact \(shipped attention\)|production \(r\.pAttnFA\)) +[0-9]' "$LOG/p1.log" | head -3
  for f in gate-3900 gate-3900-null gate-2048; do echo "-- $f"; grep -E 'SUSPECTED|verdict|KL ratio|critA|critB|ceiling|floor overridden' "$LOG/$f.log" | tail -8; done
  echo "-- speed"; grep -E 'METRIC|keys: legacy' "$LOG/speed.log"
} | tee "$LOG/results.txt"
echo "finished: $(date '+%F %T %Z')" | tee -a "$LOG/provenance.txt"
[ $FAILED -eq 0 ] || { echo "a step failed: see $LOG"; exit 1; }
