#!/usr/bin/env bash
# Night job: D-G01's grade (docs/tasks/task-metal-audit-2026-10.md, "D-G01: pre-registration"), pre-registered there on
# 2026-10-03 before any graded run. From the tagged metal test binary pinned at REV, run from metal/ (the tests read
# ../testdata): TestDG01_expertMajorMoEPrefill on the Qwen1.5-MoE 4-layer slice, M = 64 and 512, 10 set-A prompts.
#   1. the three planted mutations, each of which must FAIL the gate (a mutation that passes stops the grade)
#   2. the gate itself, no mutation
#
# Binary, built at REV:
#   (cd ~/tmcode/goinfer && go test -c -tags goinfer_testhooks -o ~/goinfer-bench/metal-audit-2026-10/metal-tagged-<REV>.test ./metal/)
# Queued with:
#   python3 scripts/night.py add metal-audit-dg01 --est 15 --by "Claude (Mac session), Metal audit D-G01 gate" \
#     --doc docs/tasks/task-metal-audit-2026-10.md -- bash docs/measurements/metal-audit-2026-10/run-dg01-grade.sh
# Logs: ~/goinfer-logs/metal-audit-2026-10/dg01/; the readings in results.txt.
set -uo pipefail
REV=ec3cafd1
BASE=$HOME/goinfer-bench/metal-audit-2026-10
MBIN=$BASE/metal-tagged-$REV.test
REPO=$HOME/tmcode/goinfer
LOG=$HOME/goinfer-logs/metal-audit-2026-10/dg01
SLICE=$HOME/models/qwen15-moe-a27b-l4slice
mkdir -p "$LOG"
[ -x "$MBIN" ] || { echo "missing test binary $MBIN"; exit 1; }
[ -f "$SLICE/model.safetensors" ] || { echo "missing the slice $SLICE"; exit 1; }
[ -d "$REPO/testdata/prefill-gate-prose-a" ] || { echo "missing the set-A prose seeds under $REPO/testdata"; exit 1; }
{
  echo "rev:      $REV"
  echo "binary:   $MBIN (sha256 $(shasum -a 256 "$MBIN" | cut -c1-16))"
  echo "fixture:  $SLICE (model.safetensors sha256 $(shasum -a 256 "$SLICE/model.safetensors" | cut -c1-16))"
  echo "started:  $(date '+%F %T %Z')"
  echo "machine:  $(sysctl -n machdep.cpu.brand_string), $(( $(sysctl -n hw.memsize) / 1073741824 )) GB, macOS $(sw_vers -productVersion)"
  echo "load:     $(uptime | sed 's/.*load averages*: //')"
} | tee "$LOG/provenance.txt"

gate() { # <name> [mutation]
  echo "== $1 — $(date '+%T')" | tee -a "$LOG/provenance.txt"
  (cd "$REPO/metal" && env GOINFER_DG01=1 GOINFER_DG01_MODEL="$SLICE" GOINFER_DG01_MUTATION="${2:-}" \
    "$MBIN" -test.v -test.count=1 -test.timeout 60m -test.run '^TestDG01_expertMajorMoEPrefill$') > "$LOG/$1.log" 2>&1
  echo "$1 rc=$?" | tee -a "$LOG/provenance.txt"
  grep -q -- '--- SKIP' "$LOG/$1.log" && { echo "$1: SKIPPED (a skip is not a pass): stopping" | tee -a "$LOG/provenance.txt"; exit 1; }
  grep -q 'VERDICT' "$LOG/$1.log" || { echo "$1: no verdict (the instrument failed): stopping" | tee -a "$LOG/provenance.txt"; exit 1; }
}

for mu in topk-renorm rank-swap shared-off; do
  gate "mutation-$mu" "$mu"
  if grep -q 'VERDICT .*the gate PASSES' "$LOG/mutation-$mu.log"; then
    echo "mutation $mu PASSED the gate: the gate is not evidence, stopping before the graded run" | tee -a "$LOG/provenance.txt"
    exit 1
  fi
done
gate gate

grep -h -E 'RESULT|VERDICT' "$LOG"/mutation-*.log "$LOG/gate.log" | sed -E 's/^\[dg01 +[0-9.]+s\] //' | tee "$LOG/results.txt"
echo "finished: $(date '+%F %T %Z')" | tee -a "$LOG/provenance.txt"
