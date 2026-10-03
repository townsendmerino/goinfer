#!/usr/bin/env bash
# Night job: B-P02's grade (docs/tasks/task-metal-audit-2026-10.md, "B-P02: built, pending its grade"), pre-registered
# there on 2026-10-03 before any graded run. TestBP02AttnAB on internlm2-1_8b (graded) and Qwen3-0.6B (reported), G = 2:
# in-sequence attention, legacy attention_fa against the block kernel, 5 reps.
#
# Runs a tagged test binary built at REV from the metal package directory:
#   (cd ~/tmcode/goinfer && go test -c -tags goinfer_testhooks -o ~/goinfer-bench/metal-audit-2026-10/metal-tagged-<REV>.test ./metal/)
# Queued with:
#   python3 scripts/night.py add metal-audit-bp02 --est 20 --by "Claude (Mac session), Metal audit B-P02 grade" \
#     --doc docs/tasks/task-metal-audit-2026-10.md -- bash docs/measurements/metal-audit-2026-10/run-bp02-grade.sh
# The runner holds the timing lock. Logs: ~/goinfer-logs/metal-audit-2026-10/bp02/.
set -uo pipefail
REV=39b893d5
BASE=$HOME/goinfer-bench/metal-audit-2026-10
BIN=$BASE/metal-tagged-$REV.test
REPO=$HOME/tmcode/goinfer
LOG=$HOME/goinfer-logs/metal-audit-2026-10/bp02
mkdir -p "$LOG"
[ -x "$BIN" ] || { echo "missing test binary $BIN"; exit 1; }
for d in internlm2-1_8b qwen3-0.6b-bf16; do [ -f "$HOME/models/$d/config.json" ] || { echo "missing model $d"; exit 1; }; done
FAILED=0
{
  echo "rev:      $REV"
  echo "binary:   $BIN (sha256 $(shasum -a 256 "$BIN" | cut -c1-16))"
  echo "started:  $(date '+%F %T %Z')"
  echo "machine:  $(sysctl -n machdep.cpu.brand_string), $(( $(sysctl -n hw.memsize) / 1073741824 )) GB, macOS $(sw_vers -productVersion)"
  echo "load:     $(uptime | sed 's/.*load averages*: //')"
  echo "therm:    $(pmset -g therm 2>/dev/null | tr '\n' ' ')"
} | tee "$LOG/provenance.txt"
for d in internlm2-1_8b qwen3-0.6b-bf16; do
  echo "== $d — $(date '+%T')" | tee -a "$LOG/provenance.txt"
  (cd "$REPO/metal" && GOINFER_METAL_BP02=1 GOINFER_AUDIT_REPS=5 GOINFER_AUDIT_MODEL="$HOME/models/$d" \
    "$BIN" -test.v -test.count=1 -test.timeout 60m -test.run '^TestBP02AttnAB$') > "$LOG/$d.log" 2>&1
  rc=$?
  echo "$d rc=$rc" | tee -a "$LOG/provenance.txt"
  [ $rc -eq 0 ] || FAILED=1
  grep -q -- '--- SKIP' "$LOG/$d.log" && { echo "$d: SKIPPED (a skip is not a pass)" | tee -a "$LOG/provenance.txt"; FAILED=1; }
done
grep -h 'METRIC\|keys: legacy\|G = ' "$LOG"/internlm2-1_8b.log "$LOG"/qwen3-0.6b-bf16.log | tee "$LOG/results.txt"
echo "finished: $(date '+%F %T %Z')" | tee -a "$LOG/provenance.txt"
[ $FAILED -eq 0 ] || { echo "a step failed: see $LOG"; exit 1; }
