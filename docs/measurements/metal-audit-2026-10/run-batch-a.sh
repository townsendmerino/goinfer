#!/usr/bin/env bash
# Night job: Phase 2, Batch A of docs/tasks/task-metal-audit-2026-10.md (T1.1, T1.2, T1.3, T1.4, T1.10's timing),
# pre-registered in that doc's "Batch A: pre-registration" table before this ran. Timed, so night-only by rule.
#
# Runs a test binary built at REV from the metal-audit worktree, so the tree may move before tonight:
#   (cd ~/tmcode/goinfer-metal-audit && go test -c -o ~/goinfer-bench/metal-audit-2026-10/metal-<REV>.test ./metal/)
# Queued with:
#   python3 scripts/night.py add metal-audit-batch-a --est 20 --by "Claude, Metal audit Phase 2" \
#     --doc docs/tasks/task-metal-audit-2026-10.md -- bash ~/goinfer-bench/metal-audit-2026-10/run-batch-a.sh
# The runner holds the timing lock for the whole job. Logs: ~/goinfer-logs/metal-audit-2026-10/batch-a/; the RESULT
# lines are collected in results.txt there.
set -uo pipefail
REV=cfce51e1
BASE=$HOME/goinfer-bench/metal-audit-2026-10
BIN=$BASE/metal-$REV.test
LOG=$HOME/goinfer-logs/metal-audit-2026-10/batch-a
M05=$HOME/models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf
M15=$HOME/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf
M7=$HOME/models/qwen2.5-7b-instruct-q4_k_m.int4.metal.giw
mkdir -p "$LOG"
[ -x "$BIN" ] || { echo "missing test binary $BIN"; exit 1; }
for f in "$M05" "$M15" "$M7"; do [ -f "$f" ] || { echo "missing model $f"; exit 1; }; done
cd "$BASE"
{
  echo "rev:      $REV"
  echo "binary:   $BIN (sha256 $(shasum -a 256 "$BIN" | cut -c1-16))"
  echo "started:  $(date '+%F %T %Z')"
  echo "machine:  $(sysctl -n machdep.cpu.brand_string), $(( $(sysctl -n hw.memsize) / 1073741824 )) GB, macOS $(sw_vers -productVersion)"
  echo "load:     $(uptime | sed 's/.*load averages*: //')"
} | tee "$LOG/provenance.txt"

step() { echo "== $1 — $(date '+%T')" | tee -a "$LOG/provenance.txt"; }

# probe <log name> <model or ""> <test>: one process per probe and model, so each load starts from a clean heap.
probe() {
  step "$1"
  GOINFER_METAL_AUDIT_A=1 GOINFER_AUDIT_MODEL="$2" "$BIN" -test.v -test.count=1 -test.timeout 20m -test.run "^$3\$" \
    2>&1 | tee "$LOG/$1.log" || true
}

probe t1.1-1.5b "$M15" TestAuditT11_passCost
probe t1.2-1.5b "$M15" TestAuditT12_blkBelowFloor
probe t1.2-7b "$M7" TestAuditT12_blkBelowFloor
probe t1.3-1.5b "$M15" TestAuditT13_kernelGap
probe t1.3-0.5b "$M05" TestAuditT13_kernelGap
probe t1.4 "" TestAuditT14_saRows4PerByte
probe t1.10-1.5b "$M15" TestAuditT110_promptInRowsTiming
probe t1.10-7b "$M7" TestAuditT110_promptInRowsTiming

grep -h 'RESULT' "$LOG"/t1.*.log | tee "$LOG/results.txt"
echo "finished: $(date '+%F %T %Z')" | tee -a "$LOG/provenance.txt"
