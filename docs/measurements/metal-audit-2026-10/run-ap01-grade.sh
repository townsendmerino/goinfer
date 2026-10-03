#!/usr/bin/env bash
# Night job: A-P01's grade (docs/tasks/task-metal-audit-2026-10.md, "A-P01: built, pending its grade"), pre-registered
# there on 2026-10-02 before any graded run. From a test binary pinned at REV: the kernel identity gate first
# (TestGemmTile_bitIdentical, any difference kills), then on the 1.5B and the 7B TestAuditAP01_gemmSmallM (each prefill
# GEMM alone, shipped tile against the selector's, M = 16..64) and TestAuditAP01_passCost (PrefillLast at startPos 64
# and 2048, C = 16..64, under each gemmTilePolicy: shipped, bm32, bn32, both), 7 reps, interleaved.
#
# Build: (cd ~/tmcode/goinfer-metal-audit && go test -c -o ~/goinfer-bench/metal-audit-2026-10/metal-<REV>.test ./metal/)
# Queued with:
#   python3 scripts/night.py add metal-audit-ap01 --est 15 --by "Claude, Metal audit A-P01 grade" \
#     --doc docs/tasks/task-metal-audit-2026-10.md -- bash ~/goinfer-bench/metal-audit-2026-10/run-ap01-grade.sh
# The runner holds the timing lock. Logs: ~/goinfer-logs/metal-audit-2026-10/ap01/; RESULT lines in results.txt.
set -uo pipefail
REV=e8c1e13a
BASE=$HOME/goinfer-bench/metal-audit-2026-10
BIN=$BASE/metal-$REV.test
LOG=$HOME/goinfer-logs/metal-audit-2026-10/ap01
M15=$HOME/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf
M7=$HOME/models/qwen2.5-7b-instruct-q4_k_m.int4.metal.giw
mkdir -p "$LOG"
[ -x "$BIN" ] || { echo "missing test binary $BIN"; exit 1; }
for f in "$M15" "$M7"; do [ -f "$f" ] || { echo "missing model $f"; exit 1; }; done
cd "$BASE"
{
  echo "rev:      $REV"
  echo "binary:   $BIN (sha256 $(shasum -a 256 "$BIN" | cut -c1-16))"
  echo "started:  $(date '+%F %T %Z')"
  echo "machine:  $(sysctl -n machdep.cpu.brand_string), $(( $(sysctl -n hw.memsize) / 1073741824 )) GB, macOS $(sw_vers -productVersion)"
  echo "load:     $(uptime | sed 's/.*load averages*: //')"
  echo "therm:    $(pmset -g therm 2>/dev/null | tr '\n' ' ')"
} | tee "$LOG/provenance.txt"

probe() { # <log name> <model or ""> <test regexp> [AUDIT env]
  echo "== $1 — $(date '+%T')" | tee -a "$LOG/provenance.txt"
  env "${@:4}" GOINFER_AUDIT_MODEL="$2" "$BIN" -test.v -test.count=1 -test.timeout 30m -test.run "$3" 2>&1 | tee "$LOG/$1.log" || true
}

probe identity "" '^TestGemmTile_bitIdentical$'
probe gemm-1.5b "$M15" '^TestAuditAP01_gemmSmallM$' GOINFER_METAL_AUDIT_A=1
probe pass-1.5b "$M15" '^TestAuditAP01_passCost$' GOINFER_METAL_AUDIT_A=1
probe gemm-7b "$M7" '^TestAuditAP01_gemmSmallM$' GOINFER_METAL_AUDIT_A=1
probe pass-7b "$M7" '^TestAuditAP01_passCost$' GOINFER_METAL_AUDIT_A=1

grep -h -- '--- \(PASS\|FAIL\): TestGemmTile_bitIdentical\|differs from' "$LOG/identity.log" | tee "$LOG/results.txt"
grep -h 'RESULT\|startPos' "$LOG"/gemm-*.log "$LOG"/pass-*.log | tee -a "$LOG/results.txt"
echo "finished: $(date '+%F %T %Z')" | tee -a "$LOG/provenance.txt"
