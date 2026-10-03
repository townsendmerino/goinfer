#!/usr/bin/env bash
# Night job: Batch B's T1.7 (docs/tasks/task-metal-audit-2026-10.md, "Batch B: pre-registration"), pre-registered
# there on 2026-10-02 before any graded run. TestAuditT17_attnStaircase on the 1.5B and the 7B: in-sequence attention
# work per key at 1536..6144 keys in 128-key steps, production against a no-op attention (R17's method), 5 reps of 8
# tokens per arm per depth.
#
# Runs a test binary built at REV (untagged), so the tree may move before tonight:
#   (cd ~/tmcode/goinfer-metal-audit && go test -c -o ~/goinfer-bench/metal-audit-2026-10/metal-<REV>.test ./metal/)
# Queued with:
#   python3 scripts/night.py add metal-audit-t17 --est 25 --by "Claude (Mac session), Metal audit Batch B T1.7" \
#     --doc docs/tasks/task-metal-audit-2026-10.md -- bash ~/goinfer-bench/metal-audit-2026-10/run-t17.sh
# The runner holds the timing lock. Logs: ~/goinfer-logs/metal-audit-2026-10/batch-b/; RESULT/METRIC lines in t17-results.txt.
set -uo pipefail
REV=48d25e5c
BASE=$HOME/goinfer-bench/metal-audit-2026-10
BIN=$BASE/metal-$REV.test
LOG=$HOME/goinfer-logs/metal-audit-2026-10/batch-b
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
} | tee "$LOG/t17-provenance.txt"
probe() { # <log name> <model>
  echo "== $1 — $(date '+%T')" | tee -a "$LOG/t17-provenance.txt"
  GOINFER_METAL_AUDIT_B=1 GOINFER_AUDIT_MODEL="$2" "$BIN" -test.v -test.count=1 -test.timeout 60m \
    -test.run '^TestAuditT17_attnStaircase$' 2>&1 | tee "$LOG/$1.log" || true
}
probe t17-1.5b "$M15"
probe t17-7b "$M7"
grep -h 'RESULT\|METRIC\|step from\|^--- ' "$LOG"/t17-1.5b.log "$LOG"/t17-7b.log | tee "$LOG/t17-results.txt"
echo "finished: $(date '+%F %T %Z')" | tee -a "$LOG/t17-provenance.txt"
