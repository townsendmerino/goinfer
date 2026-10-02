#!/usr/bin/env bash
# Night job: E-P03's grade (docs/tasks/task-metal-audit-2026-10.md, "E-P03: built, pending its grade"), pre-registered
# there on 2026-10-02 before any graded run. TestEP03StepAB on the 7B and the 1.5B: the MC3 batched step at the FB
# calibrateFB chose against FB = 2 everywhere, in one process per model, B = 4 and 8 at depths 128 and 2048, 9 reps.
# The graded line is "E-P03 METRIC fb2/calibrated B=4 depth 128" on the 7B; the 1.5B is the A/A control.
#
# Runs a test binary built at REV (untagged), so the tree may move before tonight:
#   (cd ~/tmcode/goinfer-metal-audit && go test -c -o ~/goinfer-bench/metal-audit-2026-10/metal-<REV>.test ./metal/)
# Queued with:
#   python3 scripts/night.py add metal-audit-ep03 --est 35 --by "Claude (Mac session), Metal audit E-P03 grade" \
#     --doc docs/tasks/task-metal-audit-2026-10.md -- bash ~/goinfer-bench/metal-audit-2026-10/run-ep03-grade.sh
# The runner holds the timing lock. Logs: ~/goinfer-logs/metal-audit-2026-10/ep03/; METRIC lines in results.txt.
set -uo pipefail
REV=d9f2dba3
BASE=$HOME/goinfer-bench/metal-audit-2026-10
BIN=$BASE/metal-$REV.test
LOG=$HOME/goinfer-logs/metal-audit-2026-10/ep03
M7=$HOME/models/qwen2.5-7b-instruct-q4_k_m.int4.metal.giw
M15=$HOME/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf
mkdir -p "$LOG"
[ -x "$BIN" ] || { echo "missing test binary $BIN"; exit 1; }
for f in "$M7" "$M15"; do [ -f "$f" ] || { echo "missing model $f"; exit 1; }; done
cd "$BASE"
{
  echo "rev:      $REV"
  echo "binary:   $BIN (sha256 $(shasum -a 256 "$BIN" | cut -c1-16))"
  echo "started:  $(date '+%F %T %Z')"
  echo "machine:  $(sysctl -n machdep.cpu.brand_string), $(( $(sysctl -n hw.memsize) / 1073741824 )) GB, macOS $(sw_vers -productVersion)"
  echo "load:     $(uptime | sed 's/.*load averages*: //')"
  echo "therm:    $(pmset -g therm 2>/dev/null | tr '\n' ' ')"
} | tee "$LOG/provenance.txt"
grade() { # <log name> <model>
  echo "== $1 — $(date '+%T')" | tee -a "$LOG/provenance.txt"
  GOINFER_METAL_MC3=1 GOINFER_METAL_EP03=1 GOINFER_METAL_MC3_MODEL="$2" "$BIN" -test.v -test.count=1 -test.timeout 60m \
    -test.run '^TestEP03StepAB$' 2>&1 | tee "$LOG/$1.log" || true
}
grade 7b "$M7"
grade 1.5b "$M15"
grep -h 'calibrated FB\|chose FB\|METRIC\|batched step fragments\|^--- ' "$LOG"/7b.log "$LOG"/1.5b.log | tee "$LOG/results.txt"
echo "finished: $(date '+%F %T %Z')" | tee -a "$LOG/provenance.txt"
