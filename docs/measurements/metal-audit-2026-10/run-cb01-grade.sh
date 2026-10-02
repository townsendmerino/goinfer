#!/usr/bin/env bash
# Night job: the grades of C-B01 and C-P02 (docs/tasks/task-metal-audit-2026-10.md, "C-B01: built, pending its grade"
# and "C-P02: built, pending its grade"), pre-registered there on 2026-10-02 before any graded run. TestCB01ChainAB on
# the 1.5B, the 0.5B and the 7B, one process per model and mode: Generate with the chain against the same request with
# it off, 9 reps of 256 tokens per arm, alternated, greedy (C-B01) then temperature-only at T=1.0 (C-P02). The graded
# lines are "C-B01 METRIC chain/off greedy" and "... sampled T=1" on the 1.5B.
#
# Runs a test binary built at REV with -tags goinfer_testhooks, so the tree may move before tonight:
#   (cd ~/tmcode/goinfer-metal-audit && go test -c -tags goinfer_testhooks \
#     -o ~/goinfer-bench/metal-audit-2026-10/metal-tagged-<REV>.test ./metal/)
# Queued with:
#   python3 scripts/night.py add metal-audit-cb01 --est 30 --by "Claude (Mac session), Metal audit C-B01 and C-P02 grades" \
#     --doc docs/tasks/task-metal-audit-2026-10.md -- bash ~/goinfer-bench/metal-audit-2026-10/run-cb01-grade.sh
# The runner holds the timing lock. Logs: ~/goinfer-logs/metal-audit-2026-10/cb01/; METRIC lines in results.txt.
set -uo pipefail
REV=d566889b
BASE=$HOME/goinfer-bench/metal-audit-2026-10
BIN=$BASE/metal-tagged-$REV.test
LOG=$HOME/goinfer-logs/metal-audit-2026-10/cb01
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
  echo "therm:    $(pmset -g therm 2>/dev/null | tr '\n' ' ')"
} | tee "$LOG/provenance.txt"

grade() { # <log name> <model> [temperature]
  echo "== $1 — $(date '+%T')" | tee -a "$LOG/provenance.txt"
  GOINFER_METAL_CB01_AB=1 GOINFER_METAL_CB01_MODEL="$2" GOINFER_METAL_CB01_TEMP="${3:-}" \
    "$BIN" -test.v -test.count=1 -test.timeout 30m -test.run '^TestCB01ChainAB$' 2>&1 | tee "$LOG/$1.log" || true
}

grade 1.5b "$M15"
grade 1.5b-t1 "$M15" 1.0
grade 0.5b "$M05"
grade 0.5b-t1 "$M05" 1.0
grade 7b "$M7"
grade 7b-t1 "$M7" 1.0

grep -h 'identity:\|METRIC\|loaded\|^--- ' "$LOG"/1.5b.log "$LOG"/1.5b-t1.log "$LOG"/0.5b.log "$LOG"/0.5b-t1.log \
  "$LOG"/7b.log "$LOG"/7b-t1.log | tee "$LOG/results.txt"
echo "finished: $(date '+%F %T %Z')" | tee -a "$LOG/provenance.txt"
