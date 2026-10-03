#!/usr/bin/env bash
# Night job: B-P04's grade (docs/tasks/task-metal-audit-2026-10.md, "B-P04: built, pending its grade"), pre-registered
# there on 2026-10-02 before any graded run. TestR18InSequence (R18's grading instrument) on the 1.5B and the 7B at
# depths 128 and 2048, 7 reps (the harness default is 5), and its defaults otherwise (20 step pairs per category, 16
# identity tokens, 3 after-idle samples). Its arms: shipped (one row per simdgroup), production (the lane-balanced SA rows kernels,
# B-P04) and pre-bp04 (the same rows per simdgroup on the kernels B-P04 replaced). The graded line is
# "B-P04 METRIC pre-bp04/production" on the 1.5B at depth 128.
#
# Runs a test binary built at REV with -tags goinfer_testhooks, so the tree may move before tonight:
#   (cd ~/tmcode/goinfer-metal-audit && go test -c -tags goinfer_testhooks \
#     -o ~/goinfer-bench/metal-audit-2026-10/metal-tagged-<REV>.test ./metal/)
# Queued with:
#   python3 scripts/night.py add metal-audit-bp04 --est 25 --by "Claude, Metal audit B-P04 grade" \
#     --doc docs/tasks/task-metal-audit-2026-10.md -- bash ~/goinfer-bench/metal-audit-2026-10/run-bp04-grade.sh
# The runner holds the timing lock. Logs: ~/goinfer-logs/metal-audit-2026-10/bp04/; METRIC lines in results.txt.
set -uo pipefail
REV=22819cc8
BASE=$HOME/goinfer-bench/metal-audit-2026-10
BIN=$BASE/metal-tagged-$REV.test
LOG=$HOME/goinfer-logs/metal-audit-2026-10/bp04
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

grade() { # <log name> <model>
  echo "== $1 — $(date '+%T')" | tee -a "$LOG/provenance.txt"
  GOINFER_METAL_R18_SEQ=1 GOINFER_METAL_R18_MODEL="$2" GOINFER_METAL_R18_DEPTHS=128,2048 GOINFER_METAL_R18_REPS=7 \
    "$BIN" -test.v -test.count=1 -test.timeout 60m -test.run '^TestR18InSequence$' 2>&1 | tee "$LOG/$1.log" || true
}

grade 1.5b "$M15"
grade 7b "$M7"

grep -h 'identity:\|METRIC\|^\[r18-seq.*loaded' "$LOG"/1.5b.log "$LOG"/7b.log | tee "$LOG/results.txt"
echo "finished: $(date '+%F %T %Z')" | tee -a "$LOG/provenance.txt"
