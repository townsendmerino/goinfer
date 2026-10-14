#!/usr/bin/env bash
# Night job: HL-G4, HL's speed grade on Metal (docs/tasks/task-forward-architecture-2026-10.md §6.1), registered there
# on 2026-10-14 before any timed run. TestHLChainAB on the 1.5B and the 7B, one process per model and shape: greedy
# Generate under a gated processor with the chain (HL) against greedyChainOff (today's path), 9 reps of 256 tokens per
# arm, alternated. The graded lines are "HL-G4 METRIC hl/today closed" (a) and "... toolcall" (b), on each model.
#
# Runs a test binary built at REV with -tags goinfer_testhooks, so the tree may move before the night:
#   (cd ~/tmcode/goinfer-hl && GOWORK=$PWD/go.work go test -c -tags goinfer_testhooks \
#     -o ~/goinfer-bench/hl-2026-10/metal-tagged-<REV>.test ./metal/)
# Queue with (about 25 minutes: 4 cells, 20 timed arms and 4 warm-up runs each, plus two loads per model):
#   python3 scripts/night.py add hl-g4-metal --est 25 --by "Claude (Mac session), HL on Metal, HL-G4" \
#     --doc docs/tasks/task-forward-architecture-2026-10.md -- bash ~/goinfer-bench/hl-2026-10/run-hl-g4.sh
# The runner holds the timing lock. Logs: ~/goinfer-logs/hl-2026-10/g4/; METRIC lines in results.txt.
set -uo pipefail
REV=e757d54c
BASE=$HOME/goinfer-bench/hl-2026-10
BIN=$BASE/metal-tagged-$REV.test
LOG=$HOME/goinfer-logs/hl-2026-10/g4
M15=$HOME/models/qwen2.5-coder-1.5b-instruct-q4_k_m.int4.metal.v16.giw
M7=$HOME/models/qwen2.5-7b-instruct-q4_k_m.int4.metal.v16.giw
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

grade() { # <log name> <model> <shape>
  echo "== $1 — $(date '+%T')" | tee -a "$LOG/provenance.txt"
  GOINFER_METAL_HL_AB=1 GOINFER_METAL_HL_MODEL="$2" GOINFER_METAL_HL_SHAPE="$3" \
    "$BIN" -test.v -test.count=1 -test.timeout 30m -test.run '^TestHLChainAB$' 2>&1 | tee "$LOG/$1.log" || true
}

grade 1.5b-closed "$M15" closed
grade 1.5b-toolcall "$M15" toolcall
grade 7b-closed "$M7" closed
grade 7b-toolcall "$M7" toolcall

grep -h 'identity:\|METRIC\|loaded\|^--- ' "$LOG"/1.5b-closed.log "$LOG"/1.5b-toolcall.log "$LOG"/7b-closed.log \
  "$LOG"/7b-toolcall.log | tee "$LOG/results.txt"
echo "finished: $(date '+%F %T %Z')" | tee -a "$LOG/provenance.txt"
