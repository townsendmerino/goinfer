#!/usr/bin/env bash
# Night job: gates F3 (0.5B, then 1.5B), F2 (1.5B) and S of docs/tasks/task-metal-int8-2026-10.md, on an idle Mac.
# F3 could not run by day: the fit guard refused its f32 reference (the 0.5B at f32 priced 3.6 GB against a 2.2 GB
# budget with the owner's apps open), and a bypass on this 16 GB Mac is a standing no. A test that the fit guard still
# refuses tonight skips with the guard's numbers in its log. S is a timed A/B, so it is night-only by rule.
#
# Runs a test binary built at REV, so the tree may move before tonight:
#   go test -c -tags goinfer_testhooks -o ~/goinfer-bench/metal-int8-2026-10/metal-<REV>.test ./metal/
# Queued with:
#   python3 scripts/night.py add metal-int8-gates --est 30 --by "Claude, Metal int8 slice 1" \
#     --doc docs/tasks/task-metal-int8-2026-10.md -- bash ~/goinfer-bench/metal-int8-2026-10/run-gates.sh
# The runner holds the timing lock for the whole job. Logs: ~/goinfer-logs/metal-int8-2026-10/.
set -uo pipefail
REV=f73b980a
BASE=$HOME/goinfer-bench/metal-int8-2026-10
BIN=$BASE/metal-$REV.test
LOG=$HOME/goinfer-logs/metal-int8-2026-10
M15=$HOME/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf
mkdir -p "$LOG"
[ -x "$BIN" ] || { echo "missing test binary $BIN"; exit 1; }
cd "$BASE"
{
  echo "rev:      $REV"
  echo "binary:   $BIN (sha256 $(shasum -a 256 "$BIN" | cut -c1-16))"
  echo "started:  $(date '+%F %T %Z')"
  echo "machine:  $(sysctl -n machdep.cpu.brand_string), $(( $(sysctl -n hw.memsize) / 1073741824 )) GB, macOS $(sw_vers -productVersion)"
  echo "load:     $(uptime | sed 's/.*load averages*: //')"
} | tee "$LOG/provenance.txt"

step() { echo "== $1 — $(date '+%T')" | tee -a "$LOG/provenance.txt"; }

step "F3 on the 0.5B"
GOINFER_HEAVY_TESTS=1 "$BIN" -test.v -test.count=1 -test.timeout 30m -test.run '^TestW8Native_F3_closerToF32$' \
  2>&1 | tee "$LOG/f3-0.5b.log" || true
step "F3 on the 1.5B"
GOINFER_HEAVY_TESTS=1 GOINFER_W8_GATE_MODEL=$M15 "$BIN" -test.v -test.count=1 -test.timeout 30m \
  -test.run '^TestW8Native_F3_closerToF32$' 2>&1 | tee "$LOG/f3-1.5b.log" || true
step "F2 on the 1.5B"
GOINFER_HEAVY_TESTS=1 GOINFER_W8_GATE_MODEL=$M15 "$BIN" -test.v -test.count=1 -test.timeout 30m \
  -test.run '^TestW8Native_F2_matchesCPUAtSameQuant$' 2>&1 | tee "$LOG/f2-1.5b.log" || true
step "S, decode speed"
GOINFER_HEAVY_TESTS=1 GOINFER_W8_GATE_S=1 GOINFER_W8_GATE_OUT=$LOG/s-samples.jsonl "$BIN" -test.v -test.count=1 \
  -test.timeout 150m -test.run '^TestW8Native_S_decodeSpeed$' 2>&1 | tee "$LOG/s.log" || true
echo "finished: $(date '+%F %T %Z')" | tee -a "$LOG/provenance.txt"
