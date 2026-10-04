#!/usr/bin/env bash
# Night job: the Metal int8 gates after the owner's 2026-10-04 decisions (docs/tasks/task-metal-int8-2026-10.md, "F3′"):
# F3′ on the 1.5B (it skips with the guard's numbers if the fit guard refuses its f32 reference; no bypass on this Mac),
# F2 on the 1.5B with precise math (its bar unchanged), and S with the fast-math arm that prices precise math. F3′ on the
# 0.5B ran and passed by day.
#
# Runs a test binary built at REV from a clean tree, from a worktree pinned at REV: F3′ reads the prefill gate's prompt
# snapshots by a path relative to the package directory (../testdata/prefill-gate-prose-a/).
#   go test -c -tags goinfer_testhooks -o ~/goinfer-bench/metal-int8-2026-10/metal-<REV>.test ./metal/
#   git worktree add --detach ~/goinfer-bench/metal-int8-2026-10/wt-<REV> <REV>
# Queued with:
#   python3 scripts/night.py add metal-int8-gates2 --est 30 --by "Claude, Metal int8 F3′" \
#     --doc docs/tasks/task-metal-int8-2026-10.md -- bash docs/measurements/metal-int8-2026-10/run-gates2.sh
# The runner holds the timing lock for the whole job. Logs: ~/goinfer-logs/metal-int8-2026-10/gates2/.
set -uo pipefail
REV=351a3f63
BASE=$HOME/goinfer-bench/metal-int8-2026-10
BIN=$BASE/metal-$REV.test
WT=$BASE/wt-$REV
LOG=$HOME/goinfer-logs/metal-int8-2026-10/gates2
M15=$HOME/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf
mkdir -p "$LOG"
[ -x "$BIN" ] || { echo "missing test binary $BIN"; exit 1; }
[ -d "$WT/testdata/prefill-gate-prose-a" ] || { echo "missing worktree $WT"; exit 1; }
cd "$WT/metal"
{
  echo "rev:      $REV"
  echo "binary:   $BIN (sha256 $(shasum -a 256 "$BIN" | cut -c1-16))"
  echo "started:  $(date '+%F %T %Z')"
  echo "machine:  $(sysctl -n machdep.cpu.brand_string), $(( $(sysctl -n hw.memsize) / 1073741824 )) GB, macOS $(sw_vers -productVersion)"
  echo "load:     $(uptime | sed 's/.*load averages*: //')"
} | tee "$LOG/provenance.txt"

rc=0
step() { echo "== $1 — $(date '+%T')" | tee -a "$LOG/provenance.txt"; }
run() { # name, then the test binary's arguments; a failing gate is recorded, and the job's exit code says so
  local name=$1; shift
  "$@" 2>&1 | tee "$LOG/$name.log"
  local s=${PIPESTATUS[0]}
  echo "$name exit $s" | tee -a "$LOG/provenance.txt"
  [ "$s" -eq 0 ] || rc=1
}

step "F3′ on the 1.5B"
run f3amended-1.5b env GOINFER_HEAVY_TESTS=1 GOINFER_W8_GATE_MODEL=$M15 "$BIN" -test.v -test.count=1 -test.timeout 40m \
  -test.run '^TestW8Native_F3amended_closerToF32$'
step "F2 on the 1.5B"
run f2-1.5b env GOINFER_HEAVY_TESTS=1 GOINFER_W8_GATE_MODEL=$M15 "$BIN" -test.v -test.count=1 -test.timeout 30m \
  -test.run '^TestW8Native_F2_matchesCPUAtSameQuant$'
step "S, decode speed (four arms)"
run s env GOINFER_HEAVY_TESTS=1 GOINFER_W8_GATE_S=1 GOINFER_W8_GATE_OUT=$LOG/s-samples.jsonl "$BIN" -test.v -test.count=1 \
  -test.timeout 150m -test.run '^TestW8Native_S_decodeSpeed$'
echo "finished: $(date '+%F %T %Z')" | tee -a "$LOG/provenance.txt"
exit $rc
