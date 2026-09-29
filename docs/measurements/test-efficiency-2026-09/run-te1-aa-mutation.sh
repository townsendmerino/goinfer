#!/bin/bash
# TE1 night run (docs/tasks/task-test-efficiency-2026-09.md, TE1): the instant idle gate against the load gate, on the
# MacBook, at night. Pre-registered in that section before this was queued. Analysed by te1_analyze.py in this directory.
#
#   A/A  — one binary in both arms (goinfer = goinfer_old = serve-cpu-b9fcde67), CPU, 0.5B / 1.5B / 7B, depth 128,
#          3 runs, two arms. Four sweeps, gates interleaved: load, instant, load, instant.
#   Mutation — 0.5B only, once per gate. A CPU hog (every core, 90 s) starts the moment the first cell's record lands,
#          which is when the harness enters the gate for the second cell. The gate must hold that cell for as long
#          as the hog runs; the instant gate must release within 10 s of the hog stopping.
#
# Runs under night.py, which holds the TE9 timing lock; bench_peer.py inherits it. The harness default
# BENCH_MAX_LOADAVG=1.0 applies (a night run).
set -u
REPO=/Users/francistownsend-merino/tmcode/goinfer
# Attempt 2 writes to its own directory: bench_peer.py resumes from an existing results file, and attempt 1's partial
# sweep (te1-2026-09-28/) must not be mixed into this one.
D=$REPO/docs/measurements/test-efficiency-2026-09/te1-attempt2
HOG=$REPO/docs/measurements/test-efficiency-2026-09/te1_hog.py
BIN=$HOME/goinfer-bench/te1-2026-09-28/serve-cpu-b9fcde67
mkdir -p "$D"
cd "$REPO" || exit 1
[ -x "$BIN" ] || { echo "missing pinned binary $BIN"; exit 1; }
export GOINFER_SERVE_CPU=$BIN GOINFER_SERVE_CPU_OLD=$BIN
export OLLAMA_BIN=/opt/homebrew/bin/ollama OLLAMA_MODELS=$HOME/.ollama/models
export BENCH_RUNS=3 BENCH_ENGINES=goinfer,goinfer_old BENCH_BACKENDS=cpu BENCH_DEPTHS=none BENCH_IDLE_WAIT=1800

# bench_peer.py's preflight REFUSES a busy box rather than waiting (by design), so wait for the 1-min load average to be
# at or under 1.0 before each sweep and each mutation run. Attempt 1 ran them back to back, and both load-gate sweeps
# were refused at start (load 1.50, 6.56). The wait is logged BEFORE the timeline's START line, so it is outside every
# timed span te1_analyze.py grades.
wait_idle() { # label
  local t0=$(date +%s) la
  while :; do
    la=$(sysctl -n vm.loadavg | awk '{print $2}')
    if python3 -c "import sys; sys.exit(0 if float('$la') <= 1.0 else 1)"; then break; fi
    if [ $(( $(date +%s) - t0 )) -ge 1800 ]; then echo "=== wait_idle $1: gave up after 1800 s at load $la" | tee -a "$D/timeline.txt"; return; fi
    sleep 10
  done
  echo "=== wait_idle $1: $(( $(date +%s) - t0 )) s to load $la" | tee -a "$D/timeline.txt"
}

sweep() { # gate tag
  wait_idle "aa gate=$1 tag=$2"
  local out=$D/aa-$1-$2.json
  echo "=== $(date '+%F %T %Z') $(date +%s) START aa gate=$1 tag=$2" | tee -a "$D/timeline.txt"
  BENCH_MODELS=0.5B,1.5B,7B BENCH_IDLE_GATE=$1 python3 scripts/bench_peer.py "$out"
  local rc=$?
  echo "=== $(date '+%F %T %Z') $(date +%s) END aa gate=$1 tag=$2 rc=$rc" | tee -a "$D/timeline.txt"
}

mutation() { # gate
  local out=$D/mut-$1.json log=$D/mut-$1.log
  rm -f "$out" "$log"
  wait_idle "mutation gate=$1"
  echo "=== $(date '+%F %T %Z') $(date +%s) START mutation gate=$1" | tee -a "$D/timeline.txt"
  # >> (O_APPEND) so the HOG lines tee'd into the same log are not overwritten by the harness's own writes
  BENCH_MODELS=0.5B BENCH_IDLE_GATE=$1 python3 scripts/bench_peer.py "$out" >> "$log" 2>&1 &
  local bp=$!
  until grep -q '"engine"' "$log" 2>/dev/null || ! kill -0 $bp 2>/dev/null; do sleep 0.2; done
  echo "=== $(python3 -c 'import time; print(f"{time.time():.2f}")') HOG START gate=$1" | tee -a "$log"
  python3 "$HOG" 90
  echo "=== $(python3 -c 'import time; print(f"{time.time():.2f}")') HOG STOP gate=$1" | tee -a "$log"
  wait $bp; local rc=$?
  echo "=== $(date '+%F %T %Z') $(date +%s) END mutation gate=$1 rc=$rc" | tee -a "$D/timeline.txt"
}

sweep load 1
sweep instant 1
sweep load 2
sweep instant 2
mutation instant
mutation load
echo "=== $(date '+%F %T %Z') DONE"
