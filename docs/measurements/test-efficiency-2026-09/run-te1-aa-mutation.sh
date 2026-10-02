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
# Runs under night.py, which holds the TE9 timing lock; bench_peer.py inherits it. Attempt 5 (owner, 2026-10-02) runs the
# load gate at BENCH_MAX_LOADAVG=3.0, its pre-wait at the same cap: attempts 2-4 showed this Mac's 1-min load average at
# 1.7-2.9 with the CPU near idle, by day and at night, so the 1.0 default never let a load sweep finish.
set -u
REPO=/Users/francistownsend-merino/tmcode/goinfer
# Each attempt writes to its own directory: bench_peer.py resumes from an existing results file, and an earlier
# attempt's partial sweep (te1-2026-09-28/, te1-attempt2/ ... te1-attempt4/) must not be mixed into this one.
D=$REPO/docs/measurements/test-efficiency-2026-09/te1-attempt5
LOAD_CAP=3.0
HOG=$REPO/docs/measurements/test-efficiency-2026-09/te1_hog.py
BIN=$HOME/goinfer-bench/te1-2026-09-28/serve-cpu-b9fcde67
mkdir -p "$D"
cd "$REPO" || exit 1
[ -x "$BIN" ] || { echo "missing pinned binary $BIN"; exit 1; }
export GOINFER_SERVE_CPU=$BIN GOINFER_SERVE_CPU_OLD=$BIN
export OLLAMA_BIN=/opt/homebrew/bin/ollama OLLAMA_MODELS=$HOME/.ollama/models
export BENCH_RUNS=3 BENCH_ENGINES=goinfer,goinfer_old BENCH_BACKENDS=cpu BENCH_DEPTHS=none BENCH_IDLE_WAIT=1800

# bench_peer.py's preflight REFUSES a busy box rather than waiting (by design), so before each sweep and each mutation
# wait until THAT sweep's own gate would pass (up to 30 min): the load gate's 1-min load average <= LOAD_CAP, or the instant
# gate's own sample (bench_peer.instant_idle_sample: CPU busy <= BENCH_MAX_BUSY and no timed workload active).
# Attempt 1 ran sweeps back to back, and its load-gate sweeps were refused at start. Attempt 2 waited on the load
# average for BOTH gates, which put the instant sweeps behind the very signal TE1 is testing (2026-09-29 08:42: CPU
# 90.5% idle at load 2.92). The wait is logged BEFORE the timeline's START line, outside every span te1_analyze.py
# grades.
gate_idle() { # gate -> exit 0 if that gate would pass now
  if [ "$1" = instant ]; then
    python3 -B -c "import sys; sys.path.insert(0, 'scripts'); import bench_peer as b; busy, active = b.instant_idle_sample(); sys.exit(0 if busy is not None and busy <= b.BUSY_CAP and not active else 1)"
  else
    python3 -c "import os, sys; sys.exit(0 if os.getloadavg()[0] <= $LOAD_CAP else 1)"
  fi
}
wait_idle() { # gate label
  local t0=$(date +%s)
  until gate_idle "$1"; do
    if [ $(( $(date +%s) - t0 )) -ge 1800 ]; then
      echo "=== wait_idle $2: gave up after 1800 s (load $(sysctl -n vm.loadavg))" | tee -a "$D/timeline.txt"; return
    fi
    sleep 10
  done
  echo "=== wait_idle $2: $(( $(date +%s) - t0 )) s (load $(sysctl -n vm.loadavg))" | tee -a "$D/timeline.txt"
}

sweep() { # gate tag
  wait_idle "$1" "aa gate=$1 tag=$2"
  local out=$D/aa-$1-$2.json
  echo "=== $(date '+%F %T %Z') $(date +%s) START aa gate=$1 tag=$2" | tee -a "$D/timeline.txt"
  BENCH_MODELS=0.5B,1.5B,7B BENCH_IDLE_GATE=$1 BENCH_MAX_LOADAVG=$LOAD_CAP python3 scripts/bench_peer.py "$out"
  local rc=$?
  echo "=== $(date '+%F %T %Z') $(date +%s) END aa gate=$1 tag=$2 rc=$rc" | tee -a "$D/timeline.txt"
}

mutation() { # gate
  local out=$D/mut-$1.json log=$D/mut-$1.log
  rm -f "$out" "$log"
  wait_idle "$1" "mutation gate=$1"
  echo "=== $(date '+%F %T %Z') $(date +%s) START mutation gate=$1" | tee -a "$D/timeline.txt"
  # >> (O_APPEND) so the HOG lines tee'd into the same log are not overwritten by the harness's own writes
  BENCH_MODELS=0.5B BENCH_IDLE_GATE=$1 BENCH_MAX_LOADAVG=$LOAD_CAP python3 scripts/bench_peer.py "$out" >> "$log" 2>&1 &
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
