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
D=$REPO/docs/measurements/test-efficiency-2026-09/te1-2026-09-28
BIN=$HOME/goinfer-bench/te1-2026-09-28/serve-cpu-b9fcde67
mkdir -p "$D"
cd "$REPO" || exit 1
[ -x "$BIN" ] || { echo "missing pinned binary $BIN"; exit 1; }
export GOINFER_SERVE_CPU=$BIN GOINFER_SERVE_CPU_OLD=$BIN
export OLLAMA_BIN=/opt/homebrew/bin/ollama OLLAMA_MODELS=$HOME/.ollama/models
export BENCH_RUNS=3 BENCH_ENGINES=goinfer,goinfer_old BENCH_BACKENDS=cpu BENCH_DEPTHS=none BENCH_IDLE_WAIT=1800

sweep() { # gate tag
  local out=$D/aa-$1-$2.json
  echo "=== $(date '+%F %T %Z') $(date +%s) START aa gate=$1 tag=$2" | tee -a "$D/timeline.txt"
  BENCH_MODELS=0.5B,1.5B,7B BENCH_IDLE_GATE=$1 python3 scripts/bench_peer.py "$out"
  local rc=$?
  echo "=== $(date '+%F %T %Z') $(date +%s) END aa gate=$1 tag=$2 rc=$rc" | tee -a "$D/timeline.txt"
}

mutation() { # gate
  local out=$D/mut-$1.json log=$D/mut-$1.log
  rm -f "$out" "$log"
  echo "=== $(date '+%F %T %Z') $(date +%s) START mutation gate=$1" | tee -a "$D/timeline.txt"
  # >> (O_APPEND) so the HOG lines tee'd into the same log are not overwritten by the harness's own writes
  BENCH_MODELS=0.5B BENCH_IDLE_GATE=$1 python3 scripts/bench_peer.py "$out" >> "$log" 2>&1 &
  local bp=$!
  until grep -q '"engine"' "$log" 2>/dev/null || ! kill -0 $bp 2>/dev/null; do sleep 0.2; done
  echo "=== $(python3 -c 'import time; print(f"{time.time():.2f}")') HOG START gate=$1" | tee -a "$log"
  python3 -c "
import multiprocessing as mp, time
def spin(t):
    end = time.time() + t
    while time.time() < end: pass
ps = [mp.Process(target=spin, args=(90,)) for _ in range(mp.cpu_count())]
[p.start() for p in ps]; [p.join() for p in ps]"
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
