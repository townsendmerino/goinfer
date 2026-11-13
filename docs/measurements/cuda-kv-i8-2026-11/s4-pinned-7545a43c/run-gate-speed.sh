#!/usr/bin/env bash
# S4 steps 2 and 3 on the pinned tree 7545a43c (docs/measurements/cuda-kv-i8-2026-11/s4-registration.md, addendum of 2026-11-08 02:51):
#   2. the G4 gate, S re-read then D7, the timing lock held exclusively by the gate binary's run;
#   3. the speed check as registered: one serve binary, goinfer = default (auto: f16 on these families), goinfer_old = -kv i8, the 1.5B and
#      the 7B at depths 128, 512, 4096, 16384, two passes with the arm order reversed (2 runs each: A B B A), bench_peer.py holding the
#      lock and applying the void rule per cell. Every load the speed check makes is fit-checked first by the server's own banner; a
#      refusal stops the run.
# It starts when step 1 (run-refs.sh) has written refs.done, and stops if it wrote refs.failed. Nothing else of mine runs beside it.
set -uo pipefail
P=$HOME/goinfer-bench/s4-pinned
WT=$P/wt
L=$P/logs
SERVE=$P/bin/serve-cuda-7545a43c
export PATH=/usr/local/go/bin:$HOME/go/bin:$PATH
state() { echo "load $(cut -d' ' -f1-3 /proc/loadavg) MemAvailable $(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo) MiB"; }
for p in "$HOME/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf" "$HOME/models/qwen2.5-7b-instruct-q4_k_m.gguf"; do
  r=$(readlink -f "$p"); case "$r" in /srv/models*|/Volumes/*) echo "FATAL: $p resolves to the archive ($r)"; exit 2;; esac
  [ -e "$r" ] || { echo "FATAL: $p is missing"; exit 2; }
done

echo "waiting for step 1 (refs) $(date '+%F %T %Z')"
while [ ! -e $L/refs.done ]; do
  [ -e $L/refs.failed ] && { echo "STOP: step 1 failed (refs.failed); the gate does not run"; exit 1; }
  sleep 60
done
echo "step 1 done at $(cat $L/refs.done)"

# Step 2: the gate.
echo "== step 2 (gate) start $(date '+%F %T %Z') $(state)" | tee $L/gate.start
(cd $WT/backend/cuda && GOINFER_HEAVY_TESTS=1 GOINFER_PREFILL_GATE_PROMPTS=b GOINFER_CUDA_FLASH_DECODE=16 GOINFER_KV8_GATE_CELLS=S:3900,D7:8000 \
  python3 $WT/scripts/timing_lock.py run --label s4-gate -- $P/bin/cuda.test -test.run '^TestKV8GateVsReference$' -test.v -test.count=1 -test.timeout 4h) > $L/gate.log 2>&1
rc=$?
echo "== step 2 (gate) end $(date '+%F %T %Z') exit $rc" | tee -a $L/gate.start
grep -E '=== KV8|--- (FAIL|PASS)' $L/gate.log | tee $L/gate.summary
if ! grep -q -- '--- PASS: TestKV8GateVsReference ' $L/gate.log; then
  echo "STOP: the gate binary did not pass (a precondition, a refusal or a crash: read gate.log); the speed check does not run"
  exit 1
fi

# Step 3a: fit-check every load the speed check makes, by the server's own banner.
CTX=16896
fit() { # fit <name> <model> [flags...]
  local name=$1 model=$2; shift 2
  local log=$L/fit-$name.log
  $SERVE -backend cuda -require-backend -model "$model" -ctx $CTX -addr 127.0.0.1:18099 "$@" > $log 2>&1 &
  local pid=$!
  # Success is the server reaching "serving on" (a banner can say "cannot" about tools); a refused load exits before it.
  for _ in $(seq 1 240); do
    grep -q 'serving on' $log && break
    kill -0 $pid 2>/dev/null || break
    sleep 1
  done
  kill $pid 2>/dev/null; wait $pid 2>/dev/null
  if ! grep -q 'serving on' $log; then
    echo "STOP: the fit check of $name did not reach a serving server (read $log); the speed check does not run"; return 1
  fi
  echo "fit $name: $(grep -iE 'fit:|KV plan:|KV pool|\bkv\b' $log | head -4 | tr '\n' ' ')"
}
echo "== step 3a (fit checks) start $(date '+%F %T %Z') $(state)"
fit 15B-auto "$HOME/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf" || exit 1
fit 15B-i8 "$HOME/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf" -kv i8 || exit 1
fit 7B-auto "$HOME/models/qwen2.5-7b-instruct-q4_k_m.gguf" || exit 1
fit 7B-i8 "$HOME/models/qwen2.5-7b-instruct-q4_k_m.gguf" -kv i8 || exit 1

# Step 3b: the speed passes.
OUT=$L/speed
mkdir -p $OUT
{ echo "rev:      $(git -C $WT rev-parse HEAD)"; echo "binary:   $SERVE ($(sha256sum $SERVE | cut -c1-16))"; echo "started:  $(date '+%F %T %Z')"
  echo "driver:   $(nvidia-smi --query-gpu=driver_version --format=csv,noheader)"; echo "arms:     goinfer = default (auto, f16 here), goinfer_old = -kv i8 (one binary)"
  echo "cells:    1.5B, 7B x 128, 512, 4096, 16384; BENCH_CTX=$CTX; 2 runs a pass, two passes in reversed arm order"; } | tee $OUT/provenance.txt
export GOINFER_SERVE_CUDA=$SERVE GOINFER_SERVE_CUDA_OLD=$SERVE BENCH_BACKENDS=cuda BENCH_RUNS=2 BENCH_CTX=$CTX
export BENCH_GOINFER_ARGS="" BENCH_GOINFER_OLD_ARGS="-kv i8"
settle() {
  for _ in $(seq 1 90); do awk '{exit !($1 < 0.7)}' /proc/loadavg && return 0; sleep 10; done
  echo "!! load still $(cut -d' ' -f1 /proc/loadavg) after 15 min; bench_peer decides"
}
src=0
pass() { # pass <name> <engines>
  local name=$1 eng=$2
  settle
  echo "== $name start $(date '+%T') $(state)"
  (cd $WT && BENCH_ENGINES=$eng BENCH_MODELS=1.5B,7B BENCH_DEPTHS=128,512,4096,16384 BENCH_SERVE_LOG_DIR="$OUT/$name-serve-logs" \
    python3 -u scripts/bench_peer.py "$OUT/$name.json") > "$OUT/$name.log" 2>&1
  local r=$?
  echo "== $name exit $r $(date '+%T')"
  [ $r -eq 0 ] && [ -s "$OUT/$name.json" ] || { echo "!! $name exited $r or wrote no results (bench_peer exits 0 when it refuses a busy box)"; src=1; }
}
pass f16first goinfer,goinfer_old
pass i8first goinfer_old,goinfer
echo "finished: $(date '+%F %T %Z') rc=$src" | tee -a $OUT/provenance.txt
date '+%F %T %Z' > $L/speed.done
exit $src
