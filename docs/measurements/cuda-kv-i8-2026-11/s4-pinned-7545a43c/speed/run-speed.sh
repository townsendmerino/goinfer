#!/usr/bin/env bash
# S4 step 3, the speed check as registered (docs/measurements/cuda-kv-i8-2026-11/s4-registration.md "Speed"), re-run after bench_peer's
# archive check refused the 08:02 attempt over two table entries this run does not load. The served binary is the pinned build of
# 7545a43c; bench_peer.py runs from 33bfef1d (#777: the archive check covers the selected checkpoints), whose bench_peer.py, prompts.json,
# hostmem.py and timing_lock.py differ from 7545a43c's only by that fix. Every load is fit-checked first by the server's own banner; a
# refusal stops the run. Two passes, arm order reversed (A B B A), bench_peer holding the lock and voiding a cell started under heavy load.
set -uo pipefail
P=$HOME/goinfer-bench/s4-pinned
H=$P/harness-33bfef1d
L=$P/logs
SERVE=$P/bin/serve-cuda-7545a43c
export PATH=/usr/local/go/bin:$HOME/go/bin:$PATH
state() { echo "load $(cut -d' ' -f1-3 /proc/loadavg) MemAvailable $(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo) MiB"; }
for p in "$HOME/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf" "$HOME/models/qwen2.5-7b-instruct-q4_k_m.gguf"; do
  r=$(readlink -f "$p"); case "$r" in /srv/models*|/Volumes/*) echo "FATAL: $p resolves to the archive ($r)"; exit 2;; esac
  [ -e "$r" ] || { echo "FATAL: $p is missing"; exit 2; }
done
CTX=16896
fit() { # fit <name> <model> [flags...]
  local name=$1 model=$2; shift 2
  local log=$L/fit2-$name.log
  $SERVE -backend cuda -require-backend -model "$model" -ctx $CTX -addr 127.0.0.1:18099 "$@" > $log 2>&1 &
  local pid=$!
  for _ in $(seq 1 240); do
    grep -q 'serving on' $log && break
    kill -0 $pid 2>/dev/null || break
    sleep 1
  done
  kill $pid 2>/dev/null; wait $pid 2>/dev/null
  if ! grep -q 'serving on' $log; then
    echo "STOP: the fit check of $name did not reach a serving server (read $log); the speed check does not run"; return 1
  fi
  echo "fit $name: $(grep -iE 'fit:|KV plan:|resident KV slots' $log | head -3 | tr '\n' ' ')"
}
echo "== fit checks start $(date '+%F %T %Z') $(state)"
fit 15B-auto "$HOME/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf" || exit 1
fit 15B-i8 "$HOME/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf" -kv i8 || exit 1
fit 7B-auto "$HOME/models/qwen2.5-7b-instruct-q4_k_m.gguf" || exit 1
fit 7B-i8 "$HOME/models/qwen2.5-7b-instruct-q4_k_m.gguf" -kv i8 || exit 1

OUT=$L/speed2
mkdir -p $OUT
{ echo "served:   $(git -C $P/wt rev-parse HEAD) ($SERVE, sha256 $(sha256sum $SERVE | cut -c1-16))"
  echo "harness:  $(git -C $H rev-parse HEAD) (bench_peer.py from #777)"; echo "started:  $(date '+%F %T %Z')"
  echo "driver:   $(nvidia-smi --query-gpu=driver_version --format=csv,noheader)"; echo "arms:     goinfer = default (auto, f16 here), goinfer_old = -kv i8 (one binary)"
  echo "cells:    1.5B, 7B x 128, 512, 4096, 16384 (16,436 tokens); BENCH_CTX=$CTX; 2 runs a pass, two passes in reversed arm order"
  echo "prompts:  $P/prompts-s4.json (sha256 $(sha256sum $P/prompts-s4.json | cut -c1-16))"; } | tee $OUT/provenance.txt
export GOINFER_SERVE_CUDA=$SERVE GOINFER_SERVE_CUDA_OLD=$SERVE BENCH_BACKENDS=cuda BENCH_RUNS=2 BENCH_CTX=$CTX
export BENCH_GOINFER_ARGS="" BENCH_GOINFER_OLD_ARGS="-kv i8"
# scripts/prompts.json at 33bfef1d has no 4,096 or 16,384 prompt for these models; prompts-s4.json is it plus those four, calibrated by
# scripts/bench_prompts_calibrate.py against this serve binary (logs/calibrate.log).
export BENCH_PROMPTS=$P/prompts-s4.json
settle() {
  for _ in $(seq 1 90); do awk '{exit !($1 < 0.7)}' /proc/loadavg && return 0; sleep 10; done
  echo "!! load still $(cut -d' ' -f1 /proc/loadavg) after 15 min; bench_peer decides"
}
src=0
pass() { # pass <name> <engines>
  local name=$1 eng=$2
  settle
  echo "== $name start $(date '+%T') $(state)"
  (cd $H && BENCH_ENGINES=$eng BENCH_MODELS=1.5B,7B BENCH_DEPTHS=128,512,4096,16384 BENCH_SERVE_LOG_DIR="$OUT/$name-serve-logs" \
    python3 -u scripts/bench_peer.py "$OUT/$name.json") > "$OUT/$name.log" 2>&1
  local r=$?
  echo "== $name exit $r $(date '+%T')"
  if [ $r -ne 0 ] || [ ! -s "$OUT/$name.json" ]; then
    echo "!! $name exited $r or wrote no results (bench_peer exits 0 when it refuses a busy box)"; tail -5 "$OUT/$name.log"; src=1; return 1
  fi
}
pass f16first goinfer,goinfer_old || { echo "STOP: the first pass did not complete; the second does not run"; date '+%F %T %Z' > $L/speed2.done; exit 1; }
pass i8first goinfer_old,goinfer
echo "finished: $(date '+%F %T %Z') rc=$src" | tee -a $OUT/provenance.txt
date '+%F %T %Z' > $L/speed2.done
exit $src
