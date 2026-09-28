#!/bin/bash
# MC3 on CUDA — W7 grading (docs/tasks/task-concurrency-2026-09.md, "MC3 on CUDA", pre-registered in 01bcb640): nobara,
# RTX 2070 SUPER, driver 595.91.07, qwen2.5-coder-1.5b q4_k_m int4 from ~/models, serve defaults (-max-concurrent 4,
# -kv-sessions 4, -prefill-chunk 512), greedy, 6 turns x 128, --fixed-nonce, a fresh server per cell.
# old = serve-cuda @ 01bcb640 (MC1: slots, one generation at a time); new = serve-cuda @ 7a44a58e (MC3 on CUDA).
set -u
OLD=01bcb640; NEW=7a44a58e
cd /home/francis/mycode/goinfer || exit 1
B=/home/francis/goinfer-bench/mc3-cuda-2026-09-27
M15=$HOME/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf
M7=$HOME/models/qwen2.5-7b-instruct-q4_k_m.gguf
ts() { date '+%H:%M:%S'; }
BASE_MIB=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
BASE_PROCS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)
gate() {
  waited=0
  while :; do
    l1=$(awk '{print $1}' /proc/loadavg)
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    procs=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)
    if awk -v l="$l1" 'BEGIN{exit !(l <= 2.0)}' && [ "$procs" -le "$BASE_PROCS" ] && [ "$used" -le $((BASE_MIB + 256)) ]; then
      echo "$(ts) idle: load1=$l1 gpu_used=${used}MiB compute_procs=$procs"; return 0
    fi
    [ "$waited" -ge 1800 ] && { echo "$(ts) NOT IDLE after 1800s (load1=$l1 gpu_used=${used}MiB procs=$procs) — stopping"; exit 1; }
    [ $((waited % 60)) -eq 0 ] && echo "$(ts) waiting for idle: load1=$l1 gpu_used=${used}MiB procs=$procs (${waited}s)"
    sleep 5; waited=$((waited + 5))
  done
}
cell() {
  k=$1; n=$2; model=$3; out=$4
  case $k in old*) bin=$B/serve-cuda-$OLD;; new*) bin=$B/serve-cuda-$NEW;; esac
  gate
  echo "$(ts) cell $k clients=$n bin=$(basename $bin) model=$(basename $model)"
  BENCH_W7_MODEL=$model GOINFER_SERVE_CPU=$bin python3 -u scripts/bench_w7_plain.py $out --clients $n --engines goinfer \
    --backend cuda --key $k --fixed-nonce --server-log $B/w7-servers.log || { echo "$(ts) cell $k FAILED"; exit 1; }
  [ -s "$out" ] || { echo "$(ts) no results file after cell $k"; exit 1; }
}
for m in "$M15" "$M7"; do case "$m" in /srv/models/*|/Volumes/*) echo "archive path"; exit 1;; esac; [ -f "$m" ] || { echo "no $m"; exit 1; }; done
echo "$(ts) == MC3-CUDA W7 start; tree $(git rev-parse --short HEAD); old $OLD new $NEW; driver $(nvidia-smi --query-gpu=driver_version --format=csv,noheader); gpu baseline ${BASE_MIB}MiB, ${BASE_PROCS} compute proc(s); $(cat /proc/loadavg)"
for k in old4_1 new4_1 new4_2 old4_2 old4_3 new4_3; do cell $k 4 "$M15" $B/w7.json; done
for k in old1_1 new1_1 new1_2 old1_2 old1_3 new1_3; do cell $k 1 "$M15" $B/w7.json; done
for k in old2_1 new2_1; do cell $k 2 "$M15" $B/w7.json; done
echo "$(ts) == reported: the 7B, 4 clients, one pair"
cell old7b4_1 4 "$M7" $B/w7-7b.json
cell new7b4_1 4 "$M7" $B/w7-7b.json
echo "$(ts) == DONE"
