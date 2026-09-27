#!/bin/bash
# MC3 S4 W7 grading (docs/tasks/task-concurrency-2026-09.md, registered in 915bf6fc): Metal, serve defaults, greedy,
# --fixed-nonce. old = serve-metal @ 915bf6fc (pre-S4), new = @ 7fa344b2 (S4).
set -u
cd /Users/francistownsend-merino/tmcode/goinfer || exit 1
B=/Users/francistownsend-merino/goinfer-bench/mc3-2026-09-26
OUT7=$B/w7-s4-7b.json; OUT15=$B/w7-s4-15b.json
export BENCH_MIN_FREE_MB=100  # vm_stat "Pages free" only: model mmaps leave it ~150 MB with 70%+ reclaimable
M7=$HOME/models/qwen2.5-7b-instruct-q4_k_m.gguf; M15=$HOME/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf
ts() { date '+%H:%M:%S'; }
gate() { waited=0; while :; do l1=$(sysctl -n vm.loadavg | awk '{print $2}'); if awk -v l="$l1" 'BEGIN{exit !(l <= 2.0)}'; then echo "$(ts) idle: load1=$l1 swap $(sysctl -n vm.swapusage | awk '{print $6}')"; return 0; fi; [ "$waited" -ge 1800 ] && { echo "$(ts) NOT IDLE after 1800s — stopping"; exit 1; }; [ $((waited % 60)) -eq 0 ] && echo "$(ts) waiting for idle: load1=$l1 (${waited}s)"; sleep 10; waited=$((waited + 10)); done; }
cell() {
  model=$1 out=$2 k=$3 n=$4
  case $k in old*) bin=$B/serve-metal-old-915bf6fc;; new*) bin=$B/serve-metal-new-s4-7fa344b2;; esac
  gate
  echo "$(ts) cell $(basename $out) $k clients=$n bin=$(basename $bin)"
  BENCH_W7_MODEL=$model GOINFER_SERVE_CPU=$bin python3 -u scripts/bench_w7_plain.py $out --clients $n --engines goinfer --backend metal \
    --key $k --fixed-nonce --server-log $B/w7-s4-servers.log
}
echo "$(ts) == MC3 S4 W7 start; tree $(git rev-parse --short HEAD); $(sysctl -n vm.loadavg)"
for n in 2 4 1; do for k in old${n}_1 new${n}_1 new${n}_2 old${n}_2 old${n}_3 new${n}_3; do cell $M7 $OUT7 $k $n; done; done
for k in old2_1 new2_1 new2_2 old2_2 old2_3 new2_3; do cell $M15 $OUT15 $k 2; done
echo "$(ts) == DONE"
