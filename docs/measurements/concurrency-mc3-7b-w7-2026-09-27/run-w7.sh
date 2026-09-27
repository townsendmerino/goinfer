#!/bin/bash
# MC3 on the 7B, end to end — W7 grading (docs/tasks/task-concurrency-2026-09.md, pre-registered in 21e88ae4): Metal,
# qwen2.5-7b-instruct q4_k_m int4 from ~/models, serve defaults, greedy, --fixed-nonce.
# old = serve-metal @ 9efc3185 (pre-MC3, serialised); new = @ cc5f8c2c (MC3 + S2 + S3 + chunked prefill).
set -u
cd /Users/francistownsend-merino/tmcode/goinfer || exit 1
B=/Users/francistownsend-merino/goinfer-bench/mc3-2026-09-26
OUT=$B/w7-7b.json
export BENCH_W7_MODEL=$HOME/models/qwen2.5-7b-instruct-q4_k_m.gguf
export BENCH_MIN_FREE_MB=100  # vm_stat "Pages free" only: model mmaps leave it ~150 MB with 70%+ reclaimable
ts() { date '+%H:%M:%S'; }
gate() { waited=0; while :; do l1=$(sysctl -n vm.loadavg | awk '{print $2}'); if awk -v l="$l1" 'BEGIN{exit !(l <= 2.0)}'; then echo "$(ts) idle: load1=$l1"; return 0; fi; [ "$waited" -ge 1800 ] && { echo "$(ts) NOT IDLE after 1800s — stopping"; exit 1; }; [ $((waited % 60)) -eq 0 ] && echo "$(ts) waiting for idle: load1=$l1 (${waited}s)"; sleep 10; waited=$((waited + 10)); done; }
cell() {
  k=$1; n=$2
  case $k in old*) bin=$B/serve-metal-old-9efc3185;; new*) bin=$B/serve-metal-new-cc5f8c2c;; esac
  gate
  echo "$(ts) cell $k clients=$n bin=$(basename $bin)"
  GOINFER_SERVE_CPU=$bin python3 -u scripts/bench_w7_plain.py $OUT --clients $n --engines goinfer --backend metal \
    --key $k --fixed-nonce --server-log $B/w7-7b-servers.log
}
case "$BENCH_W7_MODEL" in /Volumes/*|/srv/models/*) echo "model on the archive, not the bench set"; exit 1;; esac
echo "$(ts) == MC3 7B W7 start; tree $(git rev-parse --short HEAD); model $BENCH_W7_MODEL; $(sysctl -n vm.loadavg)"
for k in old4_1 new4_1 new4_2 old4_2 old4_3 new4_3; do cell $k 4; done
for k in old1_1 new1_1 new1_2 old1_2 old1_3 new1_3; do cell $k 1; done
for k in old2_1 new2_1 new2_2 old2_2 old2_3 new2_3; do cell $k 2; done
echo "$(ts) == DONE"
