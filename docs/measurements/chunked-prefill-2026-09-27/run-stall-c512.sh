#!/bin/bash
# Chunked-prefill stall grading (docs/tasks/task-concurrency-2026-09.md, revised candidate registered in b69c73ad). Metal, qwen2.5-coder-1.5b
# int4, serve defaults. old = serve-metal @ 731f4f4e (S3: whole prefill), new = @ b69c73ad with -prefill-chunk 512 (tail fix).
# stall cells: 3 decoders x 400 streamed tokens + 3 newcomers of ~2400 words; solo cells: the 3 newcomers alone.
set -u
cd /Users/francistownsend-merino/tmcode/goinfer || exit 1
B=/Users/francistownsend-merino/goinfer-bench/mc3-2026-09-26
OUT=$B/stall-c512.json
export BENCH_MIN_FREE_MB=100
ts() { date '+%H:%M:%S'; }
gate() { waited=0; while :; do l1=$(sysctl -n vm.loadavg | awk '{print $2}'); if awk -v l="$l1" 'BEGIN{exit !(l <= 2.0)}'; then echo "$(ts) idle: load1=$l1"; return 0; fi; [ "$waited" -ge 1800 ] && { echo "$(ts) NOT IDLE after 1800s — stopping"; exit 1; }; [ $((waited % 60)) -eq 0 ] && echo "$(ts) waiting for idle: load1=$l1 (${waited}s)"; sleep 10; waited=$((waited + 10)); done; }
cell() {
  k=$1; dec=$2
  case $k in old*) bin=$B/serve-metal-new-731f4f4e;; new*) bin=$B/serve-metal-new-b69c73ad;; esac
  gate; args=""; case $k in new*) args="--serve-args=-prefill-chunk=512";; esac
  echo "$(ts) cell $k decoders=$dec bin=$(basename $bin)"
  GOINFER_SERVE_CPU=$bin python3 -u scripts/bench_prefill_stall.py $OUT --key $k --decoders $dec --newcomers 3 \
    --prompt-words 2400 --decode-tokens 400 --server-log $B/stall-c512-servers.log $args
}
echo "$(ts) == chunked-prefill stall grading start; tree $(git rev-parse --short HEAD); $(sysctl -n vm.loadavg)"
for k in old_1 new_1 new_2 old_2 old_3 new_3; do cell $k 3; done
for k in oldsolo_1 newsolo_1 newsolo_2 oldsolo_2 oldsolo_3 newsolo_3; do cell $k 0; done
echo "$(ts) == DONE"
