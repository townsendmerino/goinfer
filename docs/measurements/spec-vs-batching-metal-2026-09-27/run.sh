#!/bin/bash
# MC4's spec trigger on Metal (docs/tasks/task-concurrency-2026-09.md, MC4, pre-registered in b11bcfc3):
# serve-metal @ cc5f8c2c, arms batch (defaults: MC3) vs spec (-spec ngram, one generation at a time); greedy.
set -u
cd /Users/francistownsend-merino/tmcode/goinfer || exit 1
B=/Users/francistownsend-merino/goinfer-bench/spec-vs-batching-2026-09-27
mkdir -p $B
BIN=/Users/francistownsend-merino/goinfer-bench/mc3-2026-09-26/serve-metal-new-cc5f8c2c
export GOINFER_SERVE_CPU=$BIN
export BENCH_MIN_FREE_MB=100  # vm_stat "Pages free" only: model mmaps leave it ~150 MB with 70%+ reclaimable
M15=$HOME/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf
M7=$HOME/models/qwen2.5-7b-instruct-q4_k_m.gguf
ts() { date '+%H:%M:%S'; }
gate() { waited=0; while :; do l1=$(sysctl -n vm.loadavg | awk '{print $2}'); if awk -v l="$l1" 'BEGIN{exit !(l <= 2.0)}'; then echo "$(ts) idle: load1=$l1"; return 0; fi; [ "$waited" -ge 1800 ] && { echo "$(ts) NOT IDLE after 1800s — stopping"; exit 1; }; [ $((waited % 60)) -eq 0 ] && echo "$(ts) waiting for idle: load1=$l1 (${waited}s)"; sleep 10; waited=$((waited + 10)); done; }
args() { case $1 in spec*) echo "--serve-args=-spec=ngram";; *) echo "";; esac; }
copy() { model=$1 out=$2 k=$3 n=$4; gate; echo "$(ts) copy $(basename $out) $k clients=$n"
  BENCH_W7_MODEL=$model python3 -u scripts/bench_spec_copy.py $B/$out --key $k --clients $n $(args $k) --server-log $B/servers.log; }
chat() { k=$1 n=$2; gate; echo "$(ts) chat $k clients=$n"
  BENCH_W7_MODEL=$M15 python3 -u scripts/bench_w7_plain.py $B/chat-15b.json --clients $n --engines goinfer --backend metal \
    --key $k --fixed-nonce $(args $k) --server-log $B/servers.log; }
echo "$(ts) == spec vs batching start; tree $(git rev-parse --short HEAD); bin $(basename $BIN); $(sysctl -n vm.loadavg)"
for n in 4 1; do for k in batch${n}_1 spec${n}_1 spec${n}_2 batch${n}_2 batch${n}_3 spec${n}_3; do copy $M15 copy-15b.json $k $n; done; done
for n in 4 1; do for k in batch${n}_1 spec${n}_1 spec${n}_2 batch${n}_2 batch${n}_3 spec${n}_3; do chat $k $n; done; done
for n in 4 1; do for k in batch${n}_1 spec${n}_1 spec${n}_2 batch${n}_2 batch${n}_3 spec${n}_3; do copy $M7 copy-7b.json $k $n; done; done
echo "$(ts) == DONE"
