#!/bin/bash
# Metal spec verify on the step kernels — served grading (docs/tasks/task-concurrency-2026-09.md MC4, registered in
# 8bb80b1f). 1 client, greedy. plain and spec-new: serve-metal @ 1e153876 (plain's path is unchanged by it);
# spec-old: @ 8bb80b1f, the build's parent. Arms rotate per round.
set -u
cd /Users/francistownsend-merino/tmcode/goinfer || exit 1
B=/Users/francistownsend-merino/goinfer-bench/mc3-2026-09-26
V=/Users/francistownsend-merino/goinfer-bench/spec-step-verify-2026-09-27
mkdir -p $V
export BENCH_MIN_FREE_MB=100
NEW=$B/serve-metal-specnew-1e153876; OLD=$B/serve-metal-specold-8bb80b1f
ts() { date '+%H:%M:%S'; }
gate() { waited=0; while :; do l1=$(sysctl -n vm.loadavg | awk '{print $2}'); if awk -v l="$l1" 'BEGIN{exit !(l <= 2.0)}'; then echo "$(ts) idle: load1=$l1"; return 0; fi; [ "$waited" -ge 1800 ] && { echo "$(ts) NOT IDLE after 1800s — stopping"; exit 1; }; [ $((waited % 60)) -eq 0 ] && echo "$(ts) waiting for idle: load1=$l1 (${waited}s)"; sleep 10; waited=$((waited + 10)); done; }
bin() { case $1 in specold) echo $OLD;; *) echo $NEW;; esac; }
args() { case $1 in plain) echo "";; *) echo "--serve-args=-spec=ngram";; esac; }
copy() { model=$1 tag=$2 arm=$3 r=$4; gate; echo "$(ts) copy $tag ${arm}_$r"
  GOINFER_SERVE_CPU=$(bin $arm) BENCH_W7_MODEL=$model python3 -u scripts/bench_spec_copy.py $V/copy-$tag.json --key ${arm}_$r --clients 1 $(args $arm) --server-log $V/servers.log; }
chat() { model=$1 tag=$2 arm=$3 r=$4; gate; echo "$(ts) chat $tag ${arm}_$r"
  GOINFER_SERVE_CPU=$(bin $arm) BENCH_W7_MODEL=$model python3 -u scripts/bench_w7_plain.py $V/chat-$tag.json --clients 1 --engines goinfer --backend metal --key ${arm}_$r --fixed-nonce $(args $arm) --server-log $V/servers.log; }
ORDER=("plain specold specnew" "specnew plain specold" "specold specnew plain")
echo "$(ts) == spec step-verify served grading start; tree $(git rev-parse --short HEAD); $(sysctl -n vm.loadavg)"
for tag in 15b 7b; do
  case $tag in 15b) M=$HOME/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf;; 7b) M=$HOME/models/qwen2.5-7b-instruct-q4_k_m.gguf;; esac
  for r in 1 2 3; do for arm in ${ORDER[$((r-1))]}; do copy $M $tag $arm $r; done; done
  for r in 1 2 3; do for arm in ${ORDER[$((r-1))]}; do chat $M $tag $arm $r; done; done
done
echo "$(ts) == DONE"
