#!/bin/bash
# MC1 measurement (docs/tasks/task-concurrency-2026-09.md): W7 plain workload on Metal at 1/2/4 clients, the MC1 build
# (-kv-sessions default 4 -> 4 resident KV slots) against the pre-MC1 build, same session, idle-gated. Order:
# old 1,2,4 then new 1,2,4, then old 4 and new 4 again (a repeat of the deciding cell, reversed order).
set -u
cd /Users/francistownsend-merino/tmcode/goinfer || exit 1
L=$HOME/goinfer-bench/mc1-2026-09-26
OLD=$HOME/goinfer-bench/mc0-2026-09-26/serve-metal-lrufix   # pre-MC1 (LRU fix only; Metal resident unchanged)
NEW=$L/serve-metal-mc1
ts() { date '+%H:%M:%S'; }
waited=0
while :; do
  l1=$(sysctl -n vm.loadavg | awk '{print $2}')
  if awk -v l="$l1" 'BEGIN{exit !(l <= 2.0)}'; then echo "$(ts) idle: load1=$l1"; break; fi
  [ "$waited" -ge 1800 ] && { echo "$(ts) NOT IDLE after 1800s (load1=$l1) — stopping"; exit 1; }
  [ $((waited % 60)) -eq 0 ] && echo "$(ts) waiting for idle: load1=$l1 (${waited}s)"
  sleep 10; waited=$((waited + 10))
done
echo "$(ts) == MC1 start; commit $(git rev-parse --short HEAD); old $OLD; new $NEW"
GOINFER_SERVE_CPU=$OLD python3 -u scripts/bench_w7_plain.py $L/mc1.json --clients 1,2,4 --engines goinfer --backend metal --key old
GOINFER_SERVE_CPU=$NEW python3 -u scripts/bench_w7_plain.py $L/mc1.json --clients 1,2,4 --engines goinfer --backend metal --key new
GOINFER_SERVE_CPU=$NEW python3 -u scripts/bench_w7_plain.py $L/mc1.json --clients 4 --engines goinfer --backend metal --key new_rep
GOINFER_SERVE_CPU=$OLD python3 -u scripts/bench_w7_plain.py $L/mc1.json --clients 4 --engines goinfer --backend metal --key old_rep
echo "$(ts) == DONE"
