#!/bin/bash
# MC3c W7 measurement (docs/tasks/task-concurrency-2026-09.md gates 3-5): CPU backend, qwen2.5-coder-1.5b int4, 1/2/4
# clients. Arms: old = the MC1 build (serialized, pre-MC3c); n1 = the MC3c build at -max-concurrent 1 (default);
# n4 = the MC3c build at -max-concurrent 4. Idle-gated.
set -u
cd /Users/francistownsend-merino/tmcode/goinfer || exit 1
L=/Users/francistownsend-merino/goinfer-bench/mc3c-2026-09-26
export BENCH_MIN_FREE_MB=500
ts() { date '+%H:%M:%S'; }
waited=0
while :; do
  l1=$(sysctl -n vm.loadavg | awk '{print $2}')
  if awk -v l="$l1" 'BEGIN{exit !(l <= 2.0)}'; then echo "$(ts) idle: load1=$l1"; break; fi
  [ "$waited" -ge 1800 ] && { echo "$(ts) NOT IDLE after 1800s — stopping"; exit 1; }
  [ $((waited % 60)) -eq 0 ] && echo "$(ts) waiting for idle: load1=$l1 (${waited}s)"
  sleep 10; waited=$((waited + 10))
done
echo "$(ts) == MC3c start; commit $(git rev-parse --short HEAD) + uncommitted MC3c"
GOINFER_SERVE_CPU=/Users/francistownsend-merino/goinfer-bench/mc1-2026-09-26/serve-metal-mc1 python3 -u scripts/bench_w7_plain.py $L/mc3c.json --clients 1,2,4 --engines goinfer --backend cpu --key old
GOINFER_SERVE_CPU=$L/serve-mc3c python3 -u scripts/bench_w7_plain.py $L/mc3c.json --clients 1,2,4 --engines goinfer --backend cpu --key n1
GOINFER_SERVE_CPU=$L/serve-mc3c python3 -u scripts/bench_w7_plain.py $L/mc3c.json --clients 1,2,4 --engines goinfer --backend cpu --key n4 --serve-args "-max-concurrent 4"
GOINFER_SERVE_CPU=$L/serve-mc3c python3 -u scripts/bench_w7_plain.py $L/mc3c.json --clients 4 --engines goinfer --backend cpu --key n4_rep --serve-args "-max-concurrent 4"
GOINFER_SERVE_CPU=$L/serve-mc3c python3 -u scripts/bench_w7_plain.py $L/mc3c.json --clients 4 --engines goinfer --backend cpu --key n1_rep
echo "$(ts) == DONE"
