#!/bin/bash
# MC0 (docs/tasks/task-concurrency-2026-09.md): does the GPU-resident path's single KV thrash under 2 interleaved
# conversations? Metal 1 and 2 clients, then the CPU control (sessionLRU, -kv-sessions default 4) 1 and 2 clients.
# Waits for the R18 e2e sweep to finish (same GPU) and for idle.
set -u
cd /Users/francistownsend-merino/tmcode/goinfer || exit 1
M=$HOME/goinfer-bench/mc0-2026-09-26
E=$HOME/goinfer-bench/r18-e2e-2026-09-26/run.log
export GOINFER_SERVE_CPU=$HOME/goinfer-bench/r18-e2e-2026-09-26/serve-metal-622b1f9b
ts() { date '+%H:%M:%S'; }
echo "$(ts) MC0 armed; waiting for the R18 e2e to finish ($E)"
until grep -q "== DONE\|NOT IDLE" "$E" 2>/dev/null; do sleep 30; done
echo "$(ts) e2e finished: $(tail -1 $E)"
waited=0
while :; do
  l1=$(sysctl -n vm.loadavg | awk '{print $2}')
  if awk -v l="$l1" 'BEGIN{exit !(l <= 2.0)}'; then echo "$(ts) idle: load1=$l1"; break; fi
  [ "$waited" -ge 1800 ] && { echo "$(ts) NOT IDLE after 1800s (load1=$l1) — stopping"; exit 1; }
  [ $((waited % 60)) -eq 0 ] && echo "$(ts) waiting for idle: load1=$l1 (${waited}s)"
  sleep 10; waited=$((waited + 10))
done
echo "$(ts) == MC0 start; commit $(git rev-parse --short HEAD); serve $GOINFER_SERVE_CPU"
python3 -u scripts/bench_w7_plain.py $M/mc0.json --clients 1,2 --engines goinfer --backend metal
echo "$(ts) == metal exit=$?"
python3 -u scripts/bench_w7_plain.py $M/mc0.json --clients 1,2 --engines goinfer --backend cpu
echo "$(ts) == cpu exit=$?"
echo "$(ts) == DONE"
