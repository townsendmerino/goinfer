#!/bin/bash
# MC0b: the same four cells with the session-LRU preamble fix (pickSession), to re-establish the CPU control and to
# confirm Metal's resident path still thrashes independently of the LRU. Idle-gated.
set -u
cd /Users/francistownsend-merino/tmcode/goinfer || exit 1
M=$HOME/goinfer-bench/mc0-2026-09-26
export GOINFER_SERVE_CPU=$M/serve-metal-lrufix
ts() { date '+%H:%M:%S'; }
waited=0
while :; do
  l1=$(sysctl -n vm.loadavg | awk '{print $2}')
  if awk -v l="$l1" 'BEGIN{exit !(l <= 2.0)}'; then echo "$(ts) idle: load1=$l1"; break; fi
  [ "$waited" -ge 1800 ] && { echo "$(ts) NOT IDLE after 1800s (load1=$l1) — stopping"; exit 1; }
  [ $((waited % 60)) -eq 0 ] && echo "$(ts) waiting for idle: load1=$l1 (${waited}s)"
  sleep 10; waited=$((waited + 10))
done
echo "$(ts) == MC0b start; commit $(git rev-parse --short HEAD) + uncommitted LRU fix; serve $GOINFER_SERVE_CPU"
python3 -u scripts/bench_w7_plain.py $M/mc0b.json --clients 1,2 --engines goinfer --backend cpu
echo "$(ts) == cpu exit=$?"
python3 -u scripts/bench_w7_plain.py $M/mc0b.json --clients 1,2 --engines goinfer --backend metal
echo "$(ts) == metal exit=$?"
echo "$(ts) == DONE"
