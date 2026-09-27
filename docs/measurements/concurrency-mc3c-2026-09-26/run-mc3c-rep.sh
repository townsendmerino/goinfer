#!/bin/bash
set -u
cd /Users/francistownsend-merino/tmcode/goinfer || exit 1
L=/Users/francistownsend-merino/goinfer-bench/mc3c-2026-09-26
export BENCH_MIN_FREE_MB=500
ts() { date '+%H:%M:%S'; }
gate() { waited=0; while :; do l1=$(sysctl -n vm.loadavg | awk '{print $2}'); if awk -v l="$l1" 'BEGIN{exit !(l <= 2.0)}'; then echo "$(ts) idle: load1=$l1"; return 0; fi; [ "$waited" -ge 1800 ] && { echo "$(ts) NOT IDLE after 1800s"; exit 1; }; [ $((waited % 60)) -eq 0 ] && echo "$(ts) waiting for idle: load1=$l1 (${waited}s)"; sleep 10; waited=$((waited + 10)); done; }
echo "$(ts) == MC3c repeat pairs (the first repeats ran under external load 8.6: void)"
for k in n1_r2 n4_r2 n4_r3 n1_r3; do
  gate
  args=""; case $k in n4*) args="-max-concurrent 4";; esac
  GOINFER_SERVE_CPU=$L/serve-mc3c python3 -u scripts/bench_w7_plain.py $L/mc3c.json --clients 4 --engines goinfer --backend cpu --key $k --serve-args "$args"
done
echo "$(ts) == DONE"
