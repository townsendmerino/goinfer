#!/bin/bash
# OP3a gate S, the parked cell's served gate (docs/tasks/task-forward-architecture-2026-10.md §4.8, "Gate S's served cell").
# CPU decode on Qwen2.5-Coder-0.5B q4_k_m at depth 128, the walker (d5752b99) against the commit OP3a branched from
# (2c56d9c6), two arms and no peer, BENCH_RUNS=5, two passes in reversed engine order. Binaries were built by day
# (2026-10-16 08:26) and fit-checked: both admit the 0.5B (0.3 GB weights, 4.1 GB budget).
set -u
B=$HOME/goinfer-bench/op3-S
NEW=d5752b99 OLD=2c56d9c6
cd "$B/wt-$NEW" || exit 1
for r in $NEW $OLD; do [ -x "$B/serve-cpu-$r" ] || { echo "missing $B/serve-cpu-$r"; exit 1; }; done
ts() { echo "$(date '+%F %T %Z') / $(date -u '+%T UTC')"; }
export GOINFER_SERVE_CPU=$B/serve-cpu-$NEW GOINFER_SERVE_CPU_OLD=$B/serve-cpu-$OLD
export BENCH_RUNS=5 BENCH_BACKENDS=cpu BENCH_DEPTHS=none BENCH_MODELS=0.5B BENCH_SERVE_LOG_DIR=$B/serve-logs
{ echo "new $NEW old $OLD"; echo "host $(hostname) $(uname -srm)"; sysctl -n machdep.cpu.brand_string; echo "load $(sysctl -n vm.loadavg)"; } | tee "$B/provenance.txt"
echo "=== $(ts) START pass 1 (goinfer, goinfer_old)"
BENCH_ENGINES=goinfer,goinfer_old python3 scripts/bench_peer.py "$B/S-served-pass1.json"; rc1=$?
echo "=== $(ts) END pass 1 rc=$rc1"
echo "=== $(ts) START pass 2 (goinfer_old, goinfer)"
BENCH_ENGINES=goinfer_old,goinfer python3 scripts/bench_peer.py "$B/S-served-pass2.json"; rc2=$?
echo "=== $(ts) END pass 2 rc=$rc2"
exit $(( rc1 | rc2 ))
