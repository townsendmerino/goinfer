#!/bin/bash
# L1 arm64 fix, served gate (PASS 1 (goinfer=FIX 2fd6f59, goinfer_old=OLD 3cd62e6d)). Pre-registered in goinfer e351fad4 before any fix code or timing:
# bench_peer.py CPU depth 128, 3 runs, two order-reversed passes, fix vs old 3cd62e6d (f32 scales) with Ollama 0.32.5,
# combined by geometric mean. Closed: combined fix/old >= 0.97 on all three models; 0.90-0.97 owner; < 0.90 fall back to
# the load-time f32 copy. Fix = goinfer e351fad4 with aikit replaced by the local branch arm64-f16-widen @ 2fd6f59.
set -u
L=$HOME/goinfer-bench/l1-mac

cd /Users/francistownsend-merino/tmcode/goinfer
export GOINFER_SERVE_CPU=$L/serve-cpu-fix-2fd6f59 GOINFER_SERVE_CPU_OLD=$L/serve-cpu-3cd62e6d
export OLLAMA_BIN=/opt/homebrew/bin/ollama OLLAMA_MODELS=$HOME/.ollama/models
export BENCH_RUNS=3 BENCH_ENGINES=goinfer,goinfer_old,ollama BENCH_BACKENDS=cpu BENCH_DEPTHS=none BENCH_MODELS=0.5B,1.5B,7B
export BENCH_MAX_LOADAVG=2.5 BENCH_IDLE_WAIT=1800
for i in $(seq 1 90); do l=$(sysctl -n vm.loadavg | awk '{print $2}'); awk -v l="$l" 'BEGIN{exit !(l<=2.5)}' && break; sleep 20; done
echo "=== $(date '+%F %T %Z') / $(date -u '+%T UTC') START L1 arm64 fix served PASS 1 (goinfer=FIX 2fd6f59, goinfer_old=OLD 3cd62e6d)"
sysctl vm.swapusage
python3 scripts/bench_peer.py $L/fix-speed-served.json; rc=$?
sysctl vm.swapusage
echo "=== $(date '+%F %T %Z') / $(date -u '+%T UTC') END PASS 1 rc=$rc"
