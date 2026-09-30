#!/bin/bash
# MC4's spec-vs-batching premise measurement on CUDA (docs/prompts/nobara-mc4-spec-alone-cuda-2026-09.md, a copy of
# the 2026-09-27 Metal design, docs/measurements/spec-vs-batching-metal-2026-09-27/run.sh): nobara-pc, RTX 2070
# SUPER, driver 595.91.07, serve-cuda @ 2d052fc4 (item 30 + the MC3-CUDA batched-head lever both already in),
# arms batch (serve's defaults: MC3 on) vs spec (-spec ngram, one generation at a time); greedy.
set -u
cd /home/francis/mycode/goinfer || exit 1
B=/home/francis/goinfer-bench/spec-vs-batching-cuda-2026-09-28
BIN=$B/serve-cuda-2d052fc4
export GOINFER_SERVE_CPU=$BIN
M15=$HOME/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf
M7=$HOME/models/qwen2.5-7b-instruct-q4_k_m.gguf
ts() { date '+%H:%M:%S'; }
# Idle gate: CUDA's own (concurrency-mc3-cuda-2026-09-27/run-w7.sh's shape) — load1 from /proc/loadavg, GPU compute
# process count and memory back to the run's own starting baseline, not the Mac's sysctl vm.loadavg alone.
BASE_MIB=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
BASE_PROCS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)
gate() {
  waited=0
  while :; do
    l1=$(awk '{print $1}' /proc/loadavg)
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    procs=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)
    if awk -v l="$l1" 'BEGIN{exit !(l <= 2.0)}' && [ "$procs" -le "$BASE_PROCS" ] && [ "$used" -le $((BASE_MIB + 256)) ]; then
      echo "$(ts) idle: load1=$l1 gpu_used=${used}MiB compute_procs=$procs"; return 0
    fi
    [ "$waited" -ge 1800 ] && { echo "$(ts) NOT IDLE after 1800s (load1=$l1 gpu_used=${used}MiB procs=$procs) — stopping"; exit 1; }
    [ $((waited % 60)) -eq 0 ] && echo "$(ts) waiting for idle: load1=$l1 gpu_used=${used}MiB procs=$procs (${waited}s)"
    sleep 5; waited=$((waited + 5))
  done
}
args() { case $1 in spec*) echo "--serve-args=-spec=ngram";; *) echo "";; esac; }
copy() { model=$1 out=$2 k=$3 n=$4; gate; echo "$(ts) copy $(basename $out) $k clients=$n"
  BENCH_W7_MODEL=$model python3 -u scripts/bench_spec_copy.py $B/$out --key $k --clients $n --backend cuda \
    $(args $k) --server-log $B/servers.log; }
chat() { k=$1 n=$2; gate; echo "$(ts) chat $k clients=$n"
  BENCH_W7_MODEL=$M15 python3 -u scripts/bench_w7_plain.py $B/chat-15b.json --clients $n --engines goinfer --backend cuda \
    --key $k --fixed-nonce $(args $k) --server-log $B/servers.log; }
for m in "$M15" "$M7"; do case "$m" in /srv/models/*|/Volumes/*) echo "$(ts) $m is on the archive, not the bench set"; exit 1;; esac; [ -f "$m" ] || { echo "$(ts) no model at $m"; exit 1; }; done
[ -x "$BIN" ] || { echo "$(ts) no binary $BIN"; exit 1; }
echo "$(ts) == spec-vs-batching-cuda start; tree $(git rev-parse --short HEAD); bin $(basename $BIN); driver $(nvidia-smi --query-gpu=driver_version --format=csv,noheader); gpu baseline ${BASE_MIB}MiB, ${BASE_PROCS} compute proc(s); $(cat /proc/loadavg)"
for n in 4 1; do for k in batch${n}_1 spec${n}_1 spec${n}_2 batch${n}_2 batch${n}_3 spec${n}_3; do copy $M15 copy-15b.json $k $n; done; done
for n in 4 1; do for k in batch${n}_1 spec${n}_1 spec${n}_2 batch${n}_2 batch${n}_3 spec${n}_3; do chat $k $n; done; done
for n in 4 1; do for k in batch${n}_1 spec${n}_1 spec${n}_2 batch${n}_2 batch${n}_3 spec${n}_3; do copy $M7 copy-7b.json $k $n; done; done
echo "$(ts) == DONE"
