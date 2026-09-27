#!/bin/bash
# MC1 on CUDA — W7 grading (docs/tasks/task-concurrency-2026-09.md MC1, pre-registered before any timing): nobara-pc,
# RTX 2070 SUPER 8 GB, qwen2.5-coder-1.5b-instruct q4_k_m int4 from ~/models (NVMe), serve defaults (-kv-sessions 4),
# greedy, 6 turns x 128 tokens, --fixed-nonce, a fresh server per cell.
# old = serve-cuda @ $OLD (one KV slot); new = serve-cuda @ $NEW (MC1-CUDA). Then one reported 4-client pair on the 7B.
set -u
OLD=${OLD:?}; NEW=${NEW:?}
cd /home/francis/mycode/goinfer || exit 1
B=/home/francis/goinfer-bench/mc1-cuda-2026-09-27
OUT=$B/w7.json
OUT7=$B/w7-7b.json
M15=$HOME/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf
M7=$HOME/models/qwen2.5-7b-instruct-q4_k_m.gguf
ts() { date '+%H:%M:%S'; }
# Idle gate: CPU load1 <= 2.0 (/proc/loadavg), and the GPU back to its desktop baseline — no compute process beyond
# those present at the start (kwin_wayland, the compositor, is always one) and memory used within 256 MiB of what it
# was at the start — so a server's free-VRAM-derived context and slot count never see a predecessor's leftovers.
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
cell() {
  k=$1; n=$2; model=$3; out=$4
  case $k in old*) bin=$B/serve-cuda-$OLD;; new*) bin=$B/serve-cuda-$NEW;; esac
  gate
  echo "$(ts) cell $k clients=$n bin=$(basename $bin) model=$(basename $model)"
  BENCH_W7_MODEL=$model GOINFER_SERVE_CPU=$bin python3 -u scripts/bench_w7_plain.py $out --clients $n --engines goinfer \
    --backend cuda --key $k --fixed-nonce --server-log $B/w7-servers.log || { echo "$(ts) cell $k FAILED"; exit 1; }
  [ -s "$out" ] || { echo "$(ts) no results file after cell $k"; exit 1; }
}
for m in "$M15" "$M7"; do
  case "$m" in /srv/models/*|/Volumes/*) echo "model $m is on the archive, not the bench set"; exit 1;; esac
  [ -f "$m" ] || { echo "no model at $m"; exit 1; }
done
echo "$(ts) == MC1-CUDA W7 start; tree $(git rev-parse --short HEAD); old $OLD new $NEW; driver $(nvidia-smi --query-gpu=driver_version --format=csv,noheader); gpu baseline ${BASE_MIB}MiB, ${BASE_PROCS} compute proc(s) ($(nvidia-smi --query-compute-apps=process_name --format=csv,noheader | tr '\n' ' ')); $(cat /proc/loadavg)"
for n in 1 2 4; do
  for k in old${n}_1 new${n}_1 new${n}_2 old${n}_2 old${n}_3 new${n}_3; do cell $k $n "$M15" "$OUT"; done
done
echo "$(ts) == 1.5B cells done; the reported 7B pair"
cell old7b4_1 4 "$M7" "$OUT7"
cell new7b4_1 4 "$M7" "$OUT7"
echo "$(ts) == DONE"
