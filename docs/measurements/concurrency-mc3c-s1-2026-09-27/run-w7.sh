#!/bin/bash
# MC3c step 2 S1 — W7 4-client confirmation (reported; identity hard), pre-registered in 5f7480e0: old = step 2 (2c1d89ec), new = S1 fused projections (000efe2e).
# qwen2.5-7b-instruct q4_k_m int4 from ~/models, serve defaults (-max-concurrent 4, -kv-sessions 4, -cpu-batch auto),
# greedy, 6 turns x 128 tokens, --fixed-nonce, a fresh server per cell.
# old = serve @ 0bc06f90 (step 1: independent workers); new = serve @ 2c1d89ec (step 2: auto batches the 7B).
set -u
OLD=2c1d89ec; NEW=000efe2e
cd /home/francis/mycode/goinfer || exit 1
B=/home/francis/goinfer-bench/mc3c-s1-2026-09-27
M7=$HOME/models/qwen2.5-7b-instruct-q4_k_m.gguf
M05=$HOME/models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf
ts() { date '+%H:%M:%S'; }
busy() { ps -eo comm= | grep -E '^(go|.*\.test|serve.*)$' | tr '\n' ' '; }
gate() {
  w=0
  while :; do
    l1=$(awk '{print $1}' /proc/loadavg); b=$(busy)
    if awk -v l="$l1" 'BEGIN{exit !(l <= 1.0)}' && [ -z "$b" ]; then echo "$(ts) idle: load1=$l1"; return 0; fi
    [ $((w % 60)) -eq 0 ] && echo "$(ts) waiting: load1=$l1 busy=[$b] (${w}s)"
    [ "$w" -ge 3600 ] && { echo "$(ts) NOT IDLE after 3600s — stopping"; exit 1; }
    sleep 10; w=$((w + 10))
  done
}
cell() { # key clients model out [serve-args]
  k=$1; n=$2; model=$3; out=$4; extra=${5:-}
  case $k in old*) bin=$B/serve-cpu-$OLD;; new*) bin=$B/serve-cpu-$NEW;; esac
  gate
  echo "$(ts) cell $k clients=$n bin=$(basename $bin) model=$(basename $model) extra=[$extra] loadavg $(cat /proc/loadavg)"
  BENCH_W7_MODEL=$model GOINFER_SERVE_CPU=$bin python3 -u scripts/bench_w7_plain.py $out --clients $n --engines goinfer \
    --backend cpu --key $k --fixed-nonce --serve-args "$extra" --server-log $B/w7-servers.log || { echo "$(ts) cell $k FAILED"; exit 1; }
  [ -s "$out" ] || { echo "$(ts) no results file after cell $k"; exit 1; }
}
for m in "$M7" "$M05"; do case "$m" in /srv/models/*|/Volumes/*) echo "archive path"; exit 1;; esac; [ -f "$m" ] || { echo "no $m"; exit 1; }; done
echo "$(ts) == MC3c S1 W7 confirmation start; tree $(git rev-parse --short HEAD); old $OLD new $NEW; kernel $(uname -r); governor $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null); $(nproc) threads; $(cat /proc/loadavg)"
for k in old4_1 new4_1 new4_2 old4_2 old4_3 new4_3; do cell $k 4 "$M7" $B/w7.json; done
echo "$(ts) == DONE"
