#!/usr/bin/env bash
# Phase 1b speed gate (registered f26964a7): CUDA resident greedy decode, goinfer against goinfer,
# BenchmarkResidentDecode, 6 paired rounds, alternating order, separate processes, first sample of each
# process discarded, idle box (CPU >= 95% idle over 3 s; waits indefinitely, logging each minute).
#   phi3-mini: q4k vs int8int8 + per-32 at depths 128 and 2048   >=1.15 SHIP, [1.05,1.15) AMBIGUOUS, <1.05 FAIL (worse depth decides)
#   qwen2.5-7b: q4k vs today's int4 at depth 128                  >=0.85 SHIP, [0.78,0.85) AMBIGUOUS, <0.78 FAIL
D=$HOME/goinfer-logs/q4k-speedgate-1b; M=$HOME/models
exec >> "$D/run.log" 2>&1
temp() { awk '{printf "%.1fC", $1/1000}' /sys/class/hwmon/hwmon1/temp1_input; }
idle() {
  n=0
  while :; do
    read -r _ a b c d e f g h _ < /proc/stat; sleep 3; read -r _ a2 b2 c2 d2 e2 f2 g2 h2 _ < /proc/stat
    tot=$(( (a2+b2+c2+d2+e2+f2+g2+h2) - (a+b+c+d+e+f+g+h) )); idl=$(( (d2+e2) - (d+e) ))
    pct=$(( 100 * idl / tot ))
    if [ "$pct" -ge 95 ]; then echo "=== idle: ${pct}% cpu idle, $(temp)"; return; fi
    n=$((n+1)); [ $((n % 12)) = 1 ] && echo "=== $(date -u +%FT%TZ) waiting for idle: ${pct}% cpu idle, $(temp)"
    sleep 2
  done
}
P3=$M/phi3-mini-4k-gguf/Phi-3-mini-4k-instruct-q4.gguf; Q7=$M/qwen2.5-7b-instruct-q4_k_m.gguf
run() { # name path quant group depth round
  idle
  echo "=== $(date -u +%FT%TZ) $1 $3 g$4 depth $5 round $6/6"
  GOINFER_BENCH_MODEL=$2 GOINFER_BENCH_QUANT=$3 GOINFER_BENCH_ACT_GROUP=$4 GOINFER_BENCH_DEPTH=$5 \
    $D/cuda.test -test.run '^$' -test.bench 'BenchmarkResidentDecode$' -test.benchtime 64x -test.count 2 2>&1 \
    | grep -E '^BenchmarkResidentDecode|FAIL|resident did not|load:' | awk -v n=$1 -v q=$3 -v g=$4 -v d=$5 -v r=$6 '{print n, q, "g"g, "d"d, r, $0}' >> $D/e2e.txt
}
for round in 1 2 3 4 5 6; do
  for depth in 128 2048; do
    if [ $((round % 2)) = 1 ]; then run phi3-mini $P3 int8int8 32 $depth $round; run phi3-mini $P3 q4k 0 $depth $round
    else run phi3-mini $P3 q4k 0 $depth $round; run phi3-mini $P3 int8int8 32 $depth $round; fi
  done
  if [ $((round % 2)) = 1 ]; then run qwen2.5-7b $Q7 int4 0 128 $round; run qwen2.5-7b $Q7 q4k 0 128 $round
  else run qwen2.5-7b $Q7 q4k 0 128 $round; run qwen2.5-7b $Q7 int4 0 128 $round; fi
done
echo "=== $(date -u +%FT%TZ) ALL DONE"
