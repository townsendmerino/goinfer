#!/usr/bin/env bash
# Lever 3 gate (pre-registered fcaac84f): BenchmarkResidentDecode depth 128, 6 paired rounds, alternating,
# separate processes, first sample discarded, idle check (CPU >= 95% idle over 3 s).
#   bar:   qwen2.5-coder-1.5b q4k (fused per-32) vs int4 — effective bandwidth (q4k 1.057 GB, int4 0.970 GB/token)
#   guard: qwen2.5-7b q4k fused vs unfused (GOINFER_BENCH_NO_FUSE) >= 0.98
#   info:  phi3-mini q4k fused vs unfused
D=$HOME/goinfer-logs/q4k-lever3; M=$HOME/models
exec >> "$D/run.log" 2>&1
temp() { awk '{printf "%.1fC", $1/1000}' /sys/class/hwmon/hwmon1/temp1_input; }
idle() { n=0; while :; do read -r _ a b c d e f g h _ < /proc/stat; sleep 3; read -r _ a2 b2 c2 d2 e2 f2 g2 h2 _ < /proc/stat
  tot=$(( (a2+b2+c2+d2+e2+f2+g2+h2) - (a+b+c+d+e+f+g+h) )); idl=$(( (d2+e2) - (d+e) )); pct=$(( 100 * idl / tot ))
  [ "$pct" -ge 95 ] && { echo "=== idle: ${pct}% cpu idle, $(temp)"; return; }
  n=$((n+1)); [ $((n % 12)) = 1 ] && echo "=== $(date -u +%FT%TZ) waiting for idle: ${pct}% cpu idle, $(temp)"; sleep 2; done; }
Q15=$M/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf; Q7=$M/qwen2.5-7b-instruct-q4_k_m.gguf; P3=$M/phi3-mini-4k-gguf/Phi-3-mini-4k-instruct-q4.gguf
run() { # label model quant nofuse round
  idle; echo "=== $(date -u +%FT%TZ) $1 round $5/6"
  GOINFER_BENCH_NO_FUSE=$4 GOINFER_BENCH_MODEL=$2 GOINFER_BENCH_QUANT=$3 GOINFER_BENCH_ACT_GROUP=0 GOINFER_BENCH_DEPTH=128 \
    $D/cuda.test -test.run '^$' -test.bench 'BenchmarkResidentDecode$' -test.benchtime 64x -test.count 2 2>&1 \
    | grep -E '^BenchmarkResidentDecode|FAIL|resident did not|load:' | awk -v l=$1 -v r=$5 '{print l, r, $0}' >> $D/e2e.txt
}
for round in 1 2 3 4 5 6; do
  if [ $((round % 2)) = 1 ]; then
    run 1.5B-int4 $Q15 int4 "" $round; run 1.5B-q4k-fused $Q15 q4k "" $round
    run 7B-q4k-unfused $Q7 q4k 1 $round; run 7B-q4k-fused $Q7 q4k "" $round
    run phi3-q4k-unfused $P3 q4k 1 $round; run phi3-q4k-fused $P3 q4k "" $round
  else
    run 1.5B-q4k-fused $Q15 q4k "" $round; run 1.5B-int4 $Q15 int4 "" $round
    run 7B-q4k-fused $Q7 q4k "" $round; run 7B-q4k-unfused $Q7 q4k 1 $round
    run phi3-q4k-fused $P3 q4k "" $round; run phi3-q4k-unfused $P3 q4k 1 $round
  fi
done
echo "=== $(date -u +%FT%TZ) ALL DONE"
