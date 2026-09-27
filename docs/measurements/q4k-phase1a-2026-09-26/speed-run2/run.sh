#!/usr/bin/env bash
# Phase 1a speed gate (registered 306534a7): CPU decode, goinfer against goinfer, 6 paired rounds,
# alternating order, separate processes, first sample of each process discarded, idle box.
#   qwen2.5-7b: q4k vs today's int4 (per-row activations)      bands: >=0.85 SHIP, [0.78,0.85) AMBIGUOUS, <0.78 FAIL
#   phi3-mini:  q4k vs int8int8 with per-32 activations         bands: >=1.15 SHIP, [1.05,1.15) AMBIGUOUS, <1.05 FAIL
D=$HOME/goinfer-logs/q4k-speedgate-1a; M=$HOME/models
exec >> "$D/run.log" 2>&1
# Idle check (changed 2026-09-26 4:33 pm PDT, owner's choice, before any valid pair): CPU idle measured
# over 3 s from /proc/stat must be >= 95% (one busy thread of 16 is ~94%, so any competing work
# blocks). It replaces the 1-minute loadavg < 0.8 check, which mostly waited ~2-3 min for the
# previous run's own trailing average to decay. CPU temperature (k10temp Tctl) is logged per run,
# so a thermal effect from back-to-back runs is visible in the record.
temp() { awk '{printf "%.1fC", $1/1000}' /sys/class/hwmon/hwmon1/temp1_input; }
idle() {
  for i in $(seq 1 120); do
    read -r _ a b c d e f g h _ < /proc/stat; sleep 3; read -r _ a2 b2 c2 d2 e2 f2 g2 h2 _ < /proc/stat
    tot=$(( (a2+b2+c2+d2+e2+f2+g2+h2) - (a+b+c+d+e+f+g+h) )); idl=$(( (d2+e2) - (d+e) ))
    pct=$(( 100 * idl / tot ))
    if [ "$pct" -ge 95 ]; then echo "=== idle: ${pct}% cpu idle, $(temp)"; return; fi
    echo "=== waiting for idle: ${pct}% cpu idle, $(temp)"; sleep 5
  done
}
cd $HOME/mycode/goinfer/decoder
run() { # name path quant group round
  idle
  echo "=== $(date -u +%FT%TZ) $1 $3 g$4 round $5/6"
  GOINFER_PREQUANT_GGUF=$2 GOINFER_BENCH_QUANT=$3 GOINFER_BENCH_ACT_GROUP=$4 \
    $D/decoder.test -test.run '^$' -test.bench 'BenchmarkDecode$' -test.benchtime 48x -test.count 2 2>&1 \
    | grep -E '^BenchmarkDecode' | awk -v n=$1 -v q=$3 -v g=$4 -v r=$5 '{print n, q, "g"g, r, $0}' >> $D/e2e.txt
}
for round in 1 2 3 4 5 6; do
  if [ $((round % 2)) = 1 ]; then
    run qwen2.5-7b $M/qwen2.5-7b-instruct-q4_k_m.gguf int4 0 $round
    run qwen2.5-7b $M/qwen2.5-7b-instruct-q4_k_m.gguf q4k 0 $round
    run phi3-mini $M/phi3-mini-4k-gguf/Phi-3-mini-4k-instruct-q4.gguf int8int8 32 $round
    run phi3-mini $M/phi3-mini-4k-gguf/Phi-3-mini-4k-instruct-q4.gguf q4k 0 $round
  else
    run qwen2.5-7b $M/qwen2.5-7b-instruct-q4_k_m.gguf q4k 0 $round
    run qwen2.5-7b $M/qwen2.5-7b-instruct-q4_k_m.gguf int4 0 $round
    run phi3-mini $M/phi3-mini-4k-gguf/Phi-3-mini-4k-instruct-q4.gguf q4k 0 $round
    run phi3-mini $M/phi3-mini-4k-gguf/Phi-3-mini-4k-instruct-q4.gguf int8int8 32 $round
  fi
done
echo "=== $(date -u +%FT%TZ) ALL DONE"
