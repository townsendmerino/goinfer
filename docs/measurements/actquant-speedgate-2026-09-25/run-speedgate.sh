#!/usr/bin/env bash
# Track A speed gate (pre-registered 73d739e4), run on an idle box after Track B.
# Kernel: aikit BenchmarkActGroupSplitHalf / BenchmarkActGroupW8A8, per-row vs per-32, -count 10.
# End to end: goinfer BenchmarkDecode, 1.5B / 7B x int4 / int8int8, per-row vs per-32 in SEPARATE
# processes, alternating order per round, 6 rounds, the first sample of each process discarded.
D=$HOME/goinfer-logs/actquant-speedgate; M=$HOME/models
until grep -q "ALL DONE" $HOME/goinfer-logs/actquant-trackb/run.log 2>/dev/null; do sleep 30; done
# Re-run Track B's mse qwen2.5-7b cell FIRST: the original was refused by the fit guard because a
# concurrent CUDA test held ~15 GB of host memory (2026-09-26T01:46Z). Nothing else runs meanwhile.
T=$HOME/goinfer-logs/actquant-trackb
echo "=== $(date -u +%FT%TZ) START rerun mse qwen2.5-7b"
INT4_SCHEME=mse ACT_GROUP=32 QUANTS=int4 timeout 5400 $T/actsweep qwen2.5-7b $HOME/models/qwen2.5-7b-instruct-q4_k_m.gguf >> $T/sweep-mse.jsonl 2>> $T/errors.log
echo "=== $(date -u +%FT%TZ) END rerun mse qwen2.5-7b rc=$?"
idle() { for i in $(seq 1 90); do l=$(cut -d" " -f1 /proc/loadavg); awk -v l="$l" "BEGIN{exit !(l<0.8)}" && return; echo "=== waiting for idle: loadavg $l"; sleep 20; done; }
idle
echo "=== $(date -u +%FT%TZ) START kernel benches"
cd $HOME/mycode/aikit/aikit/linalg && $D/linalg.test -test.run '^$' -test.bench 'BenchmarkActGroup(SplitHalf|W8A8)' -test.count 10 -test.benchtime 200x > $D/kernel.txt 2>&1
echo "=== $(date -u +%FT%TZ) END kernel benches rc=$?"
cd $HOME/mycode/goinfer/decoder
for spec in 1.5B=$M/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf 7B=$M/qwen2.5-7b-instruct-q4_k_m.gguf; do
  name=${spec%%=*}; path=${spec#*=}
  for quant in int4 int8int8; do
    for round in 1 2 3 4 5 6; do
      order="0 32"; [ $((round % 2)) = 0 ] && order="32 0"
      for g in $order; do
        idle
        echo "=== $(date -u +%FT%TZ) $name $quant round $round/6 group $g"
        GOINFER_PREQUANT_GGUF=$path GOINFER_BENCH_QUANT=$quant GOINFER_BENCH_ACT_GROUP=$g \
          $D/decoder.test -test.run '^$' -test.bench 'BenchmarkDecode$' -test.benchtime 48x -test.count 2 2>&1 \
          | grep -E '^BenchmarkDecode' | awk -v n=$name -v q=$quant -v r=$round -v g=$g '{print n, q, r, g, $0}' >> $D/e2e.txt
      done
    done
  done
done
echo "=== $(date -u +%FT%TZ) ALL DONE"
