#!/bin/bash
# TK-G3: A B B A x 3, each cell one BenchmarkEncode run. A cell starts only when the 1-minute load average is <= 4.0
# (waiting up to 10 min); the load at its start is recorded.
T=~/.claude/jobs/b64794f1/tmp
for round in 1 2 3; do
  for arm in old new new old; do
    for w in $(seq 1 120); do
      la=$(sysctl -n vm.loadavg | awk '{print $2}')
      awk -v l=$la 'BEGIN{exit !(l<=4.0)}' && break
      sleep 5
    done
    d=$([ $arm = old ] && echo ~/goinfer-bench/mem13tok-base/tokenizer || echo ~/goinfer-bench/mem13tok-2026-10/tokenizer)
    ns=$(cd $d && $T/tok-$arm.test -test.run '^$' -test.bench '^BenchmarkEncode$' -test.benchtime 2s -test.count 1 | awk '/BenchmarkEncode/{print $3}')
    echo "$(date +%H:%M:%S) round=$round arm=$arm load1=$la ns_op=$ns"
  done
done
