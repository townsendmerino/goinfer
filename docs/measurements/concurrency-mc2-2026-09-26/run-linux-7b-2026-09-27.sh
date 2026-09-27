#!/bin/bash
# MC3c step-2 trigger (docs/tasks/task-concurrency-2026-09.md MC3c, pre-registered in 2f5185c0): MC2 throughput on the
# 7B, nobara CPU, int4, depths 128 and 512, harness defaults. One depth per go test invocation. Before each depth: load1
# <= 1.0 and no other go / test binary / serve process on the box.
export PATH=/usr/local/go/bin:$PATH
cd ~/mycode/goinfer || exit 1
D=$HOME/goinfer-bench/mc2-7b-2026-09-27
M=$HOME/models/qwen2.5-7b-instruct-q4_k_m.gguf
case "$M" in /srv/models/*|/Volumes/*) echo "model on the archive"; exit 1;; esac
ts() { date "+%F %T %Z"; }
busy() { ps -eo comm= | grep -E '^(go|.*\.test|serve.*)$' | grep -v '^$' | tr '\n' ' '; }
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
echo "$(ts) start $(git rev-parse --short HEAD); $(uptime)"
for DEPTH in 128 512; do
  gate
  echo "== $(ts) MC2 throughput 7b depth $DEPTH int4; commit $(git rev-parse --short HEAD); loadavg $(cat /proc/loadavg)"
  GOINFER_MC2=1 GOINFER_MC2_MODEL=$M GOINFER_MC2_DEPTH=$DEPTH go test -count=1 -timeout 90m \
    -run "^TestMC2_batchedDecodeThroughput$" -v ./decoder/ 2>&1 | tee $D/depth-$DEPTH.log | grep -E "^\[mc2|FAIL|panic|^--- |^ok"
  echo "== $(ts) depth $DEPTH done; loadavg $(cat /proc/loadavg)"
done
echo "$(ts) DONE"
