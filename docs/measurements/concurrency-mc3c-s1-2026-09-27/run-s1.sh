#!/bin/bash
# MC3c step 2 S1 (docs/tasks/task-concurrency-2026-09.md, pre-registered in 5f7480e0): fused vs unfused batched W4A8
# projections, TestCPUBatchS1_fusedVsUnfused on the 7B, depths 128 and 512, one go test per depth, idle-gated.
export PATH=/usr/local/go/bin:$PATH
cd ~/mycode/goinfer || exit 1
D=$HOME/goinfer-bench/mc3c-s1-2026-09-27
M=$HOME/models/qwen2.5-7b-instruct-q4_k_m.gguf
case "$M" in /srv/models/*|/Volumes/*) echo "archive path"; exit 1;; esac
ts() { date "+%F %T %Z"; }
busy() { ps -eo comm= | grep -E '^(go|.*\.test|serve.*)$' | tr '\n' ' '; }
gate() { w=0; while :; do l1=$(awk '{print $1}' /proc/loadavg); b=$(busy)
  if awk -v l="$l1" 'BEGIN{exit !(l <= 1.0)}' && [ -z "$b" ]; then echo "$(ts) idle: load1=$l1"; return 0; fi
  [ $((w % 60)) -eq 0 ] && echo "$(ts) waiting: load1=$l1 busy=[$b] (${w}s)"; [ "$w" -ge 3600 ] && { echo "$(ts) NOT IDLE — stopping"; exit 1; }
  sleep 10; w=$((w + 10)); done; }
echo "$(ts) start $(git rev-parse --short HEAD) (+ working tree: $(git status --short decoder | tr '\n' ' '))"
for DEPTH in 128 512; do
  gate
  echo "== $(ts) S1 depth $DEPTH; loadavg $(cat /proc/loadavg)"
  GOINFER_CPUBATCH_S1_MODEL=$M GOINFER_CPUBATCH_S1_DEPTH=$DEPTH go test -tags goinfer_testhooks -count=1 -timeout 90m \
    -run '^TestCPUBatchS1_fusedVsUnfused$' -v ./decoder/ 2>&1 | tee $D/s1-d$DEPTH.log | grep -E '^\[s1|^--- |^ok|FAIL|panic'
done
echo "$(ts) DONE"
