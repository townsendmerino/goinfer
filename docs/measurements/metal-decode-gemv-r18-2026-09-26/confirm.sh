#!/bin/bash
cd /Users/francistownsend-merino/tmcode/goinfer
export GOWORK=/Users/francistownsend-merino/tmcode/goinfer/go.work
for M in 1.5b 7b; do
  if [ $M = 1.5b ]; then P=$HOME/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf; else P=$HOME/models/qwen2.5-7b-instruct-q4_k_m.int4.metal.giw; fi
  {
    echo "$(date '+%F %T %Z') start $(git rev-parse --short HEAD) (tree: $(git status --short metal/ | tr '\n' ' ')); load1=$(sysctl -n vm.loadavg | awk '{print $2}'); CONFIRMATION $M: i2244, 7 reps, depths 128,2048,3900"
    GOINFER_METAL_R18_SEQ=1 GOINFER_METAL_R18_MODEL=$P GOINFER_METAL_R18_CANDS=i2244 GOINFER_METAL_R18_REPS=7 GOINFER_METAL_R18_DEPTHS=128,2048,3900 GOINFER_METAL_R18_IDLE=3 /usr/local/go/bin/go test -tags goinfer_testhooks -count=1 -timeout 45m -v -run '^TestR18InSequence$' ./metal/ 2>&1
    echo "$(date '+%F %T %Z') end $M exit $?"
  } > /Users/francistownsend-merino/goinfer-bench/metal-decode-gemv-r18-2026-09-26/confirm-$M.log
done
