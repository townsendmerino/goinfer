#!/bin/bash
cd /Users/francistownsend-merino/tmcode/goinfer/metal || exit 1
w=0; while :; do l=$(sysctl -n vm.loadavg | awk '{print $2}'); awk -v l="$l" 'BEGIN{exit !(l<=2.0)}' && { echo "$(date +%T) idle load1=$l"; break; }; [ $w -ge 1800 ] && { echo "NOT IDLE"; exit 1; }; [ $((w%60)) -eq 0 ] && echo "$(date +%T) waiting load1=$l"; sleep 10; w=$((w+10)); done
echo "$(date +%T) start commit $(git rev-parse --short=8 HEAD) dirty(tracked)=$(git status --porcelain | grep -v '^??' | awk '{print $2}' | tr '\n' ' ')"
GOINFER_METAL_DDECOMP=1 GOINFER_METAL_DDECOMP_DEPTHS=2048,3900 GOINFER_METAL_DDECOMP_SPLITS=0,20,28,32,48,64 \
  go test -tags goinfer_testhooks -run '^TestMetalDecodeDecomp$' -v -count=1 -timeout 40m . 2>&1
echo "$(date +%T) exit=$?"
