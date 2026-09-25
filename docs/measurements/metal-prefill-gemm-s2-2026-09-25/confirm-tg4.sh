#!/bin/bash
# R16 CONFIRMATION RUN (amendment 2026-09-25): prototype 4 (gemm_w4f16_tg4) alone, fresh session, current kernel as the
# do-nothing arm, K=512, 1.5B, 7 paired reps (fixed before the run). This run grades; the exploratory run only selected.
cd /Users/francistownsend-merino/tmcode/goinfer/metal || exit 1
w=0; while :; do l=$(sysctl -n vm.loadavg | awk '{print $2}'); awk -v l="$l" 'BEGIN{exit !(l<=2.0)}' && { echo "$(date +%T) idle load1=$l"; break; }; [ $w -ge 1800 ] && { echo "NOT IDLE after 1800s"; exit 1; }; [ $((w%60)) -eq 0 ] && echo "$(date +%T) waiting load1=$l"; sleep 10; w=$((w+10)); done
echo "$(date +%T) CONFIRMATION start commit $(git rev-parse --short=8 HEAD) dirty=$(git status --porcelain -- . | wc -l | tr -d ' '); $(memory_pressure | tail -1)"
GOINFER_METAL_DECOMP=1 GOINFER_METAL_DECOMP_K=512 GOINFER_METAL_DECOMP_REPS=7 GOINFER_METAL_S2=1 GOINFER_METAL_S2_KERNEL=gemm_w4f16_tg4 \
  go test -tags goinfer_testhooks -run '^TestMetalPrefillDecomp$' -v -count=1 -timeout 30m . 2>&1
echo "$(date +%T) exit=$?; $(memory_pressure | tail -1)"
