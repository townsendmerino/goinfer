#!/bin/bash
# R16 production wiring gates (tree: the uncommitted wiring of prototype 4 as gemm_w4f16_store).
cd /Users/francistownsend-merino/tmcode/goinfer/metal || exit 1
wait_idle() { local w=0; while :; do l=$(sysctl -n vm.loadavg | awk '{print $2}'); awk -v l="$l" 'BEGIN{exit !(l<=2.0)}' && { echo "$(date +%T) idle load1=$l"; return 0; }; [ $w -ge 1800 ] && { echo "NOT IDLE after 1800s"; return 1; }; [ $((w%60)) -eq 0 ] && echo "$(date +%T) waiting load1=$l"; sleep 10; w=$((w+10)); done; }
echo "$(date +%T) start commit $(git rev-parse --short=8 HEAD) dirty=$(git status --porcelain | grep -v '^??' | wc -l | tr -d ' ') ($(git status --porcelain | grep -v '^??' | awk '{print $2}' | tr '\n' ' '))"
wait_idle || exit 1
echo "$(date +%T) ##### GATE 1: §3.2 pooled fidelity gate, S"
GOINFER_HEAVY_TESTS=1 go test -tags goinfer_testhooks -run '^TestPrefillGateVsReference$/^S$' -v -count=1 -timeout 2h . 2>&1
echo "$(date +%T) ##### GATE 1 exit=$?"
wait_idle || exit 1
echo "$(date +%T) ##### GATE 2: real 1.5B, retired kernel (r15) vs production, K=512, 5 reps"
GOINFER_METAL_DECOMP=1 GOINFER_METAL_DECOMP_K=512 GOINFER_METAL_S2=1 GOINFER_METAL_S2_KERNEL=gemm_w4f16_store_r15 \
  go test -tags goinfer_testhooks -run '^TestMetalPrefillDecomp$' -v -count=1 -timeout 30m . 2>&1
echo "$(date +%T) ##### GATE 2 exit=$?"
wait_idle || exit 1
echo "$(date +%T) ##### GATE 3: full metal suite (non-heavy)"
go test -tags goinfer_testhooks -count=1 -v -timeout 40m . > $HOME/goinfer-bench/metal-gemm-s2-2026-09-25/metal-suite.log 2>&1
rc=$?
echo "$(date +%T) ##### GATE 3 exit=$rc pass=$(grep -c '^--- PASS' $HOME/goinfer-bench/metal-gemm-s2-2026-09-25/metal-suite.log) skip=$(grep -c '^--- SKIP' $HOME/goinfer-bench/metal-gemm-s2-2026-09-25/metal-suite.log) fail=$(grep -c '^--- FAIL' $HOME/goinfer-bench/metal-gemm-s2-2026-09-25/metal-suite.log)"
grep -E '^--- FAIL' $HOME/goinfer-bench/metal-gemm-s2-2026-09-25/metal-suite.log | head
echo "$(date +%T) DONE"
