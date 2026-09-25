#!/bin/bash
w=0; while :; do l=$(sysctl -n vm.loadavg | awk '{print $2}'); awk -v l="$l" 'BEGIN{exit !(l<=2.0)}' && { echo "$(date +%T) idle load1=$l"; break; }; [ $w -ge 1800 ] && { echo "NOT IDLE after 1800s"; exit 1; }; [ $((w%60)) -eq 0 ] && echo "$(date +%T) waiting load1=$l"; sleep 10; w=$((w+10)); done
echo "$(date +%T) start; macOS $(sw_vers -productVersion); $(memory_pressure | tail -1)"
$HOME/goinfer-bench/metal-gemm-ceiling-2026-09-25/mps_gemm_ceiling 5 2>&1
echo "$(date +%T) exit=$?"
