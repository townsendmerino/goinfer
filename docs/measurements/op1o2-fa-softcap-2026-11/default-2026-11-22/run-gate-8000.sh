#!/usr/bin/env bash
# The G22B:8000 reading (op1o2 README addendum, 22f2ab3bf): settle under load 1.0, record the start state, then the registered gate.
set -uo pipefail
L=$HOME/goinfer-bench/op1o2-default-logs
t0=$(date +%s)
while awk -v l="$(cut -d' ' -f1 /proc/loadavg)" 'BEGIN{exit !(l>=1.0)}' && [ $(( $(date +%s) - t0 )) -lt 900 ]; do sleep 15; done
echo "gate start $(date '+%F %T %Z') load $(cut -d' ' -f1-3 /proc/loadavg) MemAvailable $(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo) MiB swap $(awk '/SwapTotal/{t=$2}/SwapFree/{f=$2}END{print int((t-f)/1024)}' /proc/meminfo) MiB gpu $(nvidia-smi --query-compute-apps=process_name,used_memory --format=csv,noheader | tr '\n' ';')"
cd $HOME/goinfer-bench/op1o2-default-wt/backend/cuda
GOINFER_HEAVY_TESTS=1 GOINFER_PREFILL_GATE_PROMPTS=b GOINFER_CUDA_FLASH_DECODE=16 GOINFER_FA_GATE_CELLS=G22B:8000 \
  python3 ../../scripts/timing_lock.py run --label op1o2-g22b-gate-8000 -- $L/bin/cuda-22f2ab3bf.test -test.run '^TestFlashDecodeGateVsReference$' -test.v -test.timeout 2h
echo "gate end $(date '+%F %T %Z') rc=$?"
