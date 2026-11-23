#!/usr/bin/env bash
# hd96 gate 2 (docs/measurements/fa-hd96-phi3-2026-11-22/README.md): settle under load 1.0, record the start state, then the lane's registered
# fidelity gate on the P3M cell (Phi-3 mini q4k, f16 KV, K=3900, set B) under the exclusive timing lock.
set -uo pipefail
L=$HOME/goinfer-bench/fa-hd96-run
echo "queued for the lock $(date '+%F %T %Z') load $(cut -d' ' -f1-3 /proc/loadavg) MemAvailable $(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo) MiB gpu $(nvidia-smi --query-compute-apps=process_name,used_memory --format=csv,noheader | tr '\n' ';')"
cd $HOME/goinfer-bench/fa-hd96-wt/backend/cuda
GOINFER_HEAVY_TESTS=1 GOINFER_PREFILL_GATE_PROMPTS=b GOINFER_CUDA_FLASH_DECODE=16 GOINFER_FA_GATE_CELLS=P3M:3900 \
  python3 ../../scripts/timing_lock.py run --wait 2700 --label hd96-gate2 -- bash -c '
    t0=$(date +%s); while awk -v l="$(cut -d" " -f1 /proc/loadavg)" "BEGIN{exit !(l>=1.0)}" && [ $(( $(date +%s) - t0 )) -lt 600 ]; do sleep 15; done
    echo "gate start (lock held) $(date "+%F %T %Z") load $(cut -d" " -f1-3 /proc/loadavg) MemAvailable $(awk "/MemAvailable/{print int(\$2/1024)}" /proc/meminfo) MiB"
    exec '"$L"'/bin/cuda-8c4b42ab7.test -test.run "^TestFlashDecodeGateVsReference$" -test.v -test.timeout 1h' 
echo "gate end $(date '+%F %T %Z') rc=$?"
