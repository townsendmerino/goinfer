#!/usr/bin/env bash
# The attention-share profile (README § "Profile"). Run under the exclusive timing lock:
#   python3 scripts/timing_lock.py run --wait 14400 --label cpu-attn-share -- bash docs/measurements/cpu-kv-f16-2026-11-23/run_profile.sh <decoder test binary> <log>
set -euo pipefail
bin=$1 log=$2
repo=$(cd "$(dirname "$0")/../../.." && pwd)
for _ in $(seq 60); do
  awk '{exit !($1 < 1.0)}' /proc/loadavg && break
  sleep 10
done
{
  echo "start $(date -Iseconds) load $(cut -d' ' -f1-3 /proc/loadavg) memavail_mib $(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo) swap_used_mib $(awk '/SwapTotal/{t=$2}/SwapFree/{f=$2}END{print int((t-f)/1024)}' /proc/meminfo)"
  cd "$repo/decoder"
  CPU_ATTN_PROBE=1 GOINFER_PREQUANT_GGUF=$HOME/models/phi3-mini-4k-gguf/Phi-3-mini-4k-instruct-q4.gguf GOINFER_BENCH_QUANT=q4k \
    GOINFER_BENCH_BACKEND=cpu GOINFER_BENCH_DEPTH=3900 CPU_ATTN_PROBE_TOKENS=32 \
    /usr/bin/time -v "$bin" -test.run '^TestCPUAttnShareProbe$' -test.v -test.timeout 30m
  echo "end $(date -Iseconds) swap_used_mib $(awk '/SwapTotal/{t=$2}/SwapFree/{f=$2}END{print int((t-f)/1024)}' /proc/meminfo)"
} > "$log" 2>&1
