#!/usr/bin/env bash
# G46 (docs/measurements/g46-gemma3-gemm-fidelity-2026-10-21.md): Gemma-3-4B's default-settings prefill against the per-token path,
# six depths below the shared-memory bound, two prompts. Builds the test binary from the tree it is run in, runs it under the timing lock,
# and samples the test process's own VmHWM and VmSwap every 5 s. Run from the repo root: bash docs/measurements/g46-.../run-g46.sh
set -eo pipefail
cd "$(git rev-parse --show-toplevel)"
D=docs/measurements/g46-gemma3-gemm-fidelity-2026-10-21
BIN="$HOME/goinfer-bench/g46/cuda.test"
mkdir -p "$(dirname "$BIN")" "$D/run"
rev=$(git rev-parse --short=8 HEAD)
{
  echo "commit $rev  go $(go version | cut -d' ' -f3)  driver $(nvidia-smi --query-gpu=driver_version --format=csv,noheader)  start $(date -Is)"
  echo "load1 $(cut -d' ' -f1 /proc/loadavg)  MemAvailable_kB $(awk '/MemAvailable/{print $2}' /proc/meminfo)  gpu_free_MiB $(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)"
} | tee "$D/run/start.txt"
go test -c -tags 'cuda goinfer_testhooks' -o "$BIN" ./backend/cuda/
export BIN D
cd backend/cuda
rc=0
GOINFER_HEAVY_TESTS=1 G46_REAL=1 python3 ../../scripts/timing_lock.py run --label g46 -- bash -c '
  "$BIN" -test.v -test.count=1 -test.timeout 30m -test.run "^TestG46_gemma3GEMMFidelityLadder$" > "../../$D/run/g46.log" 2>&1 &
  pid=$!
  : > "../../$D/run/proc-samples.txt"
  while kill -0 $pid 2>/dev/null; do
    printf "%s " "$(date +%T)" >> "../../$D/run/proc-samples.txt"
    grep -E "^(VmHWM|VmSwap)" /proc/$pid/status 2>/dev/null | tr "\n\t" "  " >> "../../$D/run/proc-samples.txt" || true
    echo >> "../../$D/run/proc-samples.txt"
    sleep 5
  done
  wait $pid
' || rc=$?
echo "peak VmHWM / VmSwap (kB) of the test process:"
awk 'BEGIN{h=0;s=0} {for(i=1;i<=NF;i++){if($i=="VmHWM:"&&$(i+1)>h)h=$(i+1); if($i=="VmSwap:"&&$(i+1)>s)s=$(i+1)}} END{print "VmHWM_peak_kB", h, " VmSwap_peak_kB", s}' "../../$D/run/proc-samples.txt" | tee "../../$D/run/peak.txt"
grep -E '^\[g46\]|G46 (PASS|FAIL|AMBIGUOUS)|^(--- |FAIL|ok|PASS)' "../../$D/run/g46.log" | tail -60 || true
exit $rc
