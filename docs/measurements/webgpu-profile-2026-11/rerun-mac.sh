#!/bin/bash
# WG1 Mac: decode run 2 re-run in place (run 2 void: sleep 11:49:24-11:49:44). Waits for AC with the lid open, latest start 20:00 PST.
set -u
W=$HOME/goinfer-bench/wg1-mac
R=$W/results
M=$HOME/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf
ready() { pmset -g batt | head -1 | grep -q "AC Power" && ioreg -r -k AppleClamshellState -d 4 | grep -q '"AppleClamshellState" = No'; }
if [ "${1:-}" = wait ]; then # outside the lock: wait for AC with the lid open, then take the exclusive lock for the run itself
  until ready; do
    [ "$(date +%H%M)" -ge 2000 ] && { echo "$(date '+%T %Z') no AC with the lid open by 20:00: re-run not started" | tee -a $R/provenance.txt; exit 75; }
    sleep 180
  done
  exec python3 $HOME/goinfer-runner/scripts/timing_lock.py run --wait 1800 --label wg1-mac-rerun -- bash "$0" run
fi
ready || { echo "$(date '+%T %Z') lost AC or the lid closed before the run: not started" | tee -a $R/provenance.txt; exit 75; }
st() { echo "$(date '+%T %Z') load $(sysctl -n vm.loadavg | tr -d '{}' | awk '{print $1}') swap $(sysctl -n vm.swapusage | sed -E 's/.*used = ([0-9.]+)M.*/\1/') MB $(pmset -g batt | head -1 | sed "s/.*'\(.*\)'.*/\1/")"; }
t0=$(date '+%Y-%m-%d %H:%M:%S')
echo "== decode run 2 re-run start $(st)" | tee -a $R/provenance.txt
cd $W/wt/backend/webgpu
caffeinate -i env GOINFER_HEAVY_TESTS=1 GOINFER_DECODE_GGUF=$M GOINFER_DECODE_POS=128,2048,8192,16384 $W/bin/webgpu.test -test.run '^TestDecode_dispatchProfile$' -test.v -test.count=1 -test.timeout 60m > $R/decode-run2-rerun.log 2>&1
rc=$?
t1=$(date '+%Y-%m-%d %H:%M:%S')
echo "== decode run 2 re-run end rc=$rc $(st)" | tee -a $R/provenance.txt
# The event type is pmset's fourth column; a sleep, wake or dark wake there, or any line on battery, voids the run. Matching
# the whole line would also match the "Idle Sleep Preventer" assertions the kernel logs while awake.
ev=$(pmset -g log | awk -v a="$t0" -v b="$t1" '{ts=$1" "$2} ts>=a && ts<=b && ($4=="Sleep" || $4=="Wake" || $4=="DarkWake" || /Using Batt/)')
if [ -n "$ev" ]; then echo "VOID: sleep, wake or battery during the re-run:"; echo "$ev"; else echo "no sleep, wake or battery during the re-run"; fi | tee -a $R/provenance.txt
