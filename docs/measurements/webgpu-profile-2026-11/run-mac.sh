#!/bin/bash
# WG1's Mac half (docs/measurements/webgpu-profile-2026-11/registration-mac.md): each instrument twice, back to back.
set -u
W=$HOME/goinfer-bench/wg1-mac
R=$W/results
M=$HOME/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf
st() { echo "$(date '+%T %Z') load $(sysctl -n vm.loadavg | tr -d '{}' | awk '{print $1}') swap $(sysctl -n vm.swapusage | sed -E 's/.*used = ([0-9.]+)M.*/\1/') MB $(pmset -g batt | head -1 | sed "s/.*'\(.*\)'.*/\1/") therm $(pmset -g therm 2>/dev/null | grep -c -i warning)"; }
case "$(python3 -c 'import os,sys; print(os.path.realpath(sys.argv[1]))' "$M")" in /Volumes/*|/srv/models*) echo "FATAL: archive path"; exit 2;; esac
pmset -g batt | head -1 | grep -q "AC Power" || { echo "FATAL: not on AC"; exit 75; }
echo "started $(st)" | tee $R/provenance.txt
cd $W/wt/backend/webgpu
for run in 1 2; do
  echo "== decode run $run start $(st)" | tee -a $R/provenance.txt
  GOINFER_HEAVY_TESTS=1 GOINFER_DECODE_GGUF=$M GOINFER_DECODE_POS=128,2048,8192,16384 $W/bin/webgpu.test -test.run '^TestDecode_dispatchProfile$' -test.v -test.count=1 -test.timeout 60m > $R/decode-run$run.log 2>&1
  echo "== decode run $run end rc=$? $(st)" | tee -a $R/provenance.txt
done
for run in 1 2; do
  echo "== prefill run $run start $(st)" | tee -a $R/provenance.txt
  GOINFER_HEAVY_TESTS=1 GOINFER_RESIDENT_GGUF=$M GOINFER_PREFILL_PROF_P=256,1024,4096 $W/bin/webgpu.test -test.run '^TestPrefill_dispatchProfile$' -test.v -test.count=1 -test.timeout 60m > $R/prefill-run$run.log 2>&1
  echo "== prefill run $run end rc=$? $(st)" | tee -a $R/provenance.txt
done
echo "finished $(st)" | tee -a $R/provenance.txt
