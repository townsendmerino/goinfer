#!/bin/bash
# step 2: one device test per invocation, exclusive timing lock, -v, logs archived
cd ~/wt/v1-2026-11-12/backend/cuda || exit 1
L=~/goinfer-logs/v1-2026-11-12
B=$L/bin/cuda.test
run() { # name regex
  local n=$1 re=$2 s=$(date +%s)
  echo "=== $(date +%T) START $n"
  python3 ../../scripts/timing_lock.py run --label "v1-$n" -- $B -test.v -test.run "$re" -test.timeout 12m > $L/02-$n.log 2>&1
  echo "=== $(date +%T) END $n rc=$? ($(( $(date +%s)-s ))s): $(grep -cE '^\s*--- PASS' $L/02-$n.log) PASS, $(grep -cE '^\s*--- FAIL' $L/02-$n.log) FAIL, $(grep -cE '^\s*--- SKIP' $L/02-$n.log) SKIP"
}
run layerwalk '^TestCheckCUDALayerWalk_TopologiesAndRejections$'
run ragged '^TestCUDAStepRagged_rowsMatchTheirShapes$'
run kv4scrambled '^TestKV4_scrambledTables$'
run kv4shared '^TestKV4_sharedPreambleStoredOnce$'
run kv7slot '^TestKV7_slotWriteTable$'
run kvwindow-replay '^TestWindowedKV_i3Replay$'
run flintparity '^TestFlintResidentParityCUDA$'
echo "=== ALL DONE $(date +%T)"
