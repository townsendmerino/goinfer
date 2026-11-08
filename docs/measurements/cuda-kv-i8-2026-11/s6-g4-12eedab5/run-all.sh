#!/usr/bin/env bash
# S6 (docs/measurements/cuda-kv-i8-2026-11/s6-registration.md): G4 then the speed cells, on the pinned 7545a43c + the S6 commits (12eedab5), whose
# reference source equals the pinned tree's, so set B's references are current. Binaries built once here. The gate holds the timing lock
# exclusively; the speed check (run-speed.sh: fit checks by serve's banner first, then two A B B A passes through bench_peer) only runs if the gate
# binary passed (its preconditions held).
set -uo pipefail
P=$HOME/goinfer-bench/s6-g4; WT=$P/wt; L=$P/logs
export PATH=/usr/local/go/bin:$HOME/go/bin:$PATH
echo "== build start $(date '+%F %T %Z') rev $(git -C $WT rev-parse HEAD)"
(cd $WT/backend/cuda && CGO_ENABLED=0 go test -c -tags 'cuda goinfer_testhooks' -o $P/bin/cuda.test . && CGO_ENABLED=0 go build -tags cuda -o $P/bin/serve-cuda-s6 ./cmd/serve) > $L/build.log 2>&1 || { echo "STOP: build failed"; tail $L/build.log; exit 1; }
echo "== gate start $(date '+%F %T %Z') load $(cut -d' ' -f1-3 /proc/loadavg) MemAvailable $(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo) MiB"
(cd $WT/backend/cuda && GOINFER_HEAVY_TESTS=1 GOINFER_PREFILL_GATE_PROMPTS=b GOINFER_CUDA_FLASH_DECODE=16 GOINFER_KV8_GATE_CELLS=S:3900,D7:8000 \
  python3 $WT/scripts/timing_lock.py run --label s6-gate -- $P/bin/cuda.test -test.run '^TestKV8GateVsReference$' -test.v -test.count=1 -test.timeout 3h) > $L/gate.log 2>&1
echo "== gate end $(date '+%F %T %Z') exit $?"
grep -E '=== KV8|--- (FAIL|PASS)' $L/gate.log | tee $L/gate.summary
grep -q -- '--- PASS: TestKV8GateVsReference ' $L/gate.log || { echo "STOP: the gate binary did not pass (a precondition, a refusal or a crash); the speed check does not run"; exit 1; }
bash $P/run-speed.sh > $L/run-speed.log 2>&1
echo "== speed exit $? $(date '+%F %T %Z')"
