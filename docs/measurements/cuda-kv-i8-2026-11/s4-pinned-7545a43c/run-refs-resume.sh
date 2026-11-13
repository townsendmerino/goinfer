#!/usr/bin/env bash
# S4 step 1, resumed: D7 at K=8000 on the pinned tree 7545a43c. Pass 1 (logs/refs-D7-K8000-pass1.log) stored 8 of 10 prompts in
# ~3h55m and its deadline guard declined to start the last 2; this run computes those 2 from the same binary and cache. CPU only, lock shared.
set -u
P=$HOME/goinfer-bench/s4-pinned
export PATH=/usr/local/go/bin:$HOME/go/bin:$PATH
cd $P/wt/decoder
echo "refs D7 K=8000 resume start $(date '+%F %T %Z') load $(cut -d' ' -f1-3 /proc/loadavg)"
GOINFER_HEAVY_TESTS=1 GOINFER_PREFILL_GATE_PROMPTS=b GOINFER_CPU_REF_MODELS=D7 GOINFER_CPU_REF_KS=8000 \
  python3 $P/wt/scripts/timing_lock.py run --shared --label s4-refs-D7-resume -- $P/bin/decoder.test -test.run '^TestPrefillGateReference$' -test.v -test.count=1 -test.timeout 6h \
  > $P/logs/refs-D7-K8000.log 2>&1
rc=$?
echo "refs D7 K=8000 resume end $(date '+%F %T %Z') exit $rc"
if [ $rc -ne 0 ]; then echo "REFS FAILED: D7 resume"; echo failed > $P/logs/refs.failed; exit $rc; fi
date '+%F %T %Z' > $P/logs/refs.done
echo "refs DONE"
