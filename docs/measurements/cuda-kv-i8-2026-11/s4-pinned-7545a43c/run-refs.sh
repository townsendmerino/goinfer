#!/usr/bin/env bash
# S4 step 1: set-B references on the pinned tree 7545a43c, S at K=3900 then D7 at K=8000; CPU only, timing lock shared.
set -u
P=$HOME/goinfer-bench/s4-pinned
export PATH=/usr/local/go/bin:$HOME/go/bin:$PATH
cd $P/wt/decoder
for cell in S:3900 D7:8000; do
  m=${cell%%:*}; k=${cell#*:}
  echo "refs $m K=$k start $(date '+%F %T %Z') load $(cut -d' ' -f1-3 /proc/loadavg)"
  GOINFER_HEAVY_TESTS=1 GOINFER_PREFILL_GATE_PROMPTS=b GOINFER_CPU_REF_MODELS=$m GOINFER_CPU_REF_KS=$k \
    python3 $P/wt/scripts/timing_lock.py run --shared --label s4-refs-$m -- $P/bin/decoder.test -test.run '^TestPrefillGateReference$' -test.v -test.count=1 -test.timeout 6h \
    > $P/logs/refs-$m-K$k.log 2>&1
  rc=$?
  echo "refs $m K=$k end $(date '+%F %T %Z') exit $rc"
  if [ $rc -ne 0 ]; then echo "REFS FAILED: $m"; echo failed > $P/logs/refs.failed; exit $rc; fi
done
date '+%F %T %Z' > $P/logs/refs.done
echo "refs DONE"
