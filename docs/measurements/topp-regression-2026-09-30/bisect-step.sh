#!/bin/bash
# One bisect step: build the CUDA server at HEAD of this worktree, time the 0.5B top-p cell, print the rate.
# exit 0 = good (> 290 tok/s), 1 = bad (< 260), 125 = skip (no build, or a reading in between).
export PATH=$PATH:/usr/local/go/bin
W=$HOME/goinfer-bench/topp-regression-2026-09-30
C=$(git rev-parse --short=8 HEAD)
rm -f go.work go.work.sum; go work init . ./cuda ./gpu 2>/dev/null || go work init . ./cuda
if ! (cd cuda && CGO_ENABLED=0 go build -tags cuda -o $W/serve-bisect ./cmd/serve) >> $W/bisect-build.log 2>&1; then echo "$C build failed: skip" | tee -a $W/bisect-steps.log; rm -f go.work go.work.sum; exit 125; fi
rm -f go.work go.work.sum
cd $HOME/goinfer-bench/peer-sweep-2026-09-29/wt-754f12d3
GOINFER_SERVE_CUDA=$W/serve-bisect BENCH_RUNS=1 BENCH_ENGINES=goinfer BENCH_BACKENDS=cuda BENCH_DEPTHS=none BENCH_MODELS=0.5B \
  BENCH_CONFIGS=temp0.8_topp0.95 BENCH_MAX_LOADAVG=4 BENCH_SERVE_LOG_DIR=$W/serve-logs-bisect \
  python3 scripts/bench_peer.py $W/bisect-$C.json > $W/bisect-$C.log 2>&1
tok=$(python3 -c "
import json,sys
r=[c for c in json.load(open('$W/bisect-$C.json'))[1:] if c.get('config')=='temp0.8_topp0.95' and c.get('runs')]
print(round(r[0]['runs'][0],1) if r else 0)")
if awk -v t=$tok 'BEGIN{exit !(t>290)}'; then v=good; code=0; elif awk -v t=$tok 'BEGIN{exit !(t>0 && t<260)}'; then v=bad; code=1; else v=skip; code=125; fi
echo "$(date +%H:%M:%S) $C top-p ${tok} tok/s -> $v" | tee -a $W/bisect-steps.log
exit $code
