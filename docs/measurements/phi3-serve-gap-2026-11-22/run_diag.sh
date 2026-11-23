#!/usr/bin/env bash
# The gap's diagnosis (README "Diagnosis"): TestGapDiag's four arms at depth 128, one process each, in the order
# D1a D1b D2a D2b D2b D2a D1b D1a, under the exclusive timing lock; each cell waits for a 1-minute load under 1.0 and logs its start state.
#   bash run_diag.sh <cuda-gapdiag test binary> <out dir>
set -uo pipefail
BIN=$1 OUT=$2
mkdir -p "$OUT"
REPO=$(cd "$(dirname "$0")/../../.." && pwd)
export GOINFER_BENCH_MODEL=$HOME/models/phi3-mini-4k-gguf/Phi-3-mini-4k-instruct-q4.gguf GOINFER_BENCH_DEPTH=128 GAP_SERVE_AUTO=1
declare -A OPTS=([D1a]=bench [D1b]=serve [D2a]=serve [D2b]=serve) MODE=([D1a]=argmax [D1b]=argmax [D2a]=gen [D2b]=gen) MC3=([D1a]=0 [D1b]=0 [D2a]=0 [D2b]=2)
i=0
for arm in D1a D1b D2a D2b D2b D2a D1b D1a; do
  i=$((i+1))
  t0=$(date +%s)
  while awk -v l="$(cut -d' ' -f1 /proc/loadavg)" 'BEGIN{exit !(l>=1.0)}' && [ $(( $(date +%s) - t0 )) -lt 600 ]; do sleep 10; done
  echo "cell $i $arm start $(date '+%T') load $(cut -d' ' -f1 /proc/loadavg) MemAvailable $(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo) MiB"
  (cd "$REPO/backend/cuda" && GAPD_OPTS=${OPTS[$arm]} GAPD_MODE=${MODE[$arm]} GAPD_MC3=${MC3[$arm]} "$BIN" -test.run '^TestGapDiag$' -test.v -test.timeout 10m) > "$OUT/c$i-$arm.log" 2>&1
  echo "cell $i $arm rc=$? $(grep -h 'gapdiag:' "$OUT/c$i-$arm.log" | head -1)"
done
echo DIAG-DONE
