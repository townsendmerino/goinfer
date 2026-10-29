#!/usr/bin/env bash
# SB12-L3: cuda-parity, then cuda-heavy-small, each as `gate gpu` runs its cell (cmd/gate/gpu.go: cudaParity, runHeavyGroup).
# heavy-small.run is the small group's -run regex from the gate's own partition (partitionHeavy) at this commit.
set -u
cd "$(dirname "$0")/../../.."
export PATH=/usr/local/go/bin:$PATH CGO_ENABLED=0
D=docs/measurements/sb12-ledger-guard-2026-10-28
unset GOINFER_HEAVY_TESTS GOINFER_DRAIN_GROUP
echo "== cuda-parity start $(date '+%F %H:%M:%S %Z') $(git rev-parse --short HEAD) load $(cut -d' ' -f1 /proc/loadavg)"
python3 scripts/timing_lock.py run --shared --label sb12-l3-parity -- \
  go test -tags 'cuda goinfer_testhooks' -count=1 -timeout 10m -v ./cuda/ > $D/l3-cuda-parity.log 2>&1
echo "== cuda-parity exit $? $(date '+%F %H:%M:%S %Z')"
echo "== cuda-heavy-small start $(date '+%F %H:%M:%S %Z') load $(cut -d' ' -f1 /proc/loadavg)"
GOINFER_HEAVY_TESTS=1 GOINFER_ISOLATE_DEFER=1 python3 scripts/timing_lock.py run --shared --label sb12-l3-heavy-small -- \
  go test -tags 'cuda goinfer_testhooks' -count=1 -timeout 120m -v -run "$(cat $D/heavy-small.run)" ./cuda/ > $D/l3-cuda-heavy-small.log 2>&1
echo "== cuda-heavy-small exit $? $(date '+%F %H:%M:%S %Z')"
