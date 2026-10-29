#!/usr/bin/env bash
# The CUDA suite's I1 breaks, by call site: cuda-parity (20m) then cuda-heavy-small, as gate gpu runs each cell, with the
# unplanned-lease hook naming each break's creating test and allocation sites (LEDGER BREAK lines). Waits before each step
# until no other process holds a CUDA context besides the desktop's.
set -u
cd "$(dirname "$0")/../../.."
export PATH=/usr/local/go/bin:$PATH CGO_ENABLED=0
D=docs/measurements/sb12-i1i2-2026-10-29
R=docs/measurements/sb12-ledger-guard-2026-10-28/heavy-small.run
unset GOINFER_HEAVY_TESTS GOINFER_DRAIN_GROUP
gpu_free() {
  while :; do
    other=$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader | grep -v -E "kwin_wayland|Xwayland|plasmashell")
    [ -z "$other" ] && return
    echo "   waiting $(date '+%H:%M:%S'): other CUDA process(es): $other"
    sleep 60
  done
}
gpu_free
echo "== cuda-parity start $(date '+%F %H:%M:%S %Z') $(git rev-parse --short HEAD) load $(cut -d' ' -f1 /proc/loadavg)"
python3 scripts/timing_lock.py run --shared --wait 3600 --label i1i2-parity -- \
  go test -tags 'cuda goinfer_testhooks' -count=1 -timeout 20m -v ./cuda/ > $D/sites-parity.log 2>&1
echo "== cuda-parity exit $? $(date '+%F %H:%M:%S %Z')"
gpu_free
echo "== cuda-heavy-small start $(date '+%F %H:%M:%S %Z') load $(cut -d' ' -f1 /proc/loadavg)"
GOINFER_HEAVY_TESTS=1 GOINFER_ISOLATE_DEFER=1 python3 scripts/timing_lock.py run --shared --wait 3600 --label i1i2-heavy-small -- \
  go test -tags 'cuda goinfer_testhooks' -count=1 -timeout 120m -v -run "$(cat $R)" ./cuda/ > $D/sites-heavy-small.log 2>&1
echo "== cuda-heavy-small exit $? $(date '+%F %H:%M:%S %Z')"
