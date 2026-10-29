#!/usr/bin/env bash
# SB12-L3, third round: main at ddb0b751d (#415) merged with #430 (5fb5ac444, the VisionEncoder.Close fix). cuda-parity (20m),
# then cuda-heavy-small; the release check must name no device left open. Before each step it waits until no other process
# holds a CUDA context besides the desktop's.
set -u
cd "$(dirname "$0")/../../.."
export PATH=/usr/local/go/bin:$PATH CGO_ENABLED=0
D=docs/measurements/sb12-ledger-guard-2026-10-28
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
echo "== cuda-parity (20m) start $(date '+%F %H:%M:%S %Z') load $(cut -d' ' -f1 /proc/loadavg)"
python3 scripts/timing_lock.py run --shared --wait 3600 --label sb12-l3c-parity-20m -- \
  go test -tags 'cuda goinfer_testhooks' -count=1 -timeout 20m -v ./cuda/ > $D/l3-cuda-parity-3.log 2>&1
echo "== cuda-parity (20m) exit $? $(date '+%F %H:%M:%S %Z')"
gpu_free
echo "== cuda-heavy-small start $(date '+%F %H:%M:%S %Z') $(git rev-parse --short HEAD) load $(cut -d' ' -f1 /proc/loadavg)"
GOINFER_HEAVY_TESTS=1 GOINFER_ISOLATE_DEFER=1 python3 scripts/timing_lock.py run --shared --wait 3600 --label sb12-l3c-heavy-small -- \
  go test -tags 'cuda goinfer_testhooks' -count=1 -timeout 120m -v -run "$(cat $D/heavy-small.run)" ./cuda/ > $D/l3-cuda-heavy-small-3.log 2>&1
echo "== cuda-heavy-small exit $? $(date '+%F %H:%M:%S %Z')"
