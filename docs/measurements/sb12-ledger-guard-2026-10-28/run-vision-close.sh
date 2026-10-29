#!/usr/bin/env bash
# The VisionEncoder.Close fix, red first: TestSiglipCUDA_int8StaysTheInt8Tower at f0167f6a0 (test only, in a detached
# worktree beside this one) must fail, and at ce1a5b68f (the fix) pass. Waits for the timing lock and for no other CUDA
# process besides the desktop's.
set -u
cd "$(dirname "$0")/../../.."
export PATH=/usr/local/go/bin:$PATH CGO_ENABLED=0
D=docs/measurements/sb12-ledger-guard-2026-10-28
RED=${RED:-$HOME/goinfer-bench/nobara-ci01-vered-wt}
gpu_free() {
  while :; do
    other=$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader | grep -v -E "kwin_wayland|Xwayland|plasmashell")
    held=$(python3 scripts/timing_lock.py status | grep -c '^HELD')
    [ -z "$other" ] && [ "$held" = 0 ] && return
    echo "   waiting $(date '+%H:%M:%S'): lock held=$held; other CUDA: ${other:-none}"
    sleep 60
  done
}
T="go test -tags cuda,goinfer_testhooks -count=1 -timeout 5m -v -run ^TestSiglipCUDA_int8StaysTheInt8Tower$ ./cuda/"
gpu_free
echo "== red at $(git -C $RED rev-parse --short HEAD) start $(date '+%F %H:%M:%S %Z') load $(cut -d' ' -f1 /proc/loadavg)"
(cd $RED && python3 scripts/timing_lock.py run --shared --wait 3600 --label sb12-vefix-red -- $T) > $D/vision-close-red.log 2>&1
echo "== red exit $? $(date '+%H:%M:%S')"
gpu_free
echo "== green at $(git rev-parse --short HEAD) start $(date '+%F %H:%M:%S %Z') load $(cut -d' ' -f1 /proc/loadavg)"
python3 scripts/timing_lock.py run --shared --wait 3600 --label sb12-vefix-green -- $T > $D/vision-close-green.log 2>&1
echo "== green exit $? $(date '+%H:%M:%S')"
