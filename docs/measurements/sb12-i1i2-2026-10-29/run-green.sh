#!/usr/bin/env bash
# The I1 fixes, green: after run-sites.sh's heavy-small (the red), at this branch's tip: cuda-parity in full, heavy-small's
# nine named tests in one process, then the three heavy groups whose I1 breaks the 2026-10-27 night printed (their -run lists
# are that night's top-level tests), each as gate gpu runs a heavy cell. A green group prints no LEDGER BREAK and no I1 line.
set -u
cd "$(dirname "$0")/../../.."
export PATH=/usr/local/go/bin:$PATH CGO_ENABLED=0
D=docs/measurements/sb12-i1i2-2026-10-29
unset GOINFER_HEAVY_TESTS GOINFER_DRAIN_GROUP
until grep -q "cuda-heavy-small exit" $D/sites-driver.log; do sleep 30; done
gpu_free() {
  while :; do
    other=$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader | grep -v -E "kwin_wayland|Xwayland|plasmashell")
    [ -z "$other" ] && return
    echo "   waiting $(date '+%H:%M:%S'): other CUDA process(es): $other"
    sleep 60
  done
}
T="go test -tags cuda,goinfer_testhooks -count=1 -v"
gpu_free
echo "== cuda-parity start $(date '+%F %H:%M:%S %Z') $(git rev-parse --short HEAD) load $(cut -d' ' -f1 /proc/loadavg)"
python3 scripts/timing_lock.py run --shared --wait 3600 --label i1i2-green-parity -- $T -timeout 20m ./cuda/ > $D/green-parity.log 2>&1
echo "== cuda-parity exit $? $(date '+%F %H:%M:%S %Z')"
for g in small-named qwen25vl-3b 7b-8b qwen3-4b; do
  run=$D/heavy-$g.run
  gpu_free
  echo "== $g start $(date '+%F %H:%M:%S %Z') load $(cut -d' ' -f1 /proc/loadavg)"
  GOINFER_HEAVY_TESTS=1 GOINFER_ISOLATE_DEFER=1 python3 scripts/timing_lock.py run --shared --wait 3600 --label i1i2-green-$g -- \
    $T -timeout 120m -run "$(cat $run)" ./cuda/ > $D/green-$g.log 2>&1
  echo "== $g exit $? $(date '+%F %H:%M:%S %Z')"
done
