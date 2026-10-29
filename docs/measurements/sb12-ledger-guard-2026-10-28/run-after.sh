#!/usr/bin/env bash
# After run-l3.sh: cuda-parity again with its timeout stated (20m; the cell's 10m fired after the ledger test passed, so the
# release check was unread: SB12-L3's void rule), then the L1 plant at fc9fc5e8b's tree, then run-seven.sh.
set -u
cd "$(dirname "$0")/../../.."
export PATH=/usr/local/go/bin:$PATH CGO_ENABLED=0
D=docs/measurements/sb12-ledger-guard-2026-10-28
until grep -q "cuda-heavy-small exit" $D/l3-driver.log; do sleep 30; done
echo "== cuda-parity (20m) start $(date '+%F %H:%M:%S %Z') $(git rev-parse --short HEAD) load $(cut -d' ' -f1 /proc/loadavg)"
python3 scripts/timing_lock.py run --shared --label sb12-l3-parity-20m -- \
  go test -tags 'cuda goinfer_testhooks' -count=1 -timeout 20m -v ./cuda/ > $D/l3-cuda-parity-20m.log 2>&1
echo "== cuda-parity (20m) exit $? $(date '+%F %H:%M:%S %Z')"
python3 scripts/timing_lock.py run --shared --label sb12-l1b -- go test -tags 'cuda goinfer_testhooks' -count=1 -timeout 5m -v \
  -run '^(TestLedger_openDeviceIsNamed|TestLedger_unplannedIsReported|TestLedger_closesAgainstDevice)$' ./cuda/ > $D/l1-plant-b.log 2>&1
echo "== l1 plant b exit $? $(date '+%F %H:%M:%S %Z')"
bash $D/run-seven.sh > $D/seven.log 2>&1
echo "== seven done $(date '+%F %H:%M:%S %Z')"
