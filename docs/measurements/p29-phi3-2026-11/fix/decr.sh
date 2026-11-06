#!/bin/bash
set -u
OUT="$HOME/goinfer-work/p29k/out"
l=$(sysctl -n vm.loadavg | awk '{print $2}'); echo "start $(date '+%F %T') load1 $l"
awk -v l="$l" 'BEGIN{exit !(l<=4.0)}' || { echo "VOID: load1 $l > 4.0 at start"; exit 3; }
(cd "$HOME/tmcode/goinfer-p29k/backend/metal" && go test -c -tags goinfer_testhooks -o "$OUT/decr.test" . 2>&1 | tail -3)
for rep in 1 2; do
  echo "== run $rep load1 $(sysctl -n vm.loadavg | awk '{print $2}')"
  (cd "$HOME/tmcode/goinfer-p29k/backend/metal" && GOINFER_ZZ_DECR=1 GOMAXPROCS=4 "$OUT/decr.test" -test.run '^TestZZDecodeR$' -test.v 2>&1 | grep -E "^\[zzd\]|FAIL|panic")
done
echo "end $(date '+%F %T') load1 $(sysctl -n vm.loadavg | awk '{print $2}')"
