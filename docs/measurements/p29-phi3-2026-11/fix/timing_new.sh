#!/bin/bash
set -u
OUT="$HOME/goinfer-work/p29k/out"
l=$(sysctl -n vm.loadavg | awk '{print $2}'); echo "start $(date '+%F %T') load1 $l"
awk -v l="$l" 'BEGIN{exit !(l<=4.0)}' || { echo "VOID: load1 $l > 4.0 at start"; exit 3; }
(cd "$HOME/tmcode/goinfer-p29k/backend/metal" && go test -c -tags goinfer_testhooks -o "$OUT/new.test" . 2>&1 | tail -3)
(cd "$HOME/tmcode/goinfer-p29k/backend/metal" && GOINFER_ZZ_ROWS=1 GOINFER_ZZ_X=1 GOMAXPROCS=4 "$OUT/new.test" -test.run '^TestZZStepRowsTiming$' -test.v 2>&1 | grep -E "^\[zz\]|FAIL|panic")
echo "end $(date '+%F %T') load1 $(sysctl -n vm.loadavg | awk '{print $2}')"
