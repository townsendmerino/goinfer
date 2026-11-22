#!/usr/bin/env bash
# Evidence for the v0.25.0 accepted reds (coordinator, 2026-11-22): quick -full's decoder cell at 36dbfbaf1 hit its 45m budget
# (run 1: 823 pass, 0 fail when cut). The same cell's command, -count=1, at a 120m budget, under nobara's exclusive lock.
set -uo pipefail
export PATH=/usr/local/go/bin:$HOME/go/bin:$PATH
D=$HOME/goinfer-logs/relgate-2026-11-21/quick-full
R=/tmp/gate-quick-20261121-213833-3275687820
cp "$R/test_decoder.log" "$D/test_decoder-budget.log"
cmd=$(python3 "$D/../cellcmd.py" "$R/test_decoder.log" 120m) || exit 2
cmd="go test -count=1 ${cmd#go test }"
dir=$(sed -n 2p "$R/test_decoder.log" | sed 's/^# (in \(.*\))$/\1/')
echo "== test_decoder start $(date '+%F %T') in $dir"; echo "$cmd" > "$D/test_decoder-120m.cmd"
cd "$dir" || exit 2
python3 "$HOME/goinfer-bench/night-src/scripts/timing_lock.py" run --wait 21600 --label "v0.25.0 evidence: quick decoder cell at 120m (coordinator)" -- bash -c "$cmd" > "$D/test_decoder-120m.json" 2>&1
rc=$?
python3 - "$D/test_decoder-120m.json" <<'P'
import json, sys
c = {}; failed = []
for l in open(sys.argv[1]):
    try: e = json.loads(l)
    except Exception: continue
    if e.get("Test") and "/" not in e["Test"] and e.get("Action") in ("pass", "fail", "skip"):
        c[e["Action"]] = c.get(e["Action"], 0) + 1
        if e["Action"] == "fail": failed.append(e["Test"])
print(c, "failed:", failed)
P
echo "rc=$rc end $(date '+%F %T')"; echo "ALL DONE"
