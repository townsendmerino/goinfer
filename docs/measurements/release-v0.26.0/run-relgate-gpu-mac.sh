#!/usr/bin/env bash
# v0.26.0 release row 4 on the Mac: `go run ./cmd/gate gpu` (Metal) at REV, through the committed runner
# (docs/measurements/gate-gpu-night-2026-10-06/run-gate-gpu.sh), under the exclusive timing lock.
set -uo pipefail
export PATH=/usr/local/go/bin:$HOME/go/bin:/usr/local/bin:$PATH:/opt/homebrew/bin
python3 -c "import yaml" 2>/dev/null || { echo "FATAL: $(command -v python3) has no PyYAML; the gate derives CI's checks with it"; exit 2; }
: "${REV:?set REV to the release candidate commit}"
SRC=$HOME/tmcode/goinfer
LOG=$HOME/goinfer-logs/relgate-2026-11-28/gate-gpu
mkdir -p "$LOG"
cd "$SRC" || exit 2
rm -f "$LOG/launcher.rc"
echo "launcher start $(date '+%F %T %Z') rev $REV load: $(uptime | sed 's/.*load averages: //')"
export REV WT=$HOME/goinfer-bench/relgate-v0260-gpu-wt
caffeinate -i python3 scripts/timing_lock.py run --wait 2400 --label "release v0.26.0 gate gpu metal ${REV:0:9}" -- \
  bash docs/measurements/gate-gpu-night-2026-10-06/run-gate-gpu.sh "$LOG"
rc=$?
echo "launcher end $(date '+%F %T %Z') rc=$rc"
echo "$rc" > "$LOG/launcher.rc"
exit $rc
