#!/usr/bin/env bash
# The Metal half of the GPU gate (RELEASING.md §C1-M) recorded for SB12 (docs/measurements/sb12-2026-10.md, "The Mac's Metal
# `gate gpu` run"), shaped like docs/measurements/gate-gpu-night-2026-10-06/run-gate-gpu.sh: REV (default origin/main) in its
# own detached worktree, the gitignored fixtures placed from FIXTURES (the main checkout) by scripts/worktree_fixtures.sh, a
# go.work, -record and -logdir outside the worktree, under the timing lock held shared (a gate only adds load). A sampler
# lists, every 30 s, every process running Metal or WebGPU tests, since two at once can crash one spuriously.
#   bash run-gate-gpu-metal.sh [LOGDIR]
set -uo pipefail
REPO=$(cd "$(dirname "$0")/../../.." && pwd)
FIXTURES=${FIXTURES:-$HOME/tmcode/goinfer}
REV=${REV:-origin/main}
LOG=${1:-$HOME/goinfer-logs/gate-gpu-metal-$(date +%F-%H%M)}
RECORD=${RECORD:-$LOG/records}
WT=${WT:-$HOME/goinfer-bench/gate-gpu-metal-wt}
mkdir -p "$LOG"
cd "$REPO" || exit 2
git fetch -q origin || echo "warning: git fetch failed; testing the local $REV"
[ -d "$WT" ] && git worktree remove --force "$WT"
git worktree add -q --detach "$WT" "$REV" || { echo "FATAL: cannot check out $REV"; exit 2; }
EXC="$LOG/untracked-fixtures.exclude"
fixtures=$(bash "$REPO/scripts/worktree_fixtures.sh" "$FIXTURES" "$WT" "$EXC") || { echo "FATAL: placing the fixtures: $fixtures"; exit 2; }
cd "$WT" || exit 2
export GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=core.excludesFile GIT_CONFIG_VALUE_0="$EXC"
go work init . ./gpu ./metal ./demo/agent || { echo "FATAL: go work init"; exit 2; }
{ echo "rev:      $(git rev-parse HEAD) ($REV)"; echo "started:  $(date '+%F %T %Z')"; echo "fixtures: $fixtures"
  echo "go:       $(go version)"; echo "host:     $(sysctl -n hw.model) $(sw_vers -productVersion) $(sysctl -n machdep.cpu.brand_string)"
  echo "power:    $(pmset -g batt | head -1)"; echo "load:     $(sysctl -n vm.loadavg)"
  echo "heavy:    $([ -n "${GOINFER_GATE_SKIP_HEAVY:-}" ] && echo SKIPPED || echo on)"; } | tee "$LOG/provenance.txt"
( while :; do o=$(ps -Ao pid,etime,command | grep -E '[m]etal\.test|[g]pu\.test|[k]ernels\.test|[g]o test .*(metal|gpu)' | grep -v "$WT" | cut -c1-200)
    [ -n "$o" ] && { echo "$(date '+%T')"; echo "$o"; }; sleep 30; done ) > "$LOG/other-metal-tests.txt" 2>&1 &
sampler=$!
python3 "$REPO/scripts/timing_lock.py" run --shared --wait 1800 --label "mac-sb11m SB12 gate gpu metal" -- \
  go run ./cmd/gate gpu -logdir "$LOG" -record "$RECORD" 2>&1 | tee "$LOG/gate-gpu.log"
rc=${PIPESTATUS[0]}
kill $sampler 2>/dev/null
echo "record(s) under $RECORD: $(ls -t "$RECORD"/*.json 2>/dev/null | head -2 | tr '\n' ' ')" | tee -a "$LOG/provenance.txt"
echo "finished: $(date '+%F %T %Z') rc=$rc" | tee -a "$LOG/provenance.txt"
cd "$REPO" && git worktree remove --force "$WT" 2>/dev/null
exit $rc
