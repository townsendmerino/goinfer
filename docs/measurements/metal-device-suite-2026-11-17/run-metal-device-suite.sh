#!/usr/bin/env bash
# The Metal module's tests with the device present, as a DAY job on the Mac (T1 of the test-health audit). CI's metal-darwin job runs 5 of the module's 882 tests (the
# runner's paravirtual Metal layer SIGSEGVs); the rest ran only in `gate gpu` / `gate quick -full` by hand before a tag. This runs
# `go test -json -count=1 -p 1 -tags goinfer_testhooks ./...` in backend/metal of REV (default origin/main, fetched at start) in its own detached worktree, with the
# fixtures placed, through scripts/cuda_device_suite.py --backend metal, and the whole thing under the timing lock (exclusive: it is a GPU run; waits up to 1 h for a
# timed run to finish). No heavy tier. Measured 4 min 12 s on 2026-10-13 (timed-run skill, gate-tiers table).
#   bash docs/measurements/metal-device-suite-2026-11-17/run-metal-device-suite.sh [logdir]
#   REV=<sha> bash .../run-metal-device-suite.sh            # a branch: files no issue and closes none
#   TESTARGS='-run ^TestSomething$' bash .../run-metal-device-suite.sh   # plumbing control
# Run by the launchd agent com.goinfer.metal-device-suite (README.md); launchd's job inherits no login PATH, so the script sets one.
set -uo pipefail
SRC=$(cd "$(dirname "$0")/../../.." && pwd)
REV=${REV:-origin/main}
LOG=${1:-$HOME/goinfer-logs/metal-device-suite-$(date +%F)}
WT=${WT:-$HOME/goinfer-bench/metal-device-suite-wt}
export PATH=/usr/local/go/bin:/opt/homebrew/bin:/usr/local/bin:$PATH
[ "$(uname -s)" = Darwin ] || { echo "FATAL: not macOS; every Metal test would skip"; exit 2; }
# Re-enter under the timing lock once; the child sees GOINFER_TIMING_LOCK and carries on.
if [ -z "${GOINFER_TIMING_LOCK:-}" ]; then
  exec python3 "$SRC/scripts/timing_lock.py" run --wait 3600 --label metal-device-suite -- bash "$0" "$@"
fi
mkdir -p "$LOG" "$(dirname "$WT")"
cd "$SRC" || exit 2
git fetch -q origin || echo "warning: git fetch failed; testing the local $REV"
[ -d "$WT" ] && git worktree remove --force "$WT"
git worktree add -q --detach "$WT" "$REV" || { echo "FATAL: cannot check out $REV"; exit 2; }
EXC="$LOG/untracked-fixtures.exclude"
fixtures=$(bash "$SRC/scripts/worktree_fixtures.sh" "$SRC" "$WT" "$EXC") || { echo "FATAL: placing the fixtures: $fixtures"; exit 2; }
cd "$WT" || exit 2
export GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=core.excludesFile GIT_CONFIG_VALUE_0="$EXC"
go work init . ./backend/webgpu ./backend/cuda ./backend/metal ./demo/agent || { echo "FATAL: go work init"; exit 2; }
{ echo "rev:      $(git rev-parse HEAD) ($REV)"; echo "started:  $(date '+%F %T %Z')"; echo "fixtures: $fixtures"
  echo "go:       $(go version)"; echo "machine:  $(sysctl -n machdep.cpu.brand_string 2>/dev/null) $(sw_vers -productVersion 2>/dev/null)"; } | tee "$LOG/provenance.txt"
# The runner is the SRC checkout's copy, so a REV from before it existed can still be judged by it.
CGO_ENABLED=0 python3 "$SRC/scripts/cuda_device_suite.py" --backend metal --cwd "$WT/backend/metal" --raw "$LOG/go-test.jsonl" -- \
  go test -json -count=1 -p 1 -timeout 25m -tags goinfer_testhooks ${TESTARGS:-} ./... 2>&1 | tee "$LOG/verdict.log"
rc=${PIPESTATUS[0]}
# Only main's run is the standing verdict and the heartbeat: a run of another REV files no issue and closes none.
if [ "$REV" = "origin/main" ]; then
  python3 "$SRC/scripts/cuda_device_suite_issue.py" --backend metal --box mac --verdict-log "$LOG/verdict.log" --sha "$(git -C "$WT" rev-parse --short=8 HEAD)" 2>&1 | tee -a "$LOG/verdict.log.issue"
else
  echo "REV is $REV, not origin/main: no GitHub issue is filed or closed for this run" | tee -a "$LOG/verdict.log.issue"
fi
echo "finished: $(date '+%F %T %Z') rc=$rc" | tee -a "$LOG/provenance.txt"
cd "$SRC" && git worktree remove --force "$WT" 2>/dev/null
exit $rc
