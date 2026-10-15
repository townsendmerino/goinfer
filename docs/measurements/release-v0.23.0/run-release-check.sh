#!/usr/bin/env bash
# The Mac's two v0.23.0 release checks as night jobs, pinned to the release commit: run-release-check.sh metal-gate |
# quick-full. Queued before the release commit is known, so the commit is read at run time from
# ~/goinfer-bench/release-v0.23.0/REV (one SHA, written when nobara sends it). With no REV file the job stops at once
# and says so, so the night goes on to the next job.
#
# metal-gate runs the main checkout's run-metal-gate.sh; quick-full runs run-quick-full.sh beside this script. Both
# write their records into the main checkout, untracked: commit them.
#   python3 scripts/night.py add release-metal-gate --est 25 --priority 10 ... -- bash ~/goinfer-bench/release-v0.23.0/run-release-check.sh metal-gate
#   python3 scripts/night.py add release-quick-full --est 20 --priority 20 ... -- bash ~/goinfer-bench/release-v0.23.0/run-release-check.sh quick-full
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
SRC=${SRC:-$HOME/tmcode/goinfer}
REVF=$HOME/goinfer-bench/release-v0.23.0/REV
[ -s "$REVF" ] || { echo "FATAL: no release commit in $REVF (nobara sends it); nothing run"; exit 2; }
REV=$(tr -d '[:space:]' <"$REVF")
cd "$SRC" || exit 2
git fetch -q --tags origin || echo "warning: git fetch failed; using what $SRC already has"
echo "release check $1 at $REV ($(date '+%F %T %Z'))"
case "${1:-}" in
metal-gate) env REV="$REV" bash "$SRC/docs/measurements/release-v0.23.0/run-metal-gate.sh" ;;
quick-full) env REV="$REV" SRC="$SRC" bash "$HERE/run-quick-full.sh" ;;
*) echo "usage: run-release-check.sh metal-gate|quick-full"; exit 2 ;;
esac
