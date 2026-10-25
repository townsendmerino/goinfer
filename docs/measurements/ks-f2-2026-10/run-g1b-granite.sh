#!/usr/bin/env bash
# KS-F2's G1b cell for granitemoehybrid on nobara's CPU (README.md here): gate identity for that family alone, OLD against NEW, with the
# gitignored fixture copied in from ~/goinfer-bench/ks-f2/fixtures. Exit 0 when the family is IDENTICAL, 1 otherwise, 2 when it could not run.
#   bash run-g1b-granite.sh <new> <old>
set -uo pipefail
NEW=${1:?the branch commit}
OLD=${2:?the commit before the change}
R=${SRC:-$HOME/goinfer-runner}
B=$HOME/goinfer-bench/ks-f2
LOG=${LOG:-$HOME/goinfer-logs/ks-f2-2026-10/g1b-granite-$(date +%F-%H%M)}
export PATH=/usr/local/go/bin:$PATH GOWORK=off
mkdir -p "$LOG"
exec > >(tee -a "$LOG/run.log") 2>&1
echo "=== $(date '+%F %T %Z') start: new $NEW, old $OLD; $(cat /proc/loadavg)"
git -C "$R" fetch -q origin || { echo "VOID: fetch failed"; exit 2; }
wt=$B/wt-g1b
[ -d "$wt" ] && git -C "$R" worktree remove --force "$wt"
git -C "$R" worktree add -q --detach "$wt" "$NEW" || { echo "VOID: cannot check out $NEW"; exit 2; }
cp -r "$B/fixtures/granite-tiny" "$wt/testdata/" || { echo "VOID: no granite-tiny in $B/fixtures"; exit 2; }
(cd "$wt" && go run ./cmd/gate identity "$OLD" "$NEW" -backend cpu -assets tiny -steps 64 -families granitemoehybrid > "$LOG/identity.log" 2>&1)
g=$?
sed -n '/== verdict ==/,$p' "$LOG/identity.log"
git -C "$R" worktree remove --force "$wt" 2>/dev/null
if [ $g = 0 ] && grep -q 'IDENTICAL.* granitemoehybrid ' "$LOG/identity.log"; then echo "G1b granitemoehybrid: IDENTICAL"; exit 0; fi
echo "G1b granitemoehybrid: not IDENTICAL (see $LOG/identity.log)"; exit 1
