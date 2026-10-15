#!/usr/bin/env bash
# `go run ./cmd/gate quick -full -base <the previous tag>` on nobara-pc: the record `gate release` row 12 wants from this box
# (RELEASING.md step 8), as an unattended night job. A record counts only from a CLEAN tree at the release code, so it runs in its own
# detached worktree of REV (default origin/main), not in the checkout somebody is editing. Streams to a log, has its own timeout,
# and copies the record it wrote to the main checkout (docs/measurements/quick-full/, untracked: commit it) and to the log directory.
#
#   python3 scripts/night.py add quick-full-linux --est 40 --by "<who>, v0.23.0 pre-flight" --doc RELEASING.md -- bash docs/measurements/integration-harness-2026-10/run-quick-full-night.sh
#   REV=<sha> BASE=v0.22.0 bash docs/measurements/integration-harness-2026-10/run-quick-full-night.sh [logdir]
set -uo pipefail
SRC=$(cd "$(dirname "$0")/../../.." && pwd)
REV=${REV:-origin/main}
BASE=${BASE:-$(cd "$SRC" && git tag --list 'v[0-9]*' --sort=-version:refname | head -1)}
LOG=${1:-$HOME/goinfer-logs/quick-full-$(date +%F)}
WT=${WT:-$HOME/goinfer-bench/quick-full-wt}
TIMEOUT=${TIMEOUT:-90m}
export PATH=/usr/local/go/bin:$PATH
mkdir -p "$LOG" "$(dirname "$WT")"
[ -n "$BASE" ] || { echo "FATAL: no previous tag found"; exit 2; }
cd "$SRC" || exit 2
git fetch -q origin || echo "warning: git fetch failed; using the local $REV"
[ -d "$WT" ] && git worktree remove --force "$WT"
git worktree add -q --detach "$WT" "$REV" || { echo "FATAL: cannot check out $REV"; exit 2; }
# -full exists to RUN the deferred tests, and a fresh worktree has none of the gitignored fixtures this box has. scripts/worktree_fixtures.sh
# places them (its header has the rule): small fixtures COPIED as real files, since tests that copy a fixture directory keep only regular files
# and drop a symlinked one; larger ones cloned, symlinks into ~/models kept as links. Untracked, not-ignored fixture directories are placed too and
# listed in $EXC: they would make the gate record the tree as dirty, and a dirty record does not count, so this process alone ignores them.
EXC="$LOG/untracked-fixtures.exclude"
fixtures=$(bash "$SRC/scripts/worktree_fixtures.sh" "$SRC" "$WT" "$EXC") || { echo "FATAL: placing the fixtures: $fixtures"; exit 2; }
export GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=core.excludesFile GIT_CONFIG_VALUE_0="$EXC"
cd "$WT" || exit 2
{ echo "rev:      $(git rev-parse HEAD) ($REV)"; echo "base:     $BASE"; echo "started:  $(date '+%F %T %Z')"; echo "go:       $(go version)"
  echo "fixtures: $fixtures"
  echo "driver:   $(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>&1 | head -1)"; echo "load:     $(uptime)"; } | tee "$LOG/provenance.txt"
# A fresh worktree has no go.work and no module cache entry for what a bump pulled in: fetch it before the gate, as CI does.
go mod download 2>&1 | tail -3
timeout "$TIMEOUT" go run ./cmd/gate quick -full -base "$BASE" 2>&1 | tee "$LOG/gate-quick-full.log"
rc=${PIPESTATUS[0]}
rec=$(ls -t docs/measurements/quick-full/linux-*.json 2>/dev/null | head -1)
if [ -n "$rec" ]; then
  cp "$rec" "$LOG/" && mkdir -p "$SRC/docs/measurements/quick-full" && cp -n "$rec" "$SRC/docs/measurements/quick-full/" && echo "record: $rec (copied to $SRC/docs/measurements/quick-full/ and $LOG)"
else
  echo "!! no record was written (gate exit $rc)"; [ "$rc" -eq 0 ] && rc=1
fi
echo "finished: $(date '+%F %T %Z') rc=$rc" | tee -a "$LOG/provenance.txt"
cd "$SRC" && git worktree remove --force "$WT" 2>/dev/null
exit $rc
