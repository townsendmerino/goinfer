#!/usr/bin/env bash
# RELEASING.md step 8: the tests the day loop defers (`go run ./cmd/gate quick -full -base v0.22.0`) for v0.23.0, as a
# night job on the MacBook, at a pinned commit. The worktree mechanics are run-metal-gate.sh's (this directory): a
# detached worktree at REV with its own go.work, and the gitignored fixtures from the main checkout. Here the fixtures
# are APFS clones (`cp -c`: real files sharing the originals' blocks), not symlinks: two tests copy a fixture directory
# (internal/modelload TestLoad_safetensorsDirGoesThroughSidecar, internal/prequant TestDirSidecar_keepsMRopeSection), and
# a copy of a directory of symlinks lost its files, so both failed under links on 2026-10-14 and pass under clones.
#
# The record: `gate quick -full` writes docs/measurements/quick-full/<os>-<arch>-<time>-<commit>.json into the tree it
# runs in, which here is the worktree. This script copies it into the main checkout ($SRC), where row 12 of
# `gate release` reads it, before the worktree is removed. Commit it.
#
# Pass rule: the gate's exit 0, and a record for REV that is a PASS on a clean tree. A skip is not a pass.
# Estimate: about 10 minutes on the Mac (RELEASING.md); queue at 20.
#   REV=<sha> bash run-quick-full.sh             # SRC defaults to ~/tmcode/goinfer, the main checkout
#   REV=<sha> DRYRUN=1 bash run-quick-full.sh    # the setup alone, by day
set -uo pipefail
REV=${REV:?set REV to the commit to gate}
SRC=${SRC:-$HOME/tmcode/goinfer}
BASE=$HOME/goinfer-bench/release-v0.23.0
WT=$BASE/wt-quick
LOG=$HOME/goinfer-logs/release-v0.23.0
command -v go >/dev/null 2>&1 || export PATH=/usr/local/go/bin:/opt/homebrew/bin:$PATH
command -v go >/dev/null 2>&1 || { echo "FATAL: go is not on PATH"; exit 2; }
mkdir -p "$BASE" "$LOG"

cd "$SRC" || exit 2
FULL=$(git rev-parse --verify "$REV^{commit}" 2>/dev/null) || { echo "FATAL: $REV is not a commit in $SRC (fetch first)"; exit 2; }
git rev-parse --verify -q v0.22.0 >/dev/null || { echo "FATAL: the base tag v0.22.0 is not in $SRC"; exit 2; }
[ -d "$WT" ] && git worktree remove --force "$WT"
git worktree prune
git worktree add -q --detach "$WT" "$FULL" || { echo "FATAL: cannot check out $FULL"; exit 2; }
n=0
link() { # $1: a path relative to the repo root that is gitignored in the main checkout; cloned, not linked (above)
  local p=${1%/}
  [ -e "$WT/$p" ] && return 0
  mkdir -p "$(dirname "$WT/$p")"
  cp -cR "$SRC/$p" "$WT/$p" || return 1
  n=$((n + 1))
}
while IFS= read -r p; do link "$p"; done < <(git ls-files --others --ignored --exclude-standard --directory -- testdata decoder/testdata)
[ -e "$SRC/demo/chat/model.gguf" ] && link demo/chat/model.gguf
stray=$(git ls-files --others --exclude-standard --directory -- testdata decoder/testdata | tr '\n' ' ')

cd "$WT" || exit 2
go work init . ./gpu ./metal ./demo/agent || { echo "FATAL: go work init"; exit 2; }
export GOWORK=$WT/go.work
finish() { cd "$SRC" && git worktree remove --force "$WT" 2>/dev/null; }
if [ "${DRYRUN:-0}" = 1 ]; then
  echo "rev $FULL; linked $n; git status entries in the worktree: $(git status --porcelain | wc -l | tr -d ' ')"; git status --porcelain | head -3
  go build ./... && go build -o /dev/null ./cmd/gate && go run ./cmd/gate quick -full -base v0.22.0 -select >/dev/null && echo "build + selection ok"
  rc=$?
  finish && echo "worktree removed"
  exit $rc
fi
OUT=$LOG/quick-full-darwin-${FULL:0:8}-night.log
{
  echo "rev:      $FULL"
  echo "started:  $(date '+%F %T %Z')"
  echo "fixtures: $n gitignored entries cloned from $SRC${stray:+; untracked and not linked: $stray}"
  echo "tree:     $(git status --porcelain | wc -l | tr -d ' ') entries in git status (0 = clean)"
  echo "go:       $(go version), GOWORK=$GOWORK"
  echo "machine:  $(sw_vers -productVersion) $(sysctl -n machdep.cpu.brand_string); $(pmset -g batt | head -1)"
  echo "load:     $(uptime)"
} | tee "$OUT"
go run ./cmd/gate quick -full -base v0.22.0 -logdir "$LOG" 2>&1 | tee -a "$OUT"
rc=${PIPESTATUS[0]}
rec=$(ls docs/measurements/quick-full/darwin-*-"${FULL:0:8}".json 2>/dev/null | tail -1)
if [ -z "$rec" ]; then
  echo "!! the gate wrote no record for ${FULL:0:8}" | tee -a "$OUT"; rc=1
else
  mkdir -p "$SRC/docs/measurements/quick-full"
  cp "$rec" "$SRC/docs/measurements/quick-full/" && rec=$SRC/docs/measurements/quick-full/$(basename "$rec")
  grep -q '"verdict": "PASS"' "$rec" || { echo "!! the record's verdict is not PASS: $rec" | tee -a "$OUT"; rc=1; }
  grep -q '"dirty": false' "$rec" || { echo "!! the record was made on a dirty tree: $rec" | tee -a "$OUT"; rc=1; }
fi
echo "record:   ${rec:-none} (untracked in the main checkout until it is committed)" | tee -a "$OUT"
echo "finished: $(date '+%F %T %Z') rc=$rc" | tee -a "$OUT"
finish
exit $rc
