#!/usr/bin/env bash
# RELEASING.md §C1-M: the Metal device gate (go run ./cmd/gate gpu) for v0.21.0, as a night job on the MacBook.
#
# Why at night: by day its heavy resident-parity tests are refused by the fit guard. Measured 2026-09-30 17:48: about
# 6.1 GB reclaimable with the owner's apps open, so TestDenseResidentParity's CPU reference (qwen2.5-coder-1.5b at
# int8int8, ~4.5 GB with KV and the mapped checkpoint) was not loaded, and the gate reported FAIL without having
# compared a forward (~/goinfer-logs/release-v0.20.0/gate-gpu-metal-cd7eb362.log).
#
# Why a worktree: pinned to REV, and with no vendor/ directory. The main checkout carries a stale, gitignored vendor/
# (aikit v1.46.0 against go.mod's v1.51.0), which fails the gate's CI[root-darwin] build and vet checks; CI has no
# vendor/, and neither does a fresh worktree. The gitignored fixtures the tests read are linked in from the main
# checkout (a directory as a real directory of symlinks, so directory ignore rules still match and the tree stays
# clean), and the worktree gets its own go.work: on this machine `go env GOWORK` is pinned to the main checkout's.
#
# v0.21.0: copied from docs/measurements/release-v0.20.0/run-metal-gate.sh with only the version paths changed. That
# release's green run (8e4fb57c, 2026-10-01 22:54-22:58) took 4 minutes; queued at 15 for the loads and settle.
#
# Pass rule: the gate's own verdict, all 9 declared check groups reporting and zero FAIL (RELEASING.md §C1-M).
set -euo pipefail
REV=${REV:?set REV to the commit to gate}
SRC=$HOME/tmcode/goinfer
BASE=$HOME/goinfer-bench/release-v0.21.0
WT=$BASE/wt-metal
LOG=$HOME/goinfer-logs/release-v0.21.0
mkdir -p "$BASE" "$LOG"

cd "$SRC"
if [ -d "$WT" ]; then git worktree remove --force "$WT"; fi
git worktree prune
git worktree add -q --detach "$WT" "$REV"
link() { # $1: a path relative to the repo root that is gitignored in the main checkout
  local p=${1%/}
  [ -e "$WT/$p" ] && return 0
  if [ -d "$SRC/$p" ]; then
    mkdir -p "$WT/$p"
    for c in "$SRC/$p"/* "$SRC/$p"/.[!.]*; do [ -e "$c" ] && ln -s "$c" "$WT/$p/"; done
  else
    mkdir -p "$(dirname "$WT/$p")"
    ln -s "$SRC/$p" "$WT/$p"
  fi
  n=$((n + 1))
}
n=0
while IFS= read -r p; do link "$p"; done < <(git ls-files --others --ignored --exclude-standard --directory -- testdata decoder/testdata)
link demo/chat/model.gguf

cd "$WT"
go work init . ./gpu ./cuda ./metal ./demo/agent
export GOWORK=$WT/go.work
if [ "${DRYRUN:-0}" = 1 ]; then # the setup alone, for checking the job by day: no gate, no GPU
  echo "linked $n; git status entries: $(git status --porcelain | wc -l | tr -d ' ')"; git status --porcelain | head -3
  go build ./... && go vet -tags goinfer_testhooks ./metal/ && echo "build + vet ok, no vendor/"
  cd "$SRC" && git worktree remove --force "$WT" && echo "worktree removed"
  exit 0
fi
OUT=$LOG/gate-gpu-metal-$(git rev-parse --short HEAD)-night.log
{
  echo "rev:      $(git rev-parse HEAD)"
  echo "started:  $(date '+%F %T %Z')"
  echo "fixtures: $n gitignored entries linked from $SRC"
  echo "tree:     $(git status --porcelain | wc -l | tr -d ' ') entries in git status (0 = clean)"
  echo "go:       $(go version), GOWORK=$GOWORK"
} | tee "$OUT"
go run ./cmd/gate gpu 2>&1 | tee -a "$OUT"
echo "finished: $(date '+%F %T %Z')" | tee -a "$OUT"
