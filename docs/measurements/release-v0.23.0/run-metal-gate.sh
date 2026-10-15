#!/usr/bin/env bash
# RELEASING.md §C1-M: the Metal device gate (`go run ./cmd/gate gpu`) for v0.23.0, as a night job on the MacBook, at a
# pinned commit. The darwin twin of docs/measurements/gate-gpu-night-2026-10-06/run-gate-gpu.sh, which names CUDA in its
# verdict check and reads nvidia-smi; the worktree mechanics are release-v0.21.0/run-metal-gate.sh's.
#
# Why at night: by day the fit guard refuses the heavy resident-parity loads with the owner's apps open (measured
# 2026-09-30), and the gate wants an idle GPU.
# Why a worktree: the gate runs REV, not whatever the main checkout holds, and its verdict is INCONCLUSIVE on a dirty
# tree. The gitignored fixtures the tests read are placed from the main checkout by scripts/worktree_fixtures.sh (small
# fixtures cloned as real files, model-sized entries linked; its header has the rule), and the worktree gets its own go.work.
#
# The record: -record points at the MAIN checkout's docs/measurements/gpu-gate, where `gate gpu-records` and row 4 of
# `gate release` read it. It is written by the gate from the worktree, so it names REV and REV's trees. Commit it.
#
# Pass rule: the gate's own PASS verdict for metal, and a record for REV whose verdict is PASS. A skip is not a pass.
# Estimate: the 2026-10-05 run took 5 min 12 s (18:35:01 to 18:40:13); the metal and webgpu suites have grown since
# (the gpu module's tests alone took 189 s on 2026-10-11), so about 12 minutes; queue at 25.
#   REV=<sha> python3 scripts/night.py add gate-gpu-metal-v0.23.0 --est 25 --by "<who>, v0.23.0 release prep" --doc RELEASING.md -- env REV=<sha> bash docs/measurements/release-v0.23.0/run-metal-gate.sh
#   REV=<sha> DRYRUN=1 bash docs/measurements/release-v0.23.0/run-metal-gate.sh    # the setup alone, by day: no gate, no GPU
set -uo pipefail
REV=${REV:?set REV to the commit to gate}
SRC=$(cd "$(dirname "$0")/../../.." && pwd)
BASE=$HOME/goinfer-bench/release-v0.23.0
WT=$BASE/wt-metal
LOG=$HOME/goinfer-logs/release-v0.23.0
RECORDS=$SRC/docs/measurements/gpu-gate
command -v go >/dev/null 2>&1 || export PATH=/usr/local/go/bin:/opt/homebrew/bin:$PATH
command -v go >/dev/null 2>&1 || { echo "FATAL: go is not on PATH"; exit 2; }
[ "$(uname -s)" = Darwin ] || { echo "FATAL: this is the Metal gate's script; on Linux use gate-gpu-night-2026-10-06/run-gate-gpu.sh"; exit 2; }
[ -d "$HOME/models" ] || { echo "FATAL: $HOME/models is missing"; exit 2; }
mkdir -p "$BASE" "$LOG"

cd "$SRC" || exit 2
FULL=$(git rev-parse --verify "$REV^{commit}" 2>/dev/null) || { echo "FATAL: $REV is not a commit in $SRC (fetch first)"; exit 2; }
[ -d "$WT" ] && git worktree remove --force "$WT"
git worktree prune
git worktree add -q --detach "$WT" "$FULL" || { echo "FATAL: cannot check out $FULL"; exit 2; }
fixtures=$(bash "$SRC/scripts/worktree_fixtures.sh" "$SRC" "$WT") || { echo "FATAL: placing the fixtures: $fixtures"; exit 2; }
[ -e "$SRC/demo/chat/model.gguf" ] && [ ! -e "$WT/demo/chat/model.gguf" ] && mkdir -p "$WT/demo/chat" && ln -s "$SRC/demo/chat/model.gguf" "$WT/demo/chat/model.gguf"
# An untracked fixture directory that is NOT ignored cannot be linked without making the worktree dirty; say so instead.
stray=$(git ls-files --others --exclude-standard --directory -- testdata decoder/testdata | tr '\n' ' ')

cd "$WT" || exit 2
go work init . ./gpu ./cuda ./metal ./demo/agent || { echo "FATAL: go work init"; exit 2; }
export GOWORK=$WT/go.work
finish() { cd "$SRC" && git worktree remove --force "$WT" 2>/dev/null; }
if [ "${DRYRUN:-0}" = 1 ]; then
  echo "rev $FULL; $fixtures; git status entries in the worktree: $(git status --porcelain | wc -l | tr -d ' ')"; git status --porcelain | head -3
  go build ./... && go vet -tags goinfer_testhooks ./metal/ && go build -o /dev/null ./cmd/gate && echo "build + vet ok"
  rc=$?
  finish && echo "worktree removed"
  exit $rc
fi
OUT=$LOG/gate-gpu-metal-${FULL:0:8}-night.log
{
  echo "rev:      $FULL"
  echo "started:  $(date '+%F %T %Z')"
  echo "fixtures: $fixtures${stray:+; untracked and not placed: $stray}"
  echo "tree:     $(git status --porcelain | wc -l | tr -d ' ') entries in git status (0 = clean)"
  echo "go:       $(go version), GOWORK=$GOWORK"
  echo "machine:  $(sw_vers -productVersion) $(sysctl -n machdep.cpu.brand_string); $(pmset -g batt | head -1)"
  echo "load:     $(uptime)"
} | tee "$OUT"
go run ./cmd/gate gpu -logdir "$LOG" -record "$RECORDS" 2>&1 | tee -a "$OUT"
rc=${PIPESTATUS[0]}
clean=$(sed $'s/\x1b\\[[0-9;]*m//g' "$OUT")
if ! grep -q -- "PASS — metal on " <<<"$clean"; then echo "!! the gate printed no PASS verdict for metal (rc=$rc); see $OUT" | tee -a "$OUT"; rc=1; fi
rec=$(ls "$RECORDS"/metal-*-"${FULL:0:8}".json 2>/dev/null | tail -1)
if [ -z "$rec" ]; then
  echo "!! the gate wrote no record for ${FULL:0:8} under $RECORDS" | tee -a "$OUT"; rc=1
elif ! grep -q '"verdict": "PASS"' "$rec"; then
  echo "!! the record's verdict is not PASS: $rec" | tee -a "$OUT"; rc=1
fi
echo "record:   ${rec:-none} (untracked in the main checkout until it is committed)" | tee -a "$OUT"
echo "finished: $(date '+%F %T %Z') rc=$rc" | tee -a "$OUT"
finish
exit $rc
