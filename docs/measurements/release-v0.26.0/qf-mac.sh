#!/usr/bin/env bash
# RELEASING.md step 8 on darwin: `go run ./cmd/gate quick -full -base v0.25.0` at REV, in a fresh detached worktree with the
# main checkout's gitignored fixtures placed by scripts/worktree_fixtures.sh, under the Mac's exclusive timing lock.
#   REV=<release candidate commit> bash qf-mac.sh
set -uo pipefail
: "${REV:?set REV to the release candidate commit}"
BASE=v0.25.0
SRC=$HOME/tmcode/goinfer
WT=$HOME/goinfer-bench/relgate-v0260-qf-wt
LOG=$HOME/goinfer-logs/relgate-2026-11-28/quick-full
export PATH=/usr/local/go/bin:$HOME/go/bin:/usr/local/bin:$PATH:/opt/homebrew/bin
python3 -c "import yaml" 2>/dev/null || { echo "FATAL: $(command -v python3) has no PyYAML; the gate derives CI's checks with it"; exit 2; }
mkdir -p "$LOG/records"; rm -f "$LOG/launcher.rc"; cd "$SRC" || exit 2
git cat-file -e "$REV^{commit}" 2>/dev/null || git fetch -q origin || exit 2
git rev-parse -q --verify "$BASE" >/dev/null || { echo "FATAL: no $BASE"; exit 2; }
[ -d "$WT" ] && git worktree remove --force "$WT"
git worktree add -q --detach "$WT" "$REV" || { echo "FATAL: cannot check out $REV"; exit 2; }
EXC="$LOG/untracked-fixtures.exclude"
fixtures=$(bash "$SRC/scripts/worktree_fixtures.sh" "$SRC" "$WT" "$EXC") || { echo "FATAL: placing the fixtures: $fixtures"; exit 2; }
export GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=core.excludesFile GIT_CONFIG_VALUE_0="$EXC"
[ -e "$SRC/demo/chat/model.gguf" ] && { mkdir -p "$WT/demo/chat"; cp -c "$SRC/demo/chat/model.gguf" "$WT/demo/chat/"; }
cd "$WT" || exit 2
bash scripts/make_work.sh >/dev/null 2>&1
# The workspace file's go line (go.work.example) is behind go.mod's toolchain line, and a workspace selects its toolchain from
# its own go line. Pin the run to go.mod's toolchain, the one CI runs (setup-go reads go.mod) and `go install` selects.
TC=$(awk '$1=="toolchain"{print $2}' go.mod); [ -n "$TC" ] || { echo "FATAL: go.mod names no toolchain"; exit 2; }
WORKGO=$(awk '$1=="go"{print $2}' go.work)
export GOTOOLCHAIN=$TC
{ echo "rev:      $(git rev-parse HEAD)"; echo "base:     $BASE"; echo "started:  $(date '+%F %T %Z')"
  echo "go:       $(go version) (GOTOOLCHAIN=$GOTOOLCHAIN, go.mod's toolchain line; the workspace's own go line is $WORKGO)"
  echo "machine:  $(sysctl -n machdep.cpu.brand_string), $(( $(sysctl -n hw.memsize) / 1073741824 )) GB, macOS $(sw_vers -productVersion)"
  echo "fixtures: $fixtures"; echo "load:     $(uptime | sed 's/.*load averages: //')"
  echo "tree:     $(git status --porcelain | wc -l | tr -d ' ') git status entries"; } | tee "$LOG/provenance.txt"
caffeinate -i python3 "$SRC/scripts/timing_lock.py" run --wait 2400 --label "release v0.26.0 quick -full ${REV:0:9}" -- \
  go run ./cmd/gate quick -full -base "$BASE" 2>&1 | tee "$LOG/gate-quick-full.log"
rc=${PIPESTATUS[0]}
if ls docs/measurements/quick-full/darwin-*"${REV:0:8}"*.json >/dev/null 2>&1; then
  cp docs/measurements/quick-full/darwin-*"${REV:0:8}"*.json "$LOG/records/"
  echo "record: $(ls "$LOG/records/" | tr '\n' ' ')"
else
  echo "!! no record was written for ${REV:0:8} (gate exit $rc)"; [ "$rc" -eq 0 ] && rc=1
fi
echo "finished: $(date '+%F %T %Z') rc=$rc" | tee -a "$LOG/provenance.txt"
echo "$rc" > "$LOG/launcher.rc"
exit $rc
