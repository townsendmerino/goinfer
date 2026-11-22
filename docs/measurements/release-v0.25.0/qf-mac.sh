#!/usr/bin/env bash
# RELEASING.md read 8 on darwin: `gate quick -full -base v0.24.1` at 36dbfbaf1 in a fresh detached worktree with the main
# checkout's gitignored fixtures as APFS clones (run-quick-full.sh's mechanics), under the Mac's exclusive timing lock.
set -uo pipefail
REV=36dbfbaf149aabdff43dce5766b49a65183113a7
SRC=$HOME/tmcode/goinfer
WT=$HOME/goinfer-bench/relgate-qf-wt
LOG=$HOME/goinfer-logs/relgate-2026-11-21/quick-full
command -v go >/dev/null 2>&1 || export PATH=/usr/local/go/bin:/opt/homebrew/bin:$PATH
mkdir -p "$LOG"; cd "$SRC" || exit 2
git cat-file -e "$REV^{commit}" 2>/dev/null || git fetch -q origin || exit 2
git rev-parse -q --verify v0.24.1 >/dev/null || { echo "FATAL: no v0.24.1"; exit 2; }
[ -d "$WT" ] && git worktree remove --force "$WT"; git worktree prune
git worktree add -q --detach "$WT" "$REV" || exit 2
n=0
while IFS= read -r p; do p=${p%/}; [ -e "$WT/$p" ] && continue; mkdir -p "$(dirname "$WT/$p")"; cp -cR "$SRC/$p" "$WT/$p" && n=$((n+1)); done \
  < <(git ls-files --others --ignored --exclude-standard --directory -- testdata decoder/testdata)
[ -e "$SRC/demo/chat/model.gguf" ] && { mkdir -p "$WT/demo/chat"; cp -c "$SRC/demo/chat/model.gguf" "$WT/demo/chat/"; }
cd "$WT" && bash scripts/make_work.sh >/dev/null 2>&1
echo "rev $(git rev-parse --short HEAD); fixtures cloned $n; git status entries: $(git status --porcelain | wc -l | tr -d ' ')"; git status --porcelain | head -3
python3 "$HOME/goinfer-runner/scripts/timing_lock.py" run --wait 1800 --label "relgate quick -full 36dbfbaf1 (coordinator)" -- go run ./cmd/gate quick -full -base v0.24.1
rc=$?
mkdir -p "$LOG/records"; cp docs/measurements/quick-full/*-36dbfbaf*.json "$LOG/records/" 2>/dev/null
echo "rc=$rc at $(date '+%F %T')"
