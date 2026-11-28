#!/usr/bin/env bash
# Launches P33's first measurement detached (launchd), inside the exclusive timing lock, from a clean worktree at the
# registration's commit. The runner and what it measures: docs/measurements/p33-metal-decode-split-2026-11/.
set -uo pipefail
export PATH=/usr/local/go/bin:$HOME/go/bin:/usr/local/bin:$PATH:/opt/homebrew/bin
WT=$HOME/goinfer-bench/p33-wt
OUT=$HOME/goinfer-logs/p33-metal-decode-split-2026-11-28
cd "$WT" || exit 2
GOTOOLCHAIN=$(awk '$1=="toolchain"{print $2}' go.mod); export GOTOOLCHAIN
BIN=$HOME/goinfer-work/eval-followup/p33/metal-tagged-8387e5723.test OUT=$OUT NEED_MB=370 \
  caffeinate -i python3 scripts/timing_lock.py run --wait 1800 --label p33-split -- \
  bash docs/measurements/p33-metal-decode-split-2026-11/run-p33-split.sh
rc=$?
echo "$rc" > "$OUT/launcher.rc"
exit $rc
