#!/usr/bin/env bash
# RELEASING.md read 8: the deferred tests on linux, `gate quick -full -base <previous tag>` at 36dbfbaf1, in the parity sweep's clean worktree (fixtures placed), under nobara's exclusive lock.
set -uo pipefail
D=$HOME/goinfer-logs/relgate-2026-11-21/quick-full
WT=$HOME/goinfer-bench/relgate-parity/wt
NS=$HOME/goinfer-bench/night-src
export PATH=/usr/local/go/bin:$HOME/go/bin:$PATH   # cgo stays on: backend/webgpu needs it (run 1 had CGO_ENABLED=0 and is void)
mkdir -p "$D"; cd "$WT" || exit 2
[ "$(git rev-parse HEAD)" = 36dbfbaf149aabdff43dce5766b49a65183113a7 ] || { echo "FATAL: not 36dbfbaf1"; exit 2; }
git status --porcelain | head -3
exec python3 "$NS/scripts/timing_lock.py" run --wait 600 --label "relgate quick -full 36dbfbaf1 (coordinator)" -- go run ./cmd/gate quick -full -base "$1"
