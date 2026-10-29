#!/usr/bin/env bash
# The CUDA `gate gpu` recorded for SB12 step 10 (docs/tasks/task-subsystems-2026-10.md, "SB12 step 10 (CUDA)"), shaped like
# docs/measurements/gate-gpu-night-2026-10-06/run-gate-gpu.sh and the Mac's run-gate-gpu-metal.sh: REV (default origin/main) in
# its own detached worktree, the gitignored fixtures placed from FIXTURES (the main checkout) by scripts/worktree_fixtures.sh, a
# go.work, -record and -logdir outside the worktree, under the timing lock held shared (a gate only adds load). The heavy tier
# is skipped unless HEAVY=1: it is a later step. A sampler lists, every 20 s, every CUDA process on the card that is not this
# run's (told apart by working directory), which voids the run under C5 and C6.
#   REV=<sha> bash run-gate-gpu-cuda.sh LOGDIR
set -uo pipefail
REPO=$(cd "$(dirname "$0")/../../.." && pwd)
FIXTURES=${FIXTURES:-$HOME/mycode/goinfer}
REV=${REV:-origin/main}
LOG=${1:?usage: run-gate-gpu-cuda.sh LOGDIR}
RECORD=${RECORD:-$LOG/records}
WT=${WT:-$HOME/goinfer-bench/gate-gpu-cuda-sb12-wt}
export PATH=/usr/local/go/bin:$PATH
[ "${HEAVY:-}" = 1 ] || export GOINFER_GATE_SKIP_HEAVY=1
mkdir -p "$LOG"
cd "$REPO" || exit 2
git fetch -q origin || echo "warning: git fetch failed; testing the local $REV"
[ -d "$WT" ] && git worktree remove --force "$WT"
git worktree add -q --detach "$WT" "$REV" || { echo "FATAL: cannot check out $REV"; exit 2; }
EXC="$LOG/untracked-fixtures.exclude"
fixtures=$(bash "$REPO/scripts/worktree_fixtures.sh" "$FIXTURES" "$WT" "$EXC") || { echo "FATAL: placing the fixtures: $fixtures"; exit 2; }
cd "$WT" || exit 2
export GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=core.excludesFile GIT_CONFIG_VALUE_0="$EXC"
go work init . ./gpu ./cuda ./metal ./demo/agent || { echo "FATAL: go work init"; exit 2; }
{ echo "rev:      $(git rev-parse HEAD) ($REV)"; echo "started:  $(date '+%F %T %Z')"; echo "fixtures: $fixtures"
  echo "go:       $(go version)"; echo "gpu:      $(nvidia-smi --query-gpu=name,driver_version,memory.used --format=csv,noheader)"
  echo "load:     $(cut -d' ' -f1-3 /proc/loadavg)"
  echo "heavy:    $([ -n "${GOINFER_GATE_SKIP_HEAVY:-}" ] && echo SKIPPED || echo on)"; } | tee "$LOG/provenance.txt"
( while :; do
    nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader | grep -v -E "kwin_wayland|Xwayland|plasmashell" |
      while IFS=', ' read -r pid name; do
        cwd=$(readlink /proc/$pid/cwd 2>/dev/null)
        case "$cwd" in "$WT"|"$WT"/*) ;; *) echo "$(date '+%T') $pid $name cwd=${cwd:-?}" ;; esac
      done
    sleep 20
  done ) > "$LOG/other-cuda-processes.txt" 2>&1 &
sampler=$!
python3 "$REPO/scripts/timing_lock.py" run --shared --wait 1800 --label "nobara-ci01 SB12 gate gpu cuda" -- \
  go run ./cmd/gate gpu -logdir "$LOG" -record "$RECORD" 2>&1 | tee "$LOG/gate-gpu.log"
rc=${PIPESTATUS[0]}
kill $sampler 2>/dev/null
echo "other CUDA processes seen: $(wc -l < "$LOG/other-cuda-processes.txt")" | tee -a "$LOG/provenance.txt"
echo "record(s) under $RECORD: $(ls -t "$RECORD"/*.json 2>/dev/null | head -2 | tr '\n' ' ')" | tee -a "$LOG/provenance.txt"
echo "finished: $(date '+%F %T %Z') rc=$rc" | tee -a "$LOG/provenance.txt"
cd "$REPO" && git worktree remove --force "$WT" 2>/dev/null
exit $rc
