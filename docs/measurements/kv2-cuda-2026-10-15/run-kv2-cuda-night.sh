#!/bin/bash
# KV2's CUDA night jobs on nobara (docs/measurements/kv2-cuda-2026-10-15.md §4). Runs with nobody watching: REV and BASE are
# commits named when it is queued, and everything is built here from detached worktrees of them. night.py holds the timing
# lock; children inherit it.
#
#   JOB=share  G5-C: the Mac's run-kv2-night.sh JOB=share (kv2_share.py on the 1.5B, graded), read from REV.
#   JOB=speed  G6-C: the Mac's run-kv2-night.sh JOB=speed (KV1's four CUDA cells, graded).
#   JOB=heavy  G1-C's real-checkpoint tests, G7-C's in-process copy cost on the 1.5B and the 7B, GV-C(a) on the 26B with
#              expert caching.
#   Every job records swap in use before and after (GV-C(c): growth of 64 MB or less passes) and, for share, the copy and
#   decline lines of every server log, which carry the slots' windowed bases and the rollback budget.
#
#   python3 scripts/night.py add kv2-<job>-cuda --est <min> --by "<who>, KV2" --doc docs/tasks/task-kv-architecture-2026-10.md \
#     -- env JOB=<job> REV=<sha> BASE=<sha> bash docs/measurements/kv2-cuda-2026-10-15/run-kv2-cuda-night.sh
#   DRY=1 ...   stops after the builds and the provenance block.
#
# BASE is main as merged into REV (the registration's §1, choice 8), not the Mac's 1cc385ef.
# Estimates: share 40 minutes (9 cells a block of about 30 s, 6 blocks, one cold 17,609-token prefill measured at 9.3 s by
# day), speed 45 (KV1 measured 31), heavy 45 (G1-C's real-checkpoint tests are most of it; the 26B loads twice).
set -uo pipefail
JOB=${JOB:?JOB is share, speed or heavy}
REV=${REV:?REV is required: the commit under test}
BASE=${BASE:?BASE is required: main as merged into REV}
SRC=$HOME/mycode/goinfer
B=${B:-$HOME/goinfer-bench/kv2-cuda-night}
OUT=${OUT:-$HOME/goinfer-logs/kv2-cuda-2026-10-15/night-$(date +%F)-$JOB}
command -v go >/dev/null 2>&1 || export PATH=/usr/local/go/bin:$PATH
mkdir -p "$B/bin" "$OUT" || exit 2
cd "$SRC" || exit 2
git fetch -q origin || echo "warning: git fetch failed; using what is here"
git rev-parse --verify -q "$REV^{commit}" >/dev/null || { echo "FATAL: $REV is not a commit here"; exit 2; }
git rev-parse --verify -q "$BASE^{commit}" >/dev/null || { echo "FATAL: $BASE is not a commit here"; exit 2; }
swap_mb() { awk '/^SwapTotal/{t=$2} /^SwapFree/{f=$2} END{print int((t-f)/1024)}' /proc/meminfo; }
{
  echo "job: $JOB (kv2-cuda)"; echo "started: $(date '+%F %T %Z')"; echo "host: $(uname -srm)"
  echo "new (REV): $(git rev-parse "$REV")"; echo "old (BASE, main as merged into REV): $(git rev-parse "$BASE")"
  echo "driver: $(nvidia-smi --query-gpu=driver_version,name --format=csv,noheader)"
  echo "aikit (go.mod, new / old): $(git show "$REV:go.mod" | awk '$1=="github.com/townsendmerino/aikit"{print $2}') / $(git show "$BASE:go.mod" | awk '$1=="github.com/townsendmerino/aikit"{print $2}')"
  echo "load: $(uptime)"
  echo "VRAM used before: $(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader)"
} | tee "$OUT/provenance.txt"
S0=$(swap_mb)
echo "swap in use before: $S0 MB" | tee "$OUT/never-swap.txt"
worktree() { # worktree <dir> <commit>: a detached worktree with this box's go.work
  [ -d "$1" ] && git worktree remove --force "$1"
  git worktree prune
  git worktree add -q --detach "$1" "$2" || { echo "FATAL: cannot check out $2"; exit 2; }
  # aikit is NOT in the workspace: it builds at the version go.mod pins, as a release does, not at a local checkout.
  printf 'go 1.27.0\n\ntoolchain go1.27.2\n\nuse (\n\t.\n\t./cuda\n\t./gpu\n)\n' > "$1/go.work"
}
rc=0
case "$JOB" in
share|speed)
  git show "$REV:docs/measurements/kv2-2026-10-14/run-kv2-night.sh" > "$OUT/run-kv2-night.sh" || exit 2
  if [ "${DRY:-0}" = 1 ]; then
    env JOB="$JOB" REV="$REV" BASE="$BASE" OUT="$OUT" DRY=1 bash "$OUT/run-kv2-night.sh"; exit $?
  fi
  env JOB="$JOB" REV="$REV" BASE="$BASE" OUT="$OUT" bash "$OUT/run-kv2-night.sh" || rc=1
  if [ -d "$OUT/serve-logs" ]; then
    grep -h -E "^(cuda: kv copy|kv share:)|^==== " "$OUT/serve-logs"/* > "$OUT/kv-copy-lines.txt" 2>/dev/null
    echo "copy and decline lines: $(grep -c -E '^(cuda: kv copy|kv share:)' "$OUT/kv-copy-lines.txt") in $OUT/kv-copy-lines.txt"
  fi
  ;;
heavy)
  WT=$B/wt-heavy; WB=$B/wt-base
  worktree "$WT" "$REV"; worktree "$WB" "$BASE"
  bash "$WT/scripts/worktree_fixtures.sh" "$SRC" "$WT" || { echo "FATAL: fixtures"; exit 2; }
  (cd "$WT/cuda" && CGO_ENABLED=0 go build -tags cuda -o "$B/bin/serve-rev" ./cmd/serve) || { echo "FATAL: the REV build failed"; exit 2; }
  (cd "$WB/cuda" && CGO_ENABLED=0 go build -tags cuda -o "$B/bin/serve-base" ./cmd/serve) || { echo "FATAL: the BASE build failed"; exit 2; }
  [ "${DRY:-0}" = 1 ] && { echo "DRY=1: stopping after the builds"; exit 0; }
  export GOWORK=$WT/go.work
  cd "$WT/cuda" || exit 2
  echo "== G1-C heavy (the CUDA identity patterns with GOINFER_HEAVY_TESTS=1): started $(date '+%T')"
  env GOINFER_HEAVY_TESTS=1 go test -count=1 -timeout 60m -tags 'cuda goinfer_testhooks' \
    -run 'Reuse|Slot|MC3|MC4|Lead|Resident|^TestWindowedKV_|^TestKV2_|^TestKV0_prefillInvariance_' -v . > "$OUT/heavy-g1-cuda.log" 2>&1 || rc=1
  grep -E '^(ok|FAIL)|^--- FAIL' "$OUT/heavy-g1-cuda.log"
  echo "PASS $(grep -c '^--- PASS' "$OUT/heavy-g1-cuda.log") SKIP $(grep -c '^--- SKIP' "$OUT/heavy-g1-cuda.log") FAIL $(grep -c '^--- FAIL' "$OUT/heavy-g1-cuda.log")"
  echo "== G7-C in process, the 1.5B and the 7B: started $(date '+%T')"
  env GOINFER_HEAVY_TESTS=1 GOINFER_KV2_COPYCOST_7B=1 go test -count=1 -timeout 20m -tags 'cuda goinfer_testhooks' \
    -run '^TestKV2_copyCost_real$' -v . > "$OUT/g7-copycost.log" 2>&1 || rc=1
  grep -E '^(ok|FAIL)|^--- |kv2-g7' "$OUT/g7-copycost.log"
  echo "== GV-C(a), the 26B with expert caching: started $(date '+%T')"
  python3 "$WT/docs/measurements/kv2-cuda-2026-10-15/gv_plan.py" "$OUT/gv-plan-26b.txt" "$B/bin/serve-base" "$B/bin/serve-rev" \
    "26B=$HOME/models/gemma4-26b-int4.giw,-moe-cache-experts,-kv-sessions,2" || rc=1
  ;;
*) echo "FATAL: JOB=$JOB"; exit 2 ;;
esac
S1=$(swap_mb)
{
  echo "swap in use after: $S1 MB; growth $((S1 - S0)) MB: $([ $((S1 - S0)) -le 64 ] && echo PASS || echo FAIL) (64 MB or less passes)"
  echo "VRAM used after: $(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader)"
} | tee -a "$OUT/never-swap.txt"
echo "finished: $(date '+%F %T %Z') rc=$rc" | tee -a "$OUT/provenance.txt"
cd "$SRC" && for w in "$B/wt-heavy" "$B/wt-base"; do git worktree remove --force "$w" 2>/dev/null; done
exit $rc
