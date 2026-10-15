#!/usr/bin/env bash
# Night jobs for G40 (docs/tasks/task-cpu-prefill-memory-2026-10.md §4, and the amendment before G40-2/-3 ran), on the
# MacBook, from pinned revisions so the tree may move before the night:
#   JOB=mem    G40-2: g40_mem.py, serve -backend cpu at REV (after) and BASE (before), one cold 20,193-token prompt per
#              cell on the 0.5B and 1.5B. About 40 minutes (two sidecar builds, four cells, the 1.5B before arm the
#              longest).
#   JOB=speed  G40-3: TestG40Speed from a test binary built at REV, under the timing lock. About 110 minutes (the 1.5B at
#              20K is three pairs of about 11 minutes a side).
#
#   python3 scripts/night.py add g40-mem --est 45 --by "<who>, G40-2" --doc docs/tasks/task-cpu-prefill-memory-2026-10.md \
#     -- env JOB=mem REV=<sha> BASE=<sha> bash ~/goinfer-bench/g40-2026-10/run-g40-night.sh
#   python3 scripts/night.py add g40-speed --est 120 --by "<who>, G40-3" --doc docs/tasks/task-cpu-prefill-memory-2026-10.md \
#     -- env JOB=speed REV=<sha> bash ~/goinfer-bench/g40-2026-10/run-g40-night.sh
#
# Exit: the harness's (0 PASS, 1 FAIL, 2 VOID) for mem; the test's for speed (the METRIC lines carry each cell's grade).
set -uo pipefail
JOB=${JOB:?JOB=mem|speed}
REV=${REV:?REV is the G40 branch commit}
SRC=$HOME/tmcode/goinfer
B=$HOME/goinfer-bench/g40-2026-10
OUT=$HOME/goinfer-logs/g40-2026-10/night-$(date +%F)-$JOB
command -v go >/dev/null 2>&1 || export PATH=/usr/local/go/bin:/opt/homebrew/bin:$PATH
unset GOWORK # an inherited workspace file would make each worktree's `go work init` write to it instead
mkdir -p "$B/bin" "$OUT" || exit 2
cd "$SRC" || exit 2
git fetch -q origin || echo "warning: fetch failed; using what $SRC has"
wt() { # <name> <rev>: a detached worktree with its own go.work
  local w=$B/wt-$1 rev
  rev=$(git rev-parse --verify "$2^{commit}") || { echo "FATAL: $2 is not a commit here"; exit 2; }
  [ -d "$w" ] && git worktree remove --force "$w"
  git worktree add -q --detach "$w" "$rev" || { echo "FATAL: cannot check out $rev"; exit 2; }
  (cd "$w" && go work init . ./gpu ./metal) || { echo "FATAL: go work init"; exit 2; }
  echo "$w"
}
{ echo "job: $JOB"; echo "started: $(date '+%F %T %Z')"; echo "rev: $REV"; echo "base: ${BASE:-}"; echo "load: $(uptime)";
  echo "swap: $(sysctl -n vm.swapusage)"; } | tee "$OUT/provenance.txt"
case "$JOB" in
mem)
  BASE=${BASE:?BASE is main at the merge base of the branch}
  for side in after:$REV before:$BASE; do
    name=${side%%:*}; w=$(wt "$name" "${side#*:}") || exit 2
    (cd "$w" && GOWORK=$w/go.work CGO_ENABLED=0 go build -o "$B/bin/serve-$name" ./cmd/serve) || { echo "FATAL: the $name build failed"; exit 2; }
  done
  python3 -u "$B/wt-after/docs/measurements/g40-2026-10/g40_mem.py" --before "$B/bin/serve-before" --after "$B/bin/serve-after" \
    --out "$OUT" --repo "$SRC" 2>&1 | tee "$OUT/g40-mem.log"
  rc=${PIPESTATUS[0]}
  ;;
speed)
  w=$(wt after "$REV") || exit 2
  (cd "$w" && GOWORK=$w/go.work go test -c -o "$B/bin/decoder-$REV.test" ./decoder/) || { echo "FATAL: the test build failed"; exit 2; }
  (cd "$w/decoder" && GOINFER_G40_SPEED=1 "$B/bin/decoder-$REV.test" -test.v -test.count=1 -test.timeout 170m \
    -test.run '^TestG40Speed$') 2>&1 | tee "$OUT/g40-speed.log"
  rc=${PIPESTATUS[0]}
  grep -h 'METRIC\|VOID' "$OUT/g40-speed.log" | tee "$OUT/results.txt"
  ;;
*) echo "FATAL: JOB=$JOB"; exit 2 ;;
esac
cd "$SRC" && for w in "$B"/wt-*; do [ -d "$w" ] && git worktree remove --force "$w"; done
echo "finished: $(date '+%F %T %Z') rc=$rc" | tee -a "$OUT/provenance.txt"
exit "$rc"
