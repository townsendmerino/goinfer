#!/usr/bin/env bash
# ci_docs_only.sh BASE HEAD   -> prints "true" or "false"
# ci_docs_only.sh --selfcheck -> exits 1 if the read-detection below has stopped catching the
#                                docs files tests are KNOWN to read (a zero-match guard)
#
# C3 (docs/tasks/task-ci-speed-2026-09.md). "true" iff EVERY file changed between BASE and HEAD is
# under docs/ AND NONE of them is read by Go code or tests. Two rules, both load-bearing:
#
#  1. Only docs/** counts. Root *.md (README.md, CHANGELOG.md, RELEASING.md) and everything else
#     run the full matrix -- readme-smoke exists because a README install line once shipped broken.
#
#  2. A docs file that Go code READS is not "docs-only", whatever its path. Measured 2026-09-22
#     over the 30 most recent commits on main: 11 were docs-only by rule 1, and 4 of those touched
#     a file a test reads -- env-vars.md (TestEnvVars_docAndCodeAgree), benchmarks.md
#     (cuda/spec_noncopy_lane_test.go), audit-metal-2026-09-12.md (metal/moe_expert_reuse_probe_test.go).
#     477da08a is the precedent: a docs-only fix to env-vars.md flipped a test from red to green,
#     so skipping `test` on a "docs-only" push would have left main red with nothing watching.
#     Detection is DYNAMIC, not a curated list (a list drifts as tests are added): a changed file's
#     basename appearing on a Go line that looks like a file read. Plain mentions in comments and
#     error strings do NOT count -- this repo's code cites task docs constantly, and counting those
#     left only 2 of the 11 eligible (the measured alternative). --selfcheck pins that the
#     read-shaped detection still catches the known reads, so a regex regression goes red in `lint`
#     instead of silently widening what gets skipped.
#
# FAIL-OPEN, ALWAYS: any doubt -- missing/zero/unknown SHAs, a diff that errors, an empty diff, a
# failed self-check -- prints "false" (run everything) and exits 0. The gate job must never fail,
# because a failed `needs:` dependency makes dependent jobs SKIP, which is the one outcome this
# script exists to never produce.
set -u

read_re='ReadFile|ReadAll|Open\(|OpenFile|Stat\(|Lstat\(|ReadDir|WalkDir|Glob\(|filepath\.Join|path\.Join|go:embed|Path'

# read_by_go BASENAME -> exit 0 if some non-vendor Go line mentions BASENAME in a read-shaped context
read_by_go() {
  git grep -F -e "$1" -- '*.go' ':!vendor' 2>/dev/null | grep -qE "$read_re"
}

selfcheck() {
  # Every entry here is a docs/ file a test READS today (see the header). If detection stops
  # seeing one, the gate has silently widened; fail loudly.
  local ok=0
  for b in env-vars.md capability-matrix.json hardware-matrix.md api-tiers.md benchmarks.md audit-metal-2026-09-12.md; do
    if ! read_by_go "$b"; then
      echo "ci_docs_only: SELF-CHECK FAILED: '$b' is read by a Go test but the read-detection no longer sees it" >&2
      ok=1
    fi
  done
  return $ok
}

if [ "${1:-}" = "--selfcheck" ]; then
  selfcheck && echo "ci_docs_only: self-check OK (read-detection still catches the known test-read docs)"
  exit $?
fi

say_false() { echo false; exit 0; }

base="${1:-}"
head="${2:-}"
[ -n "$base" ] && [ -n "$head" ] || say_false
case "$base" in *[!0-9a-f]*|"") say_false ;; esac   # not a SHA at all
case "$base" in 0000000000000000000000000000000000000000) say_false ;; esac  # branch creation: no "before"

selfcheck >/dev/null 2>&1 || say_false

files=$(git diff --name-only "$base" "$head" 2>/dev/null) || say_false
[ -n "$files" ] || say_false

while IFS= read -r f; do
  case "$f" in
    docs/*) ;;
    *) say_false ;;
  esac
done <<<"$files"

while IFS= read -r f; do
  read_by_go "$(basename "$f")" && say_false
done <<<"$files"

echo true
exit 0
