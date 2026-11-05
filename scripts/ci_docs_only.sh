#!/usr/bin/env bash
# ci_docs_only.sh BASE HEAD   -> prints "true" (the quick path) or "false" (the full suite)
# ci_docs_only.sh --classify BASE HEAD -> the CI quick path's wider rule 1 (below): prints docs, records, version, quick (a mix) or full
# ci_docs_only.sh --read PATH -> prints "read" or "not-read", rule 2's decision for one repo-relative path
# ci_docs_only.sh --selfcheck -> exits 1 if the read-detection below has stopped catching the
#                                docs files tests are KNOWN to read (a zero-match guard)
#
# C3 (docs/completed/task-ci-speed-2026-09.md). "true" iff EVERY file changed between BASE and HEAD is
# on a non-code path AND NONE of them is read by Go code or tests. Two rules, both load-bearing:
#
#  1. The default mode counts only docs/**, RELEASING.md and .claude/** (the direct-push rule in CLAUDE.md and coverage.yml
#     use it, and it does not widen). --classify, which ci.yml's `changes` job uses, counts these paths as quick (the class
#     names are for the log; the quick path is one path):
#       records  docs/measurements/**
#       docs     the rest of docs/**, RELEASING.md, .claude/**, changelog.d/**, a top-level *.md other than README.md,
#                and apps/site content (anything under apps/site/ that is not Go, go.mod/go.sum, wrangler.jsonc or cmd/)
#       version  a go.mod or go.sum (any module except apps/site) whose diff changes only lines for
#                github.com/townsendmerino/goinfer or goinfer/backend/* module versions
#     Everything else is the full suite: any Go, script, workflow or testdata file, README.md (readme-smoke reads it,
#     and so do Go tests), any other dependency's version, a go.mod/go.sum line that is not a goinfer version.
#  2. A docs file that Go code READS is not "docs-only", whatever its path. Measured 2026-09-22
#     over the 30 most recent commits on main: 11 were docs-only by rule 1, and 4 of those touched
#     a file a test reads -- env-vars.md (TestEnvVars_docAndCodeAgree), benchmarks.md
#     (cuda/spec_noncopy_lane_test.go; it reads a frozen copy since 2026-10-08, so benchmarks.md left the
#     --selfcheck list below), audit-metal-2026-09-12.md (metal/moe_expert_reuse_probe_test.go).
#     477da08a is the precedent: a docs-only fix to env-vars.md flipped a test from red to green,
#     so skipping `test` on a "docs-only" push would have left main red with nothing watching.
#     Detection is DYNAMIC, not a curated list (a list drifts as tests are added): a changed file's
#     FULL repo path on a Go line that looks like a file read, either contiguous ("../docs/api-tiers.md") or as its
#     components in order, each a quoted literal (filepath.Join(root, "docs", "env-vars.md")). Not its basename alone:
#     that held every docs/**/README.md and run.py as read, because Go tests read the root README.md (2026-10-18). Plain mentions in comments and
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

# read_by_go PATH -> exit 0 if some non-vendor, read-shaped Go line names PATH (repo-relative): the path itself, or each of
# its components as a quoted literal, in order.
read_by_go() {
  local path="$1" base comps re c
  base="$(basename "$path")"
  re=""
  IFS=/ read -r -a comps <<<"$path"
  for c in "${comps[@]}"; do
    re="${re:+$re.*}\"$(printf '%s' "$c" | sed 's/[][\.*^$+?(){}|]/\\&/g')\""
  done
  git grep -F -e "$base" -- '*.go' ':!vendor' 2>/dev/null | grep -E "$read_re" | grep -qE -e "$(printf '%s' "$path" | sed 's/[][\.*^$+?(){}|]/\\&/g')" -e "$re"
}

selfcheck() {
  # Every entry here is a docs/ file a test READS today (see the header). If detection stops
  # seeing one, the gate has silently widened; fail loudly.
  local ok=0
  for b in docs/env-vars.md docs/capability-matrix.json docs/hardware-matrix.md docs/api-tiers.md docs/audit-metal-2026-09-12.md; do
    if ! read_by_go "$b"; then
      echo "ci_docs_only: SELF-CHECK FAILED: '$b' is read by a Go test but the read-detection no longer sees it" >&2
      ok=1
    fi
  done
  return $ok
}

if [ "${1:-}" = "--read" ]; then # ci_docs_only.sh --read PATH -> "read" or "not-read": rule 2's decision for one path
  read_by_go "${2:-}" && echo read || echo not-read
  exit 0
fi

if [ "${1:-}" = "--selfcheck" ]; then
  selfcheck && echo "ci_docs_only: self-check OK (read-detection still catches the known test-read docs)"
  exit $?
fi

say_false() { echo false; exit 0; }

# class_of PATH -> records | docs | gomod | code
class_of() {
  case "$1" in
    docs/measurements/*) echo records; return ;;
    docs/*|RELEASING.md|.claude/*) echo docs; return ;;
  esac
  [ "$WIDE" = 1 ] || { echo code; return; }
  case "$1" in
    README.md) echo code ;;
    changelog.d/*) echo docs ;;
    apps/site/cmd/*|apps/site/*.go|apps/site/go.mod|apps/site/go.sum|apps/site/wrangler.jsonc) echo code ;;
    apps/site/*) echo docs ;;
    go.mod|go.sum|*/go.mod|*/go.sum) echo gomod ;;
    */*) echo code ;;
    *.md) echo docs ;;
    *) echo code ;;
  esac
}

mod_re='github\.com/townsendmerino/goinfer(/[A-Za-z0-9._/-]+)?'
gomod_line_re="^[+-][[:space:]]*(require[[:space:]]+)?${mod_re}[[:space:]]+v[0-9][^[:space:]]*([[:space:]]+//.*)?\$"
gosum_line_re="^[+-]${mod_re}[[:space:]]+v[0-9][^[:space:]]*(/go\\.mod)?[[:space:]]+h1:[A-Za-z0-9+/=]+\$"

# version_only BASE HEAD FILE -> exit 0 if the diff of FILE changes only goinfer module versions
version_only() {
  local base="$1" head="$2" f="$3" re lines
  case "$f" in
    */go.sum|go.sum) re="$gosum_line_re" ;;
    *) re="$gomod_line_re" ;;
  esac
  lines=$(git diff -U0 "$base" "$head" -- "$f" 2>/dev/null | grep -E '^[+-]' | grep -vE '^(\+\+\+|---) ') || return 1
  [ -n "$lines" ] || return 1
  ! printf '%s\n' "$lines" | grep -vE -e "$re" | grep -q .
}

# classify BASE HEAD -> docs | records | version | quick | full  (fail-open: any doubt is "full")
classify() {
  local base="$1" head="$2" files f c
  [ -n "$base" ] && [ -n "$head" ] || { echo full; return; }
  case "$base" in *[!0-9a-f]*|"") echo full; return ;; esac   # not a SHA at all
  case "$base" in 0000000000000000000000000000000000000000) echo full; return ;; esac  # branch creation: no "before"

  selfcheck >/dev/null 2>&1 || { echo full; return; }

  files=$(git diff --name-only "$base" "$head" 2>/dev/null) || { echo full; return; }
  [ -n "$files" ] || { echo full; return; }

  local n_docs=0 n_records=0 n_version=0
  while IFS= read -r f; do
    c=$(class_of "$f")
    case "$c" in
      records) n_records=$((n_records+1)) ;;
      docs) n_docs=$((n_docs+1)) ;;
      gomod)
        version_only "$base" "$head" "$f" || { echo full; return; }
        n_version=$((n_version+1)) ;;
      *) echo full; return ;;
    esac
  done <<<"$files"

  # Rule 2 does not apply to a version-only go.mod/go.sum: tests read those files, and a goinfer version line is not
  # what they read them for. It applies to every other file.
  while IFS= read -r f; do
    [ "$(class_of "$f")" = gomod ] && continue
    read_by_go "$f" && { echo full; return; }
  done <<<"$files"

  if [ "$n_docs" -gt 0 ] && [ "$n_records" -eq 0 ] && [ "$n_version" -eq 0 ]; then echo docs
  elif [ "$n_records" -gt 0 ] && [ "$n_docs" -eq 0 ] && [ "$n_version" -eq 0 ]; then echo records
  elif [ "$n_version" -gt 0 ] && [ "$n_docs" -eq 0 ] && [ "$n_records" -eq 0 ]; then echo version
  else echo quick
  fi
}

WIDE=0
if [ "${1:-}" = "--classify" ]; then
  WIDE=1
  classify "${2:-}" "${3:-}"
  exit 0
fi

[ "$(classify "${1:-}" "${2:-}")" = full ] && say_false
echo true
exit 0
