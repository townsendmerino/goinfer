#!/usr/bin/env bash
#
# pre_push_range.sh: the pre-push hook's checks over what is being pushed, run by scripts/pre-push.sh before its slower
# checks. Reads git's pre-push lines ("<local ref> <local sha> <remote ref> <remote sha>") on stdin; exits 1 to refuse.
#
#  1. gofmt -l over the .go files each pushed range changes, read from the pushed commit (not the working tree): CI's
#     `lint` runs gofmt -l . and fails on any output.
#  2. A push to refs/heads/main must be docs-only by scripts/ci_docs_only.sh's default mode, or have a release commit as its
#     tip (a subject starting "release: ", RELEASING.md). Product code reaches main through a pull request (CLAUDE.md).
#
# A range is <remote sha>..<local sha>; a new ref is ranged from its merge-base with origin/main; a deletion is skipped.
# Tests: scripts/test_pre_push.py.
cd "$(git rev-parse --show-toplevel)" || exit 1

zero=0000000000000000000000000000000000000000
refused=0
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

while read -r lref lsha rref rsha; do
    [ -n "${lsha:-}" ] || continue
    [ "$lsha" = "$zero" ] && continue
    [ "$lsha" = "$rsha" ] && continue
    base="$rsha"
    if [ "$rsha" = "$zero" ]; then
        base="$(git merge-base "$lsha" origin/main 2>/dev/null)"
    fi
    if [ -z "$base" ]; then
        echo "pre-push REFUSED: no base to range $lref against (no merge-base with origin/main); git fetch, then push again." >&2
        refused=1
        continue
    fi

    # 1. gofmt
    gofiles="$(git diff --name-only --diff-filter=d "$base" "$lsha" -- '*.go')" || {
        echo "pre-push REFUSED: git diff $base $lsha failed." >&2
        refused=1
        continue
    }
    if [ -n "$gofiles" ]; then
        if ! command -v gofmt >/dev/null 2>&1; then
            echo "pre-push REFUSED: gofmt is not on PATH, and $lref changes .go files; not silently passing." >&2
            refused=1
            continue
        fi
        dir="$tmp/$lsha"
        while IFS= read -r f; do
            mkdir -p "$dir/$(dirname "$f")"
            git show "$lsha:$f" >"$dir/$f"
        done <<<"$gofiles"
        bad="$(cd "$dir" && gofmt -l . 2>&1)"
        if [ -n "$bad" ]; then
            echo "" >&2
            echo "pre-push REFUSED: gofmt -l lists files $lref changes (CI's lint fails on them):" >&2
            printf '%s\n' "$bad" | sed 's|^\./||; s|^|  |' >&2
            echo "  Run gofmt -w on them and commit." >&2
            refused=1
        fi
    fi

    # 2. refs/heads/main
    if [ "$rref" = "refs/heads/main" ]; then
        case "$(git log -1 --format=%s "$lsha")" in
            "release: "*) continue ;;
        esac
        if [ "$(bash scripts/ci_docs_only.sh "$base" "$lsha")" != true ]; then
            echo "" >&2
            echo "pre-push REFUSED: $lref changes more than docs and is pushed to refs/heads/main." >&2
            echo "  Only a docs-only change (scripts/ci_docs_only.sh: docs/**, RELEASING.md, .claude/**, none read by Go code" >&2
            echo "  or tests) or a release (a tip whose subject starts \"release: \", RELEASING.md) goes straight to main." >&2
            echo "  Product code goes through a pull request (CLAUDE.md, \"Working in the tree\"):" >&2
            echo "    git push origin <sha>:<branch>; gh pr create; then, as its own command," >&2
            echo "    gh pr merge <n> --repo townsendmerino/goinfer --auto --merge" >&2
            echo "  If you believe the hold is wrong, stop and report it to Francis. Never --no-verify." >&2
            refused=1
        fi
    fi
done

exit "$refused"
