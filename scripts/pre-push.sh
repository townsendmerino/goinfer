#!/usr/bin/env bash
#
# pre-push: refuse to push when the committed citation lint fails.
#
# It invokes scripts/queue_citation_lint.py — the SAME entry point CI runs — rather than
# reimplementing any part of it. Same reasoning as the claims rule: the check that counts is the
# committed one, invoked as itself. A hook with its own copy of the logic would be a second
# implementation to drift, and would let this pass while CI fails.
#
# NOTE THE EXIT PATH: an explicit `if` on a captured status, never `cmd && ...` or `cmd || ...`.
# A chain here would swallow the very failure mode this hook exists for, which would be the joke
# writing itself.
cd "$(git rev-parse --show-toplevel)"

# git feeds one "<local ref> <local sha> <remote ref> <remote sha>" line per pushed ref on stdin; read it before anything else can.
refs="$(cat)"

if [ ! -f scripts/queue_citation_lint.py ]; then
    echo "pre-push: scripts/queue_citation_lint.py is missing — not silently passing." >&2
    exit 1
fi

output="$(python3 scripts/queue_citation_lint.py 2>&1)"
status=$?

if [ "$status" -ne 0 ]; then
    echo "" >&2
    echo "pre-push REFUSED: the citation lint exits $status." >&2
    echo "" >&2
    printf '%s\n' "$output" >&2
    echo "" >&2
    echo "  A rebase or amend rewrites SHAs; a cited one then resolves nowhere for anybody else." >&2
    echo "  Re-point the citation, or run --update if it is only the index that is stale." >&2
    echo "  If the red is a doc you have not committed yet, commit it — do not move it out of the tree." >&2
    echo "  If you believe the hold is wrong, stop and report it to Francis. Never --no-verify." >&2
    exit 1
fi

# The parity staleness gate. Run UNCONDITIONALLY: it costs well under a second, and every attempt
# to run it "only when the edit is core" has failed — most recently on a two-comment diff, which no
# one would have classified as core. Same invocation CI runs; no reimplementation to drift.
parity="$(go test ./decoder/ -run TestParityManifest_fresh -count=1 2>&1)"
pstatus=$?

if [ "$pstatus" -ne 0 ]; then
    echo "" >&2
    echo "pre-push REFUSED: TestParityManifest_fresh exits $pstatus." >&2
    echo "" >&2
    printf '%s\n' "$parity" | head -20 >&2
    echo "" >&2
    echo "  A code edit to a shared decoder/ file re-stales every family that uses it. Comments and" >&2
    echo "  layout are not hashed (goCodeTokens), so this is a code change: the gate working." >&2
    echo "  Numeric change:      re-run T3 (go run ./cmd/gate parity), then -update." >&2
    echo "  Provably non-numeric: scripts/refresh_parity_hashes.sh — it runs the forward goldens" >&2
    echo "                        as independent proof and REFUSES if any fail." >&2
    echo "  If you believe the hold is wrong, stop and report it to Francis. Never --no-verify." >&2
    exit 1
fi

# The comment census WARNING (CC6, docs/tasks/task-code-comments-2026-10.md): comments added or changed in the pushed .go files that carry
# a history marker. It never refuses a push (the script exits 0 and `|| true` makes that explicit): "timeout in ms" is a contract, not history.
if [ -f scripts/comment_census.py ]; then
    printf '%s\n' "$refs" | while read -r lref lsha rref rsha; do
        [ -n "${lsha:-}" ] || continue
        case "$lsha" in 0000000000000000000000000000000000000000) continue ;; esac
        base="$rsha"
        case "$rsha" in 0000000000000000000000000000000000000000) base="$(git merge-base "$lsha" origin/main 2>/dev/null || true)" ;; esac
        [ -n "$base" ] || continue
        python3 scripts/comment_census.py diff --base "$base" --head "$lsha" >&2 || true
    done
fi

exit 0
