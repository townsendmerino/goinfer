#!/usr/bin/env bash
#
# install-git-hooks.sh — install the LOCAL pre-push hook. Opt-in, per clone.
#
# WHY. Three times in one day a citation-lint failure was pushed anyway, each time because an ad-hoc
# `lint && echo GREEN` chain printed nothing on failure and the non-zero status went unread. The last
# one pushed a commit the lint had ALREADY flagged locally as ORPHANED. The rule existed and did not
# fire: the failure is at the moment, not in the knowledge, and a rule that needs remembering at the
# moment is the weakest kind.
#
# The hook removes the step where a human has to remember. It is LOCAL ONLY and deliberately so —
# CI remains the enforcement, and nothing here can be relied upon by anyone else's clone.
#
# Install:  bash scripts/install-git-hooks.sh
# Bypass:   git push --no-verify        (for a genuinely intentional push over a red lint)
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
hook=".git/hooks/pre-push"

cat > "$hook" <<'HOOK'
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
    echo "  Deliberate push over a red lint: git push --no-verify" >&2
    exit 1
fi

exit 0
HOOK

chmod +x "$hook"
echo "installed $hook"
echo "verifying it fires and that it passes on a clean tree:"
if bash "$hook" >/dev/null 2>&1; then
    echo "  clean tree -> hook exits 0 (push allowed)"
else
    echo "  clean tree -> hook exits NON-ZERO; the lint is currently red"
fi
