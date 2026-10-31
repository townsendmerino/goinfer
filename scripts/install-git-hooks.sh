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
# SECOND CHECK ADDED 2026-08-26, same disease. Three times in one session a decoder/ edit was pushed
# without running the parity staleness gate, and CI went red each time: G18, G20, and a renumber
# whose entire decoder/ diff was TWO COMMENT LINES. The staleness gate keys on a whole-file content
# hash — deliberately, so it can never miss a numeric change (since 2026-10-13 the hash of a Go file is
# its token stream, so comments and layout no longer count) — so "was this edit numeric?" is the
# wrong question and asking it is how the step gets skipped. The gate runs in well under a second,
# so the hook runs it ALWAYS rather than trying to detect whether a change "counts".
#
# The hook removes the step where a human has to remember. It is LOCAL ONLY and deliberately so —
# CI remains the enforcement, and nothing here can be relied upon by anyone else's clone.
#
# Install:  bash scripts/install-git-hooks.sh
# Bypass:   none. A hold you believe is wrong is reported to Francis, never skipped with --no-verify (CLAUDE.md).
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
hook="$(git rev-parse --git-common-dir)/hooks/pre-push" # the shared hooks directory, which a linked worktree uses too

install -m 0755 scripts/pre-push.sh "$hook" # a copy, not a link: the installed hook stays what was installed, whatever branch is checked out

echo "installed $hook"
echo "checks, under a shared hold of the timing lock: queue_citation_lint.py, then TestParityManifest_fresh, then the comment-census warning (never refuses)"
echo "verifying it fires and that it passes on a clean tree:"
if bash "$hook" </dev/null >/dev/null 2>&1; then # no refs on stdin: the hook reads it first, and would wait
    echo "  clean tree -> hook exits 0 (push allowed)"
else
    echo "  clean tree -> hook exits NON-ZERO: the lint or the parity gate is red, or a timed run holds the lock (run bash $hook </dev/null to see which)"
fi
