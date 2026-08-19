#!/usr/bin/env bash
# apidiff_check.sh — the v1.0 API-compatibility gate (release-1.0-gate.md §3).
#
# WHAT IT ACTUALLY ASSERTS, which is narrower than "the API did not change": no INCOMPATIBLE
# change to a HARD-TIER name since the baseline. docs/api-tiers.md splits the surface in two on
# purpose — the backend/residency seam, family descriptors and drafters MOVE, by design, and a gate
# that failed on those would be turned off within a minor. testdata/apidiff/hard_tier.txt is the
# machine-readable half of that document; incompatible changes outside it are REPORTED, not fatal.
#
# Usage:
#   scripts/apidiff_check.sh                 # baseline v0.13.0 (the last released tag)
#   scripts/apidiff_check.sh v0.14.0         # baseline an explicit tag
#
# Needs: apidiff (auto-installed to a temp GOBIN if absent) and the baseline TAG present locally
# (CI must not shallow-clone away the tags: actions/checkout with fetch-depth: 0).
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
BASE="${1:-v0.13.0}"
PKGS="decoder tokenizer chat constrain"
MOD="github.com/townsendmerino/goinfer"
HARD="$ROOT/testdata/apidiff/hard_tier.txt"

if ! git rev-parse -q --verify "refs/tags/$BASE" >/dev/null; then
	echo "apidiff_check: baseline tag $BASE not present locally (shallow clone?). Cannot compare." >&2
	echo "  A missing baseline is NOT a pass — fetch tags and re-run." >&2
	exit 1
fi

APIDIFF="$(command -v apidiff || true)"
if [ -z "$APIDIFF" ]; then
	TMPBIN="$(mktemp -d)/bin"
	echo "apidiff_check: installing apidiff into $TMPBIN"
	GOBIN="$TMPBIN" go install golang.org/x/exp/cmd/apidiff@latest || {
		echo "apidiff_check: could not install apidiff" >&2; exit 1; }
	APIDIFF="$TMPBIN/apidiff"
fi

WORK="$(mktemp -d)"
BASEDIR="$WORK/base"
cleanup() { git worktree remove --force "$BASEDIR" >/dev/null 2>&1 || true; rm -rf "$WORK"; }
trap cleanup EXIT

git worktree add -q --detach "$BASEDIR" "$BASE" || { echo "apidiff_check: worktree at $BASE failed" >&2; exit 1; }

echo "== apidiff: $BASE -> $(git rev-parse --short HEAD) =="
fatal=0
for p in $PKGS; do
	( cd "$BASEDIR" && "$APIDIFF" -w "$WORK/$p.base" "$MOD/$p" ) || {
		echo "  $p: could not export baseline API" >&2; exit 1; }
	out="$("$APIDIFF" "$WORK/$p.base" "$MOD/$p" 2>&1)"
	inc="$(printf '%s\n' "$out" | awk '/^Incompatible changes:/{f=1;next} /^Compatible changes:/{f=0} f' | sed '/^$/d')"
	if [ -z "$inc" ]; then
		echo "  $p: no incompatible changes"
		continue
	fi
	# Split the incompatible changes by tier. The NAME is everything before the first ": ".
	while IFS= read -r line; do
		[ -z "$line" ] && continue
		name="$(printf '%s' "$line" | sed 's/^- //; s/:.*//')"
		if grep -qxF "$p $name" "$HARD"; then
			echo "  $p: HARD-TIER BREAK  $line"
			fatal=$((fatal + 1))
		else
			echo "  $p: (experimental)   $line"
		fi
	done <<< "$inc"
done

if [ "$fatal" -gt 0 ]; then
	echo
	echo "FAIL — $fatal incompatible change(s) to HARD-TIER names since $BASE."
	echo "  docs/api-tiers.md says these follow semver from v1.0. Either revert the change, or move"
	echo "  the name to Experimental IN THE DOCUMENT and in testdata/apidiff/hard_tier.txt — as a"
	echo "  recorded decision, never as a quiet edit to make this gate pass."
	exit 1
fi
echo "PASS — no incompatible change to any Hard-tier name since $BASE."
