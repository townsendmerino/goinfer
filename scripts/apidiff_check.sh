#!/usr/bin/env bash
# apidiff_check.sh — the v1.0 API-compatibility gate (release-1.0-gate.md §3).
#
# WHAT IT ACTUALLY ASSERTS, which is narrower than "the API did not change": no INCOMPATIBLE
# change to a HARD-TIER name since the baseline. docs/api-tiers.md splits the surface in two on
# purpose — the backend/residency seam, family descriptors and drafters MOVE, by design, and a gate
# that failed on those would be turned off within a minor. testdata/apidiff/hard_tier.txt is the
# machine-readable half of that document; incompatible changes outside it are REPORTED, not fatal.
# testdata/apidiff/hard_tier_aliases.txt licenses two lines for a Hard-tier name that became a same-name alias (each entry
# Francis's call); scripts/test_apidiff_check.py holds those rules and the plants that must stay fatal.
#
# Usage:
#   scripts/apidiff_check.sh                 # baseline v0.13.0 (deliberately frozen — the
#                                             # pre-1.0 API-freeze point, not "the last released
#                                             # tag" (N-54, docs/audit-2026-09-10.md) — several
#                                             # tags have shipped since)
#   scripts/apidiff_check.sh v0.14.0         # baseline an explicit tag
#
# Needs: apidiff (auto-installed to a temp GOBIN if absent) and the baseline TAG present locally
# (CI must not shallow-clone away the tags: actions/checkout with fetch-depth: 0).
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PKGS="decoder tokenizer chat constrain"
MOD="github.com/townsendmerino/goinfer"
HARD="${APIDIFF_HARD:-$ROOT/testdata/apidiff/hard_tier.txt}"
ALIASES="${APIDIFF_ALIASES:-$ROOT/testdata/apidiff/hard_tier_aliases.txt}"

# aliased PKG LINE succeeds when LINE is one of the two lines hard_tier_aliases.txt licenses for an entry "PKG NAME = PATH.NAME":
# "NAME: changed from NAME to <module>/PATH.NAME", or a method whose old and new signature text are the same and name NAME as a
# whole word, because only that type's identity moved (such as (*Model).Config). Nothing else.
aliased() {
	local p="$1" body="${2#- }" ep name target from to
	[ -f "$ALIASES" ] || return 1
	while read -r ep name eq target _; do
		case "$ep" in '' | \#*) continue ;; esac
		[ "$ep" = "$p" ] && [ "$eq" = "=" ] || continue
		[ "${target##*.}" = "$name" ] || continue # the alias keeps its name
		[ "$body" = "$name: changed from $name to $MOD/$target" ] && return 0
		case "$body" in "("*")."*": changed from func("*" to func("*) ;; *) continue ;; esac # a method, its signature
		from="${body#*: changed from }"
		to="${from#* to }"
		from="${from%% to *}"
		[ "$from" = "$to" ] && printf '%s' "$from" | grep -qw -- "$name" && return 0
	done < "$ALIASES"
	return 1
}

# classify PKG LINE prints one of HARD, ALIAS, EXPERIMENTAL for one line of apidiff's "Incompatible changes" list.
classify() {
	local p="$1" line="$2" name
	name="$(printf '%s' "$line" | sed 's/^- //; s/:.*//')"
	if ! grep -qxF "$p $name" "$HARD"; then
		echo EXPERIMENTAL
	elif aliased "$p" "$line"; then
		echo ALIAS
	else
		echo HARD
	fi
}

# --classify PKG: classify apidiff's incompatible lines from stdin, one verdict and the line per output line; exit 1 when one is HARD.
# scripts/test_apidiff_check.py drives it with planted lines and its own copies of the two lists (APIDIFF_HARD, APIDIFF_ALIASES).
if [ "${1:-}" = "--classify" ]; then
	bad=0
	while IFS= read -r line; do
		[ -z "$line" ] && continue
		v="$(classify "$2" "$line")"
		echo "$v $line"
		[ "$v" = HARD ] && bad=1
	done
	exit "$bad"
fi

BASE="${1:-v0.13.0}"

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
		case "$(classify "$p" "$line")" in
		HARD)
			echo "  $p: HARD-TIER BREAK  $line"
			fatal=$((fatal + 1))
			;;
		ALIAS)
			echo "  $p: (listed alias)   $line"
			;;
		*)
			echo "  $p: (experimental)   $line"
			;;
		esac
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
