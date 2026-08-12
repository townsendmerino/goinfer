#!/usr/bin/env bash
#
# mutation_check.sh — a gate's own gate, committed rather than typed.
#
# The policy requires that a gate land with a demonstration it can FAIL. Running that demonstration
# as an ad-hoc one-liner has now produced two defects of its own in this repo, both of which reported
# a mutation as verified while nothing had been exercised:
#
#   - `command -v staticcheck >/dev/null && staticcheck ...` — the binary was not on PATH, the &&
#     short-circuited, the whole check evaluated to nothing, and it was reported as clean.
#   - `python3 lint.py 2>&1 | head -3; echo "exit=$?"` — $? read head's status, not the lint's, so a
#     red mutation printed exit=0.
#
# A mutation check that silently reads the wrong status certifies a gate as falsifiable when nothing
# ran: G-01 inside the mechanism built to prevent G-01. So the status path here contains NO PIPES and
# no && chains — every command's exit code is captured directly into a variable on its own line.
#
# Usage:
#   scripts/mutation_check.sh <name> <file> <sed-expr> <verify-cmd...>
#
#   <file>      is backed up and restored, including on failure or interrupt.
#   <sed-expr>  is applied in place; it MUST change the file (asserted — a no-op mutation is the
#               defect that makes a mutation check vacuous, and it happened: float32(v/sc) where both
#               operands were already float32).
#   <verify>    must EXIT 0 before the mutation and NON-ZERO after it.
#
# Example:
#   scripts/mutation_check.sh int4-quantizer decoder/weightmat.go \
#     's/^const int4GroupSize = 32$/const int4GroupSize = 64/' \
#     go test ./decoder/ -run TestInt4_forwardParity -count=1

set -u

if [ "$#" -lt 4 ]; then
	echo "usage: $0 <name> <file> <sed-expr> <verify-cmd...>" >&2
	exit 2
fi

NAME="$1"
FILE="$2"
EXPR="$3"
shift 3

cd "$(git rev-parse --show-toplevel)" || exit 2

if [ ! -f "$FILE" ]; then
	echo "mutation_check[$NAME]: no such file: $FILE" >&2
	exit 2
fi

BAK="$(mktemp)"
cp "$FILE" "$BAK"
restore() { cp "$BAK" "$FILE"; rm -f "$BAK"; }
trap restore EXIT INT TERM

echo "== mutation_check[$NAME] =="

# 1. BASELINE. A gate that is already red proves nothing when it goes red under mutation — the
#    mutation would not be what turned it. Checked first, deliberately.
"$@" >/dev/null 2>&1
base_rc=$?
if [ "$base_rc" -ne 0 ]; then
	echo "  FAIL: the gate is ALREADY RED before any mutation (exit $base_rc)."
	echo "        A mutation cannot demonstrate falsifiability against a failing baseline."
	exit 1
fi
echo "  baseline green"

# 2. MUTATE, and assert the mutation actually changed something. A sed expression that matches
#    nothing leaves a green run that looks like a verified mutation check.
sed -i "$EXPR" "$FILE"
if cmp -s "$FILE" "$BAK"; then
	echo "  FAIL: the sed expression changed NOTHING — this mutation check is vacuous."
	echo "        expr: $EXPR"
	exit 1
fi
echo "  mutation applied"

# 3. The gate must now be RED. No pipe, no &&: the exit status is captured on its own line.
"$@" >/dev/null 2>&1
mut_rc=$?
if [ "$mut_rc" -eq 0 ]; then
	echo "  FAIL: the gate PASSED under mutation — it cannot see the thing it claims to check."
	exit 1
fi
echo "  red under mutation (exit $mut_rc)"

# 4. RESTORE and confirm green again, so a red is attributable to the mutation and not to drift that
#    happened to coincide with it.
restore
trap - EXIT INT TERM
"$@" >/dev/null 2>&1
back_rc=$?
if [ "$back_rc" -ne 0 ]; then
	echo "  FAIL: still red after restore (exit $back_rc) — the tree did not come back."
	exit 1
fi
echo "  green after restore"
echo "  PASS: $NAME is falsifiable"
