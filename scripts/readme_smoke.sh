#!/usr/bin/env bash
#
# readme_smoke.sh — run the README's install/run commands the way a stranger would.
#
# WHY THIS EXISTS. A cold-user run against v0.16.0 (docs/measurements/cold-user-2026-09-06.md)
# found the README documenting a binary the release did not ship (`goinfer-serve`, used as a
# runnable command three times) and an install line that fetches a module but cannot build
# against it (`go get github.com/townsendmerino/goinfer` → four `missing go.sum entry` errors).
# Both are the kind of defect that is invisible from inside a clone: the repo is on disk, the
# workspace resolves everything, and every command in the README "works" for the person who
# wrote it.
#
# So this runs them the way the reader does: in an EMPTY directory, OUTSIDE the repo, with
# GOWORK=off so no workspace can paper over a missing module, resolving from the module proxy.
#
# WHAT IT RUNS. Every fenced bash block line marked `<!-- smoke -->` in the README. The marker is
# opt-in rather than "every code block" on purpose — the README also shows commands that need a
# multi-gigabyte model, a GPU, or a running server, and a gate that cannot run offline in CI is a
# gate that gets disabled. `<!-- smoke-help -->` marks a line whose FLAGS are checked against
# `--help` without executing it (for commands that need a real model).
set -uo pipefail

README="${1:-README.md}"
[ -f "$README" ] || { echo "readme-smoke: no $README"; exit 2; }
ROOT="$(cd "$(dirname "$README")" && pwd)"

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
export GOWORK=off GOBIN="$WORK/bin" GOFLAGS=
mkdir -p "$GOBIN"

# A throwaway module, because that is the situation a reader is in: `go get` refuses to run
# outside one ("go get is no longer supported outside a module"), and a README line that only
# works in a directory the reader does not have is the same defect class as the missing binary.
( cd "$WORK" && go mod init readmesmoke >/dev/null 2>&1 ) || true

fail=0
ran=0
# Pull the line AFTER each marker out of the fenced blocks.
mapfile -t CMDS < <(grep -A 1 -- '<!-- smoke -->' "$README" | grep -vE '^(--|.*<!-- smoke)' | sed '/^$/d')
mapfile -t HELPCMDS < <(grep -A 1 -- '<!-- smoke-help -->' "$README" | grep -vE '^(--|.*<!-- smoke)' | sed '/^$/d')

if [ "${#CMDS[@]}" -eq 0 ]; then
	echo "readme-smoke: NO <!-- smoke --> commands found — this gate would pass having checked"
	echo "              nothing, which is the failure mode it exists to prevent. Mark at least one."
	exit 1
fi

for c in "${CMDS[@]}"; do
	echo "==> $c"
	( cd "$WORK" && eval "$c" ) || { echo "    FAILED (exit $?)"; fail=1; }
	ran=$((ran + 1))
done

# smoke-help: the command is not executed (it needs a real model), but every flag it names must
# exist. That catches the README documenting a flag the binary does not have — which is exactly
# what `goinfer-chat -web` was: "flag provided but not defined: -web".
if [ "${#HELPCMDS[@]}" -gt 0 ]; then
	if [ -x "$GOBIN/serve" ]; then
		for c in "${HELPCMDS[@]}"; do
			echo "==> (flags only) $c"
			for f in $(echo "$c" | grep -oE '(^| )-[a-zA-Z][-a-zA-Z0-9]*' | tr -d ' '); do
				if ! "$GOBIN/serve" --help 2>&1 | grep -q -- "  $f\b\|^ *$f "; then
					echo "    FLAG NOT IN --help: $f"; fail=1
				fi
			done
			ran=$((ran + 1))
		done
	else
		echo "readme-smoke: INCONCLUSIVE — smoke-help lines present but no serve binary was built"
		echo "              by the <!-- smoke --> steps, so their flags could not be checked."
		fail=1
	fi
fi

# BUILD AGAINST WHAT THE INSTALL LINES INSTALLED. This is the check that actually catches the
# v0.16.0 defect, and the first version of this script did NOT have it — it ran each command and
# checked the exit code, which passes, because `go get github.com/townsendmerino/goinfer`
# SUCCEEDS. The failure the cold-user run hit was at BUILD time:
#
#   missing go.sum entry for module providing package github.com/townsendmerino/aikit/embed
#   (imported by github.com/townsendmerino/goinfer/decoder)
#
# A gate that runs the documented command and stops there cannot see that, and this one was
# proven unable to: with the old install line restored, it still went green. Building a trivial
# importer is what makes "the README's install line works" a claim about the reader's outcome
# rather than about the command's exit status.
if [ -f "$WORK/go.mod" ]; then
	echo "==> build a trivial importer against the installed packages"
	cat > "$WORK/smoke_import.go" <<'GO'
package main

import (
	"fmt"

	"github.com/townsendmerino/goinfer/decoder"
)

func main() { fmt.Println(decoder.Options{}) }
GO
	if ! ( cd "$WORK" && go build -o /dev/null . 2>&1 | sed 's/^/    /' ; exit "${PIPESTATUS[0]}" ); then
		echo "    FAILED — the README's install line fetches a module you cannot build against"
		fail=1
	fi
	ran=$((ran + 1))
fi

# The example the README links must compile. It is the stopgap for scenario C's "no library
# example exists", and an example that does not build is worse than none.
echo "==> go vet ./examples/..."
( cd "$ROOT" && GOWORK=off go vet ./examples/... ) || { echo "    FAILED"; fail=1; }
ran=$((ran + 1))

if [ "$fail" != 0 ]; then
	echo "VERDICT: FAIL — readme-smoke ran $ran check(s), at least one failed"
	exit 1
fi
echo "VERDICT: PASS — readme-smoke ran $ran check(s) from an empty dir with GOWORK=off"
