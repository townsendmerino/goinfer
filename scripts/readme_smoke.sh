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

# RESOLVE EVERY MODEL REFERENCE THE README NAMES, against the registry or HF — metadata only, no
# download. R7 (docs/measurements/cold-user-2026-09-06-nobara-pc.md): the README named
# "qwen3.5-35b-a3b" as its size-class example with no owner/repo behind it anywhere, and
# `goinfer-chat models`'s curated list did not go anywhere near that size — a cold user could not
# find ANY checkpoint actually big enough to need `-stream-weights`/`-moe-cache-experts`. A gate
# that only ran `<!-- smoke -->` commands would not have caught this: `pull demo:35b` failing is
# not something any smoke-marked line does, because the README never told anyone to run it — the
# defect was the ABSENCE of a working example, which only a check that tries to RESOLVE what the
# prose names would catch.
mapfile -t MODELCMDS < <(grep -A 1 -- '<!-- smoke-model -->' "$README" | grep -vE '^(--|.*<!-- smoke)' | sed '/^$/d')
if [ "${#MODELCMDS[@]}" -gt 0 ]; then
	echo "==> installing goinfer-chat, for registry short-name resolution"
	( cd "$WORK" && GOFLAGS= go install github.com/townsendmerino/goinfer/demo/chat@latest ) \
		|| { echo "    could not install demo/chat — registry short-name checks below will fail closed"; }
	registry=""
	if [ -x "$GOBIN/chat" ]; then
		registry="$("$GOBIN/chat" models 2>&1)"
		# G1 (docs/task-gpu-paths-2026-09.md): the pure-Go `go install .../demo/chat@latest` path
		# from the README must at least answer --version — the release-workflow assertions cover
		# the shipped binaries, this covers the path a reader actually runs.
		echo "==> goinfer-chat --version (go install .../demo/chat@latest)"
		if ! ver="$("$GOBIN/chat" --version 2>&1)"; then
			echo "    FAILED: $ver"; fail=1
		else
			echo "$ver" | sed 's/^/    /'
		fi
	fi

	for c in "${MODELCMDS[@]}"; do
		echo "==> (resolve only) $c"
		# Pull the reference argument off a `pull <ref>` or `--model <ref>` invocation — the token
		# after either, stripped of any trailing shell comment.
		ref=$(echo "$c" | sed -E 's/^.*\b(pull|--model)[[:space:]]+//' | sed -E 's/[[:space:]]+#.*$//' | awk '{print $1}')
		if [ -z "$ref" ]; then
			echo "    could not extract a model reference from this line"; fail=1
			ran=$((ran + 1)); continue
		fi
		case "$ref" in
		demo:*)
			tier="${ref#demo:}"
			if ! python3 -c "import json,sys; d=json.load(open('pull/curated.json')); sys.exit(0 if '$tier' in d['tiers'] else 1)" 2>/dev/null; then
				echo "    demo tier \"$tier\" not in pull/curated.json"; fail=1
			fi
			;;
		*/*)
			# owner/repo[:quant] or owner/repo:file.gguf — an explicit HF reference. Metadata only:
			# the models API, not resolve/ (which would start a download).
			repo="${ref%%:*}"
			code=$(curl -s -o /dev/null -w '%{http_code}' "https://huggingface.co/api/models/$repo")
			if [ "$code" != "200" ]; then
				echo "    https://huggingface.co/api/models/$repo -> HTTP $code (not found)"; fail=1
			fi
			;;
		*)
			# A bare registry short name — must appear in `goinfer-chat models`'s own output, so
			# this check reads the exact list `pull <name>` resolves against, not a second one
			# that could drift from it.
			if [ -z "$registry" ]; then
				echo "    no goinfer-chat binary to check the registry against"; fail=1
			elif ! echo "$registry" | grep -qE "^  $ref([[:space:]]|\$)"; then
				echo "    \"$ref\" is not in \`goinfer-chat models\`'s registry"; fail=1
			fi
			;;
		esac
		ran=$((ran + 1))
	done
fi

# EVERY FILE THE README CITES MUST EXIST. R12 (docs/measurements/cold-user-2026-09-06-nobara-pc.md):
# the front page makes numeric claims ("25 s vs 33 s", "192.8 tok/s") each pinned to a
# measurement file by a markdown link — the citation IS the claim's evidence, and a renamed or
# deleted file turns a measured number back into an assertion with nothing behind it. Extracts
# every `[...](path)` link whose path starts with `docs/` (repo-relative; http(s) links are a
# different, external claim and out of scope here) and asserts the target exists in the checkout
# this script itself was invoked from — not the empty $WORK dir, since these are repo docs, not
# something a `go get` installs.
echo "==> every docs/ link the README cites exists"
mapfile -t CITED < <(grep -oE '\]\(docs/[^)]+\)' "$README" | sed -E 's/^\]\(//; s/\)$//; s/#.*$//' | sort -u)
if [ "${#CITED[@]}" -eq 0 ]; then
	echo "    no docs/ citations found — this check would pass having verified nothing"; fail=1
else
	for f in "${CITED[@]}"; do
		if [ ! -e "$ROOT/$f" ]; then
			echo "    README cites $f, which does not exist"; fail=1
		fi
	done
	echo "    checked ${#CITED[@]} citation(s)"
fi
ran=$((ran + 1))

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
