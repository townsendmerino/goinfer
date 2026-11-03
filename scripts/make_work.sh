#!/bin/sh
# Create go.work from go.work.example when it is absent (go.work is gitignored and per-machine).
# `make_work.sh --check` instead verifies that go.work.example lists exactly the workspace modules in the tree.
set -eu
cd "$(dirname "$0")/.."

if [ "${1:-}" = "--check" ]; then
	listed=$(sed -n '/^use (/,/^)/p' go.work.example | sed '1d;$d' | sed 's/^[[:space:]]*//; s#^\./##' | grep -v '^$' | sort)
	# .github/actions/staticcheck pins the CI linter build and is not a workspace module.
	found=$(find . -name go.mod -not -path './testdata/*' -not -path './.github/*' -not -path './.git/*' \
		| sed 's#/go.mod$##; s#^\./##' | sort)
	if [ "$listed" = "$found" ]; then
		echo "go.work.example matches the go.mod files in the tree" >&2
		exit 0
	fi
	printf 'go.work.example and the tree disagree.\nlisted:\n%s\nfound:\n%s\n' "$listed" "$found" >&2
	exit 1
fi

# A GOWORK in the environment names the workspace go uses and overrides ./go.work without a word.
case "${GOWORK:-}" in
	"" | auto) ;;
	*)
		echo "warning: GOWORK=$GOWORK is set, so go ignores ./go.work. Run 'unset GOWORK' (and remove it from your shell profile, or 'go env -w GOWORK=') to use the generated file." >&2
		;;
esac

if [ -e go.work ]; then
	echo "go.work exists; left alone" >&2
else
	cp go.work.example go.work
	echo "wrote go.work from go.work.example" >&2
fi
