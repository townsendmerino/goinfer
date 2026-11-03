#!/bin/sh
# Create go.work from go.work.example when it is absent (go.work is gitignored and per-machine).
set -eu
cd "$(dirname "$0")/.."
if [ -e go.work ]; then
	echo "go.work exists; left alone" >&2
else
	cp go.work.example go.work
	echo "wrote go.work from go.work.example" >&2
fi
