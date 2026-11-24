#!/bin/bash
set -u
export PATH=/opt/homebrew/bin:$PATH TMPDIR=$HOME/goinfer-bench/tmp/mbp2-metal
W=$HOME/goinfer-bench/wg1-mac
mkdir -p $W/bin $W/results
cd ~/goinfer-runner && git fetch -q origin && { [ -d $W/wt ] || git worktree add --detach $W/wt ${SHA:?} >/dev/null; } && cd $W/wt && git checkout -q --detach $SHA && bash scripts/make_work.sh >/dev/null
echo "rev $(git rev-parse --short=9 HEAD) clean=$([ -z "$(git status --porcelain --untracked-files=no)" ] && echo yes || echo no)"
CGO_ENABLED=1 go test -c -tags 'gpu goinfer_testhooks' -o $W/bin/webgpu.test ./backend/webgpu && shasum -a 256 $W/bin/webgpu.test
