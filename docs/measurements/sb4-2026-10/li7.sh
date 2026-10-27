#!/usr/bin/env bash
# SB4 LI7: the loader identity at each loader-only commit of VK0's window and at its parent.
# Usage: li7.sh <repo> <out-dir> <identity.go> <sha>...   Each tree is a detached worktree under <out-dir>/wt, removed after.
set -u
repo=$1 out=$2 ident=$3; shift 3
probe=$(dirname "$0")/li7_probe_test.go.txt
mkdir -p "$out/wt"
for c in "$@"; do
  for s in "$(git -C "$repo" rev-parse "$c^")" "$(git -C "$repo" rev-parse "$c")"; do
    [ -f "$out/$s.json" ] && continue
    wt="$out/wt/$s"
    git -C "$repo" worktree add -q --detach "$wt" "$s" || { echo "$s VOID worktree"; continue; }
    cp "$ident" "$wt/decoder/loader_identity.go"
    cp "$probe" "$wt/decoder/sb4_li7_probe_test.go"
    start=$(date +%s)
    if (cd "$wt" && SB4_LI7_OUT="$out/$s.json" go test -count=1 -run '^TestSB4LI7Probe$' ./decoder/ > "$out/$s.log" 2>&1); then
      echo "$s ok $(( $(date +%s) - start ))s"
    else
      echo "$s VOID $(tail -3 "$out/$s.log" | tr '\n' ' ')"
    fi
    git -C "$repo" worktree remove --force "$wt"
  done
done
