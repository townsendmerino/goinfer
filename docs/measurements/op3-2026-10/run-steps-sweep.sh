#!/bin/bash
# OP3: the whole decoder package at every commit of #62 (rebuilt one commit per step) and at #61's head, each on its own
# (owner, 2026-10-16: "provided each commit passes the full decoder package on its own"). A detached worktree carries the
# main checkout's gitignored fixtures as APFS clones (never symlinks: two fixture-copying tests go red under links), so it
# runs the same tests the main checkout does. Per commit: go test -count=1 -timeout 25m -skip Real0p5B ./decoder/
# (Real0p5B is G40's 13-minute real-checkpoint cell, a tier 1 test of its own). One line per commit in the summary.
set -u
R=$HOME/tmcode/goinfer
B=$HOME/goinfer-bench/op3-sweep
WT=$B/wt
SUMMARY=$B/summary.txt
COMMITS="10eddc85 4f7638ab dc3d184f aaede08a 6cc6054f be6b44c2 0497c223 f1a8f06a 0218e08c ade79e9d f402f41b 3f508449"
ts() { date '+%F %T %Z'; }
mkdir -p "$B/logs"
git -C "$R" fetch -q origin || true
[ -d "$WT" ] || git -C "$R" worktree add --detach "$WT" 10eddc85 || exit 1
sed "s#\.\./aikit#$HOME/tmcode/aikit#" "$R/go.work" > "$WT/go.work"
echo "$(ts) cloning the gitignored fixtures"
(cd "$R" && git ls-files --others --ignored --exclude-standard -- testdata decoder/testdata) | while IFS= read -r f; do
  [ -e "$WT/$f" ] && continue
  mkdir -p "$WT/$(dirname "$f")" && cp -c "$R/$f" "$WT/$f"
done
echo "sweep started $(ts)" > "$SUMMARY"
n=0; total=$(echo $COMMITS | wc -w | tr -d ' ')
for c in $COMMITS; do
  n=$((n+1))
  git -C "$WT" checkout -q --detach "$c" || { echo "$c CHECKOUT FAILED" >> "$SUMMARY"; continue; }
  echo "$(ts) [$n/$total] $c $(git -C "$WT" log -1 --format=%s "$c" | cut -c1-90)"
  t0=$(date +%s)
  (cd "$WT" && go test -count=1 -timeout 25m -skip 'Real0p5B' ./decoder/) > "$B/logs/$c.log" 2>&1
  rc=$?
  fails=$(grep -E '^--- FAIL' "$B/logs/$c.log" | awk '{print $3}' | tr '\n' ' ')
  line="$c rc=$rc $(( $(date +%s) - t0 ))s ${fails:-none failed}"
  echo "$(ts) $line"
  echo "$line" >> "$SUMMARY"
done
echo "sweep done $(ts)" >> "$SUMMARY"
cat "$SUMMARY"
