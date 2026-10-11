#!/bin/bash
# Copy the two archive-only sources to ~/models (the bench set), then verify byte counts.
set -uo pipefail
for d in gemma-4-26b-a4b-it mellum2-unq; do
  echo "== $d started $(date '+%H:%M:%S')"
  rsync -a /srv/models/$d/ $HOME/models/$d/ || { echo "rsync failed for $d"; exit 1; }
  a=$(du -sb /srv/models/$d | cut -f1); b=$(du -sb $HOME/models/$d | cut -f1)
  echo "$d: archive $a bytes, local $b bytes $([ "$a" = "$b" ] && echo MATCH || echo MISMATCH)"
done
echo "pull finished $(date '+%H:%M:%S')"
