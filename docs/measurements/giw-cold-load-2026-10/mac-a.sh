#!/bin/bash
# Step 0 of docs/tasks/task-giw-cold-load-2026-10.md on the Mac (exploratory, no bar), part A: the cold load alone,
# by read-ahead mode. Before each load the bundle is dropped from memory with msync(MS_INVALIDATE) and its residency
# checked; after it, the residency is read again.
set -uo pipefail
D=$HOME/goinfer-bench/coldload-mac
res() { python3 "$D/maccache.py" resident "$1" | sed 's/.*: //'; }
echo "part A started $(date '+%H:%M:%S')"
for f in "$@"; do
  echo "== $(basename "$f") ($(stat -f %z "$f") bytes)"
  for rep in 1 2; do
    for mode in ${MODES:-default random sequential willneed}; do
      python3 "$D/maccache.py" invalidate "$f" > /dev/null; before=$(res "$f")
      if [ $mode = default ]; then r=$("$D/bin/giwload-xp" "$f" 2>&1); else r=$(XP_MADV=$mode "$D/bin/giwload-xp" "$f" 2>&1); fi
      echo "rep $rep  $mode: $r | resident before: $before; after: $(res "$f")"
    done
  done
done
echo "part A finished $(date '+%H:%M:%S')"
