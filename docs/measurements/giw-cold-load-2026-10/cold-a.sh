#!/bin/bash
# Cold load by read-ahead mode (exploratory). For each bundle and each mode the file is evicted, then loaded the way a
# self-check or a server start loads it, with no CRC (each bundle carries its verified-marker). Reported: the load
# time, the major faults, and how much of the file is in the page cache afterwards.
set -uo pipefail
D=$HOME/goinfer-bench/coldload; C=$HOME/goinfer-bench/selfcheck/cachestat.py
evict() { python3 -c 'import os,sys; fd=os.open(sys.argv[1],os.O_RDONLY); os.posix_fadvise(fd,0,0,os.POSIX_FADV_DONTNEED); os.close(fd)' "$1"; }
res() { python3 "$C" "$1" | sed 's/.*: //'; }
echo "part A started $(date '+%H:%M:%S')"
for f in "$@"; do
  [ -e "$f.verified" ] || echo "NOTE: $f has no marker; its first load runs the CRC"
  echo "== $(basename "$f") ($(stat -c %s "$f") bytes)"
  for rep in 1 2; do
    for mode in ${MODES:-default random sequential}; do
      evict "$f"; before=$(res "$f")
      if [ $mode = default ]; then r=$("$D/giwload-xp" "$f" 2>&1); else r=$(XP_MADV=$mode "$D/giwload-xp" "$f" 2>&1); fi
      echo "rep $rep  $mode: $r | resident before: $before; after: $(res "$f")"
    done
  done
done
echo "part A finished $(date '+%H:%M:%S')"
