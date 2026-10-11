#!/bin/bash
# E1: what a self-check's load costs on the finished 30B bundle, by cache state and by whether the CRC pass runs.
# The file is this build's own output of the 30B directory (weights format v15), written by e2.sh's new arm.
set -uo pipefail
D=$HOME/goinfer-bench/selfcheck
F=${1:-$D/out/30b.new.giw}
drop() { python3 -c 'import os,sys; fd=os.open(sys.argv[1],os.O_RDONLY); os.posix_fadvise(fd,0,0,os.POSIX_FADV_DONTNEED); os.close(fd)' "$F"; }
warm() { cat "$F" > /dev/null; }
echo "E1 on $F ($(stat -c %s "$F") bytes), started $(date '+%H:%M:%S')"
GOINFER_GIW_VERIFY=always "$D/giwload" "$F" > /dev/null # leaves a marker, so an unforced load skips the CRC
for rep in 1 2; do
  for state in cold warm; do
    for crc in crc nocrc; do
      if [ $state = cold ]; then drop; else warm; fi
      before=$(python3 "$D/cachestat.py" "$F" | sed 's/.*: //')
      if [ $crc = crc ]; then r=$(GOINFER_GIW_VERIFY=always "$D/giwload" "$F" 2>&1); else r=$("$D/giwload" "$F" 2>&1); fi
      after=$(python3 "$D/cachestat.py" "$F" | sed 's/.*: //')
      echo "rep $rep  $state  $crc: $r   [before: $before; after: $after]"
    done
  done
done
echo "finished $(date '+%H:%M:%S')"
