#!/bin/bash
# E3: is it the flush? A copy of the 30B bundle is written (as the transcode writes its output), most of it is dropped
# from the page cache (as happens during a transcode), and the self-check's load with its CRC is timed:
#   a) straight away, while the kernel is still writing the copy's dirty pages back;
#   b) after the copy has been synced, so nothing is being written;
#   c) after a sync with the copy left in the page cache.
set -uo pipefail
D=$HOME/goinfer-bench/selfcheck
SRC=$HOME/models/qwen3-vl-30b-a3b-instruct.int4.cpu-amd64.giw
C=$D/out/copy.giw
mi() { grep -E '^(Dirty|Writeback):' /proc/meminfo | tr -s ' ' | tr '\n' ' '; }
evict() { python3 -c 'import os,sys; fd=os.open(sys.argv[1],os.O_RDONLY); os.posix_fadvise(fd,0,0,os.POSIX_FADV_DONTNEED); os.close(fd)' "$C"; }
res() { python3 "$D/cachestat.py" "$C" | sed 's/.*: //'; }
copy() { find "$D/out" -name 'copy.giw*' -delete; dd if="$SRC" of="$C" bs=4M status=none; }
echo "E3 started $(date '+%H:%M:%S'); source $(stat -c %s "$SRC") bytes"
for rep in 1 2 3; do
  copy; evict; echo "rep $rep a: before load: $(mi)| $(res)"; r=$(GOINFER_GIW_VERIFY=always "$D/giwload" "$C" 2>&1); echo "rep $rep a (flush in progress): $r | after: $(mi)"
  copy; sync "$C"; evict; echo "rep $rep b: before load: $(mi)| $(res)"; r=$(GOINFER_GIW_VERIFY=always "$D/giwload" "$C" 2>&1); echo "rep $rep b (synced, evicted):   $r | after: $(mi)"
  copy; sync "$C"; echo "rep $rep c: before load: $(mi)| $(res)"; r=$(GOINFER_GIW_VERIFY=always "$D/giwload" "$C" 2>&1); echo "rep $rep c (synced, cached):    $r | after: $(mi)"
done
find "$D/out" -name 'copy.giw*' -delete
echo "E3 finished $(date '+%H:%M:%S')"
