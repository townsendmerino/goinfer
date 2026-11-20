#!/usr/bin/env python3
"""Drop the page cache of the given files, and of every regular file under the given directories (posix_fadvise DONTNEED; no root
needed). Prints GB evicted. The re-anchor record's evict_cache.py (peer-reanchor-2026-11-19), extended to take files."""
import os, sys
tot = 0


def drop(p):
    global tot
    try:
        if not os.path.isfile(p):
            return
        fd = os.open(p, os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED); tot += os.fstat(fd).st_size
        finally:
            os.close(fd)
    except OSError:
        pass


for root in sys.argv[1:]:
    root = os.path.expanduser(root)
    if os.path.isfile(root):
        drop(root)
        continue
    for d, _, fs in os.walk(root, followlinks=True):
        for f in fs:
            drop(os.path.join(d, f))
print(f"fadvise DONTNEED over {tot/1e9:.1f} GB of files")
