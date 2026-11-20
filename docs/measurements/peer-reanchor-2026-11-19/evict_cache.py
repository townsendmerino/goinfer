#!/usr/bin/env python3
"""Drop the page cache of every regular file under the given roots (posix_fadvise DONTNEED; no root needed). Prints GB evicted."""
import os, sys
tot = 0
for root in sys.argv[1:]:
    for d, _, fs in os.walk(os.path.expanduser(root), followlinks=True):
        for f in fs:
            p = os.path.join(d, f)
            try:
                if not os.path.isfile(p): continue
                fd = os.open(p, os.O_RDONLY)
                try:
                    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED); tot += os.fstat(fd).st_size
                finally: os.close(fd)
            except OSError: pass
print(f"fadvise DONTNEED over {tot/1e9:.1f} GB of files")
