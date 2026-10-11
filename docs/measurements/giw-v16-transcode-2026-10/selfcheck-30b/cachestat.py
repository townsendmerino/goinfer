#!/usr/bin/env python3
"""cachestat.py <file>...: how much of each file is resident in the page cache (mincore), in MB and percent."""
import ctypes, mmap, os, sys

libc = ctypes.CDLL("libc.so.6", use_errno=True)
PAGE = os.sysconf("SC_PAGESIZE")


def resident(path):
    size = os.path.getsize(path)
    if size == 0:
        return 0, 0
    n = (size + PAGE - 1) // PAGE
    vec = (ctypes.c_ubyte * n)()
    fd = os.open(path, os.O_RDONLY)
    libc.mmap.restype = ctypes.c_void_p
    libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_long]
    p = libc.mmap(None, size, mmap.PROT_READ, mmap.MAP_SHARED, fd, 0)
    os.close(fd)
    if p in (None, ctypes.c_void_p(-1).value):
        raise OSError(ctypes.get_errno(), "mmap")
    libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_ubyte)]
    if libc.mincore(p, size, vec) != 0:
        raise OSError(ctypes.get_errno(), "mincore")
    libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    libc.munmap(p, size)
    return sum(b & 1 for b in vec) * PAGE, size


if __name__ == "__main__":
    for path in sys.argv[1:]:
        try:
            r, s = resident(path)
            print(f"{path}: {r / 1e6:.0f} MB of {s / 1e6:.0f} MB resident ({100 * r / s if s else 0:.1f}%)")
        except OSError as e:
            print(f"{path}: {e}")
