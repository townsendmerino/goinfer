#!/usr/bin/env python3
"""maccache.py: page-cache helpers for macOS. `resident FILE` prints the share of FILE in memory (mincore);
`coldcopy SRC DST` writes DST as a copy of SRC with caching off on both descriptors (F_NOCACHE), so DST's pages are
not left in memory; `invalidate FILE` asks msync(MS_INVALIDATE) on a mapping of FILE."""
import ctypes, fcntl, mmap, os, sys

libc = ctypes.CDLL(None, use_errno=True)
PAGE = os.sysconf("SC_PAGESIZE")
libc.mmap.restype = ctypes.c_void_p
libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_longlong]
libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_char_p]
libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
libc.msync.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
F_NOCACHE = 48


def resident(path):
    size = os.path.getsize(path)
    fd = os.open(path, os.O_RDONLY)
    p = libc.mmap(None, size, mmap.PROT_READ, mmap.MAP_SHARED, fd, 0)
    os.close(fd)
    n = (size + PAGE - 1) // PAGE
    vec = ctypes.create_string_buffer(n)
    if libc.mincore(p, size, vec) != 0:
        raise OSError(ctypes.get_errno(), "mincore")
    libc.munmap(p, size)
    return sum(1 for b in vec.raw if b & 1) / n, size


def coldcopy(src, dst):
    i = os.open(src, os.O_RDONLY)
    o = os.open(dst, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    fcntl.fcntl(i, F_NOCACHE, 1)
    fcntl.fcntl(o, F_NOCACHE, 1)
    buf = bytearray(16 << 20)  # a page-aligned multiple, so the writes stay uncached
    mv = memoryview(buf)
    while True:
        n = os.readv(i, [mv])
        if n == 0:
            break
        os.write(o, mv[:n])
    os.fsync(o)
    os.close(i)
    os.close(o)


def invalidate(path):
    size = os.path.getsize(path)
    fd = os.open(path, os.O_RDONLY)
    p = libc.mmap(None, size, mmap.PROT_READ, mmap.MAP_SHARED, fd, 0)
    os.close(fd)
    rc = libc.msync(p, size, 0x0002)  # MS_INVALIDATE
    libc.munmap(p, size)
    return rc


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "resident":
        for f in sys.argv[2:]:
            r, s = resident(f)
            print(f"{os.path.basename(f)}: {100 * r:.1f}% of {s / 1e6:.0f} MB resident")
    elif cmd == "coldcopy":
        coldcopy(sys.argv[2], sys.argv[3])
    elif cmd == "invalidate":
        print("msync rc", invalidate(sys.argv[2]))
