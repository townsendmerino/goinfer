# N GiB of INCOMPRESSIBLE anonymous memory, kept active. Allocated ONCE and filled in 64 MiB chunks of random
# bytes, so the peak is N GiB (a first version built bytearray(os.urandom(n)) — two N GiB copies at once).
import os, sys, time
n = int(sys.argv[1]) << 30
a = bytearray(n)
chunk = 64 << 20
for off in range(0, n, chunk):
    a[off:off + chunk] = os.urandom(chunk)
print(f"hog: {n>>30} GiB resident", flush=True)
page = 16384
while True:
    for i in range(0, n, page):
        a[i] ^= 1
    time.sleep(0.2)
