import ctypes, sys, time
mib = int(sys.argv[1]); cu = ctypes.CDLL("libcuda.so.1")
assert cu.cuInit(0) == 0
dev = ctypes.c_int(); assert cu.cuDeviceGet(ctypes.byref(dev), 0) == 0
ctx = ctypes.c_void_p(); assert cu.cuCtxCreate_v2(ctypes.byref(ctx), 0, dev) == 0
ptr = ctypes.c_uint64(); r = cu.cuMemAlloc_v2(ctypes.byref(ptr), ctypes.c_size_t(mib << 20)); assert r == 0, r
print("hogging", mib, "MiB", flush=True); time.sleep(900)
