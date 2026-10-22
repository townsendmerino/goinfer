# P0a′: Metal's and WebGPU's builds read ResolveKV, 2026-10-22

Gates: `docs/tasks/task-kv-precision-everywhere-2026-10.md`, "P0a′ gates" (P0A-G), registered before the code. Machine: MacBook
Pro, M1 Pro, 16 GB, macOS 26.6.2, Go 1.27.2.

| gate | reading | result |
|---|---|---|
| P0A-G1 plumbing | `llama-tiny`: unhooked, an auto request builds f32 on WebGPU and no int8 on Metal; with the resolver hooked, Metal's build allocates int8 and WebGPU's runner takes f16, then int8, for the same auto request (`TestP0a_metalBuildFollowsResolveKV`, `TestP0a_webgpuBuildFollowsResolveKV`). Plants, each red: Metal's `r.kvI8` reading `KVCacheI8` again; WebGPU's `kvF16`/`kvI8` reading `KVCacheF16`/`KVCacheI8` again | **PASS** |
| P0A-G2 unchanged | `TestKVPrecisionDelivered` and `TestKVPrecisionMatrix_fresh` PASS, `docs/kv-precision-matrix.md` byte-unchanged; `scripts/touched.py` green, including the whole `gpu` (213 s) and `metal` packages | **PASS** |
