# WG3 families: results

Registered in [registration.md](registration.md). No cell has run yet.

## Harness review (goinfer #1254, merged before its review posted), checked against the code 2026-11-23

1. `-require-backend` was stripped from both arms' TTFT servers, the head arm included. **Holds**, fixed: only the base
   binary (`WG3_BASE_SERVE`) runs without it. Every TTFT cell also records its server's decode path, so a head cell that
   fell back to the CPU would show; WG3-phi's four head cells (its own harness, the same strip) read
   `webgpu:vulkan-resident (int8int8)` each, so none is void.
2. Decode mode never assigned the calibrated prompts. **Refuted**: `scripts/bench_peer.py` reads `BENCH_PROMPTS` itself
   at import (`_PROMPTS`), and `bench.py` imports it after the run script exports the copy's path.
3. The load settle proceeded on timeout, and ran once. **Holds**, fixed: it settles before each cell and stops (exit 75)
   if the load does not fall under 0.8 in 15 minutes. bench_peer's own idle gate and WG0's per-cell void rule applied
   downstream in either case.
4. The nested timing lock, and `$?` through `tee`. The lock is **refuted**: `timing_lock.acquire` inherits a parent
   `timing_lock.py run`'s lock (`inherited=True`). The rc line printed the right code (the shell expands `$?` before the
   pipe runs), but the script went on to the TTFT cell and exited 0 after a failed decode: **holds**, fixed (each cell's
   rc is kept and the script exits with the first failure).
5. `kwin_wayland` unrecorded, a weak base fit check, a hang on a crashed server. The first is **refuted**: on nobara the
   compositor holds a CUDA context, and the compute-apps query lists it (WG3-phi's provenance:
   `/usr/bin/kwin_wayland, 19 MiB`). The other two **hold**, fixed: the base arm's decode path must be `cpu…` or
   `webgpu-staged…`, and `fit()` stops waiting when its server exits.
