# Qwen3.6-35B-A3B with `--moe-cache-experts`: the host footprint of the load, sampled (pre-registration, 2026-10-19)

**Registered before the run** (owner's instruction, 2026-10-19; the question and the reading it tests are in `docs/measurements/c3-logs-reading-2026-10-18.md` §1.1). The run is a night job; this file is committed first.

## Question

Tack's C3 run read a peak resident set of 33.06 GB (`getrusage` `Maxrss` of the whole serve process, load included) for this arm, against the 16.9 GiB (18.12 GB) the `.giw` fit guard prices for the pinned expert copy (PR #100). From the logs alone the
remainder was inferred to be mostly the mapped `.giw`'s resident pages (reclaimable) plus about 2.4 GB of transient and process memory. This run measures it: how the peak splits into anonymous memory and the mapped file, from the start of the load, with the process's own swap and the system's.

## Instrument

`sample35b.py` (this directory) wraps `docs/measurements/cuda-build-heap-2026-10-18/build_peak.py`: one fresh `serve -backend cuda -ctx 32768 -moe-cache-experts` on `~/models/qwen3.6-35b-a3b-q4_k_m.gguf` (its existing `.giw` sidecar), sampled every 50 ms from spawn through 5 s past readiness:
`VmRSS`, `RssAnon`, `RssFile`, `VmHWM`, `VmSwap` of the process, and from `/proc/meminfo` the system's swap in use (`SwapTotal - SwapFree`), `MemAvailable`, `Cached` and `Mlocked` (page-locked memory: the pinned copy). Binary: `~/serve-cuda-333407c3`, built from main after #101 and
#95 (so the streaming pack and the `.giw` load-time swap watch are in it: the dense weights are not packed on the heap, and a swap-growing load may abort, which would be a result too). One run (the load is about 70 s; the interest is the shape and the split, not a ratio). Timing lock held, nothing else on the box.

## Pre-registered reading

Let `E` = 18.12 GB, the routed experts' page-locked copy (the memory campaign's census; what #100 prices) and `B` = the process's anonymous baseline outside the experts, taken as 0.5 GB (the 7B sidecar load's final `RssAnon` was 0.09 GB, and a CUDA process holds a little more).

1. **The peak `RssAnon` is `E` plus no other unpriced term above about 1 GB: `peak RssAnon <= E + B + 1 GB = 19.6 GB`: PASS.** Then #100's pricing covers the non-reclaimable footprint of this load to within the stated margin, and the rest of the 33 GB is file-backed.
2. **`peak RssAnon > 19.6 GB`: FAIL.** The unpriced anonymous term is the difference; the sampled timeline (the instant of the peak, what `Mlocked` and `RssFile` read then) names it, and #100 is amended or a new item filed before it merges. No bar is moved after the run.
3. Reported either way, not graded: the peak `RssFile` and the instant the file-backed share was largest; `VmSwap` of the process (peak and final) against the 512 MB threshold; the system's swap growth; whether `Mlocked` reaches `E`; and the sum `RssAnon + RssFile` at the instant of `VmHWM`, against tack's 33.06 GB.
4. **Void** if the server exits before readiness for a reason other than the swap watch's abort, or if another timed run held the lock.

## Estimated wall time

About 4 minutes of GPU (a 70 s load, 5 s of sampling past readiness, the 19.6 GB sidecar read through the page cache), plus the idle gate. Night queue job `c3-35b-sampled-load`, est 8 min.

## Results (appended when read)
