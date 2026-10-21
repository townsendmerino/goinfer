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

### Run 1, 2026-10-19 06:53:21 to 06:54:20 PDT (night job `c3-35b-sampled-load`, binary `~/serve-cuda-333407c3`): the load was aborted by the swap guard; not graded

**What stopped it.** The `.giw` load-time watch that #101 added (B33), not the serving watch (armed only after the models load) and not the system backstop. Serve log: `swap guard (load): armed for …int4.e4h.cuda.v17.giw … +512 MB over baseline, system swap +4096 MB as a backstop`, baselines 0.00 GB (server swap) and 8.41 GB (system swap),
then `swap guard (load): TRIPPED — server swap grew 0.62 GB over baseline … aborting the load`, and the build's own error: `cuda: resident build aborted while planning layer 23 of 40`. The abort is the designed one: the CUDA build's planning loop polled the channel at layer 23, which packs that layer's experts on the Go heap.

**Timeline from the log (the process samples were lost, below):** 0 to 13.8 s the whole 19.6 GB `.giw` is read once before the load ("is not in memory: reading its 19.6 GB once before the load"); the load watch arms at 14.2 s; the Go heap then grows from 134 MB to 11.4 GB allocated (8.2 GB live after the collection at 44.5 s) as the layers' expert stacks are packed;
the trip comes a second or two after, at layer 23 of 40 (about 57% of the experts, near 10 GB of the 18.1 GB).

**The 50 ms samples do not exist.** `build_peak.py` died in its cleanup (`killpg` of a process group that had already exited) before it wrote the per-sample file; `report.txt` holds that traceback. Fixed in `78c52d49` (a server that already exited no longer loses the samples). So the `RssAnon`, `RssFile`, `VmSwap` and `Mlocked` timeline the pre-registration asked for is not available from this run, and the pre-registered bar (peak `RssAnon` at most 19.6 GB) is **not assessed**: the load never got past layer 23, so the peak was not reached.
Rule 4 voids a run that ends before readiness for a reason other than the swap watch's abort; this ended by the abort, which the registration called a result in its own right. It is a result about the watch, not a measurement of the footprint.

**What it does say.** With the load watch live, this 35B C′ load does not complete on this box at the default threshold: 0.62 GB of the server's own pages went to swap while the Go heap was filling, on a machine with 34 GB free, `vm.swappiness` 10 and 9.4 GB already in swap from other processes (7.5 GB of it in zram). Tack's C3 run on the build without the watch lost 0.83 and 0.90 GB the same way and was halted for good. The unpriced term is not identified by this run; the data it would need is the lost samples.

### Run 2, 2026-10-19 07:17:31 to 07:18:19 PDT (the same job run by hand under the timing lock, the guard on, the fixed harness; `~/goinfer-logs/c3-35b-sampled-load-rerun-2026-10-19/`): the same abort, and the samples

The load watch tripped again (0.61 GB of the server's swap over its baseline, at layer 23 of 40, about 46 s in), and this time the 50 ms samples were written.

| t (s) | RssAnon | RssFile | VmSwap | Go heap (gctrace): allocated / live |
|---|---|---|---|---|
| 2 | 0.96 GB | 10.32 GB | 0 | |
| 10 | 5.51 | 11.80 | 0 | |
| 14.9 (the watch armed at 14.2) | 7.85 | 12.56 | 0 | 7.8 / 5.7 GB |
| 20 to 21 | 9.97 | 13.40 (peak 13.48 at 21.0) | 0 | |
| 22.5 | 11.69 | 12.56 | 0 | 11.4 / 8.2 GB |
| 34.2 | 17.14 | 10.62 | 0.015 GB | 16.4 / 11.9 GB |
| 42.5 | 18.15 | 11.88 | 0.209 GB | |
| 44.6 | 19.13 | 12.18 | 0.574 GB | |
| 46.3 (the abort) | 19.88 (peak 19.91 at 46.12) | 1.38 | 0.831 GB | |

`Mlocked` read 0 throughout (the pinned registration comes after the planning loop, which the abort never left); the system's swap grew 1,109 MB; the lowest `MemAvailable` was 39.9 GB; `VmHWM` 32.3 GB (19.9 anonymous + 10.6 file at that instant).

**Against the pre-registered bar: FAIL.** Peak `RssAnon` was 19.91 GB, above 19.6 GB, with the build only 57% through (layer 23 of 40). Rule 2: the unpriced anonymous term is named by the timeline. **It is the Go heap that is not live: `RssAnon` minus the collector's live heap was 2.2 GB at 14.9 s, 3.5 GB at 22.5 s and 5.3 GB at 34.2 s, and growing** (anonymous memory tracks the heap
*allocated*, 17.1 GB against 16.4 at 34 s, not the 11.9 GB live). It is garbage waiting for the next collection and the headroom the collector allows (its goal is twice the live heap after the previous cycle): `packWeightStack` packs each expert with `packWeight` into its own slices and copies them into the merged stack, so every expert is allocated twice. The pinned copy is the merged stack itself (`cacheWQ` registers
that slice in place, `RegisterMappedHostBuffer`), so the heap is not a second copy of it, and #100 prices that one. This term is on top.

**The mapped file is not the remainder, contrary to the reading in `c3-logs-reading-2026-10-18.md` §1.1.** `RssFile` peaked at 13.5 GB at 21 s and fell back to 1.4 GB as the heap grew: the kernel evicted the file's pages to make room, then swapped a little of the server's own (0.83 GB).

**Why pages were swapped with 39.9 GB available system-wide: a memory limit on the user's slice.** `/etc/systemd/system/user-1000.slice.d/50-oom-protect.conf` sets `MemoryHigh=52G` on the whole user slice (to keep `systemd-oomd` from killing the desktop and editor). Page cache and mapped files are charged to the slice as well, so a load that holds 20 GB of anonymous heap, the file's pages and everything else the slice caches can cross 52 GiB while the machine as a whole is
far from full, and the kernel then reclaims from the slice, anonymous pages included. The slice's own counters: `memory.events` `high` 13,830,234, `memory.swap.current` 9.4 GiB. This run did not sample the slice's `memory.current`, so it is the likely mechanism, not a measured one.

### Run 3, 2026-10-19 08:48:54 to 08:49:31 PDT (B36 build `ca0c37cf`, `~/serve-cuda-b36-ca0c37cf`, the guard on, no bypass; `~/goinfer-logs/c3-35b-b36-2026-10-19/`): the load completes

The same job as run 2 on the B36 build, which plans the routed-expert stacks and packs them straight into their pinned buffers when the build uploads (`cuda/pack_stream.go:planWeightStack`, `packStackPinned`). One run, by day, under the timing lock; the sampler now also reads `user-1000.slice` `memory.current` and `memory.events` `high` every 50 ms. Load only: no generation was checked on this build.

| | run 2 (`333407c3`, aborted at layer 23 of 40) | run 3 (B36, ready) |
|---|---|---|
| ready | never (aborted ~46 s in) | 29.5 s |
| peak `RssAnon` | 19.91 GB | **0.40 GB** |
| peak `RssFile` | 13.5 GB | 19.41 GB (the mapped sidecar) |
| `VmHWM` | 32.3 GB | 37.94 GB |
| `VmSwap` peak / final | 0.61 GB over baseline (tripped the guard) | **0 / 0** |
| system swap growth | 1,109 MB | 5 MB |
| Go heap (gctrace) peak allocated / live | filled to ~19 GB | 412 / 257 MB |
| lowest `MemAvailable` | 39.9 GB | 41.2 GB |

The slice, at `MemoryHigh` 55.83 GB (52 GiB): `memory.current` 20.8 GB at the start, **46.3 GB at the peak**, 46.2 GB at the end; `memory.events` `high` 13,830,234 before and after, **0 rises in any sample**. With the pinned copy no longer doubled on the heap, this load stays 9.5 GB under the slice limit and the limit forced no reclaim. The setting was read, not changed; whether it would bind a load with other work running is not measured here.

The pre-registered bar (peak `RssAnon` <= 19.6 GB) is met by a wide margin, but the peak `RssAnon` no longer includes the pinned copy at all (`Mlocked` also read 0): `cuMemHostAlloc`ed memory is in neither, so this column cannot be read as "the footprint". The slice's `memory.current` is the figure that includes it.
