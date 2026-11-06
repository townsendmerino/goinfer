# MEM10: the 26B's host footprint on CUDA today (2026-11, nobara-sb01)

**Units.** Every size here is in MiB (1 MiB = 1,048,576 bytes) or GiB (1 GiB = 1,024 MiB), with the unit written. An earlier version of this README divided MiB by 1,000 and called the result GB, which is neither;
it is retracted and every figure below is restated. The raw JSON of the first two runs (`first-reading-void/`, `reading/`) is in bytes, as that version of the script wrote it; `reading-2/` and `reading-3/` are in MiB (`_mib` keys).
The one figure quoted from elsewhere, the ledger's "M26 12.85 GB", is the ledger's own unit: decimal GB, 12.85 × 10^9 bytes = 12,254 MiB = 11.97 GiB.

MEM10 (`docs/tasks/task-memory-campaign-2026-10.md`, P36 in `docs/queue-performance.md`): "a pre-permuted `cuda` target registered from the mapping; M–L". Its row says B36 (#103) covered the rest, and §1.1's problem text (the whole expert stack in
a fresh heap copy registered as pinned host memory, 12.85 GB decimal) predates #95 and #103. Before an M–L build, a reading of what is left. **A reading, not a build.**

## MEM10-G0, a reading, registered 2026-11-05 before it ran

Read, not graded. It sets the size of what a build would save, or says there is nothing left.

- **Instrument:** `mem10_g0.py` here: `serve` (built from main) at its defaults except `--backend cuda --ctx 4096 --moe-cache-experts`, `~/models/gemma4-26b-q4_k_m.gguf` (local NVMe) through its sidecar `gemma4-26b-q4_k_m.int4.e4h.cuda.v17.giw` (13.27 GiB by `ls`), the exclusive timing lock. `/proc/<pid>/status` when the server listens, 5 s after, after one 16-token greedy completion, sampled every 250 ms for the peak; `/proc/<pid>/smaps` once at listen (the largest mappings, and `Rss` summed by pathname); device memory from `nvidia-smi`.
- **Void, enforced by the script (a plant proves each):** the one-minute load average above 8.0 when the server is started; a server that does not start; a server whose log does not say `decode path: cuda-resident`. `plants.log`: with the load limit set to 0.0 the run is `VOID` before a server starts; with the decode-path check asking for a string the log never says, it is `VOID` after the server has started.
- **Known limits of the instrument:** one run per reading; `/proc` is read from outside the process, so a peak shorter than 250 ms can be missed; `nvidia-smi` lists a process's device memory only when it reports compute apps (otherwise `device_mib` is null, printed n/a, as in the void run); a swap tripwire closing the load is not detected, only `VmSwap` is reported (0 in every run here).

## The first run was void (2026-11-05 17:41 PST)

`first-reading-void/`. Without `--moe-cache-experts` the 26B's CUDA build **declined** (`cuMemAlloc_v2: CUDA_ERROR_OUT_OF_MEMORY` in `upExperts`: the whole expert stack does not fit an 8 GiB card) and the server fell back to the CPU path (`decode path: cpu (int4)`).
Its `RssAnon` 15,258 MiB (14.90 GiB) and `RssFile` 9,728 MiB (9.50 GiB) are the CPU model's. It measures nothing about MEM10. That version of the script printed the decode path and did not enforce it; this one does.

## The readings, 2026-11-05 (the exclusive timing lock, free; `serve` built from main; `decode path: cuda-resident (int4)` in every log)

`reading/` (17:43, the first valid run; bytes), `reading-2/` (19:15, MiB), `reading-3/` (19:16, MiB; adds the `Rss`-by-pathname table). Load average 1.14, 1.70 and 1.13 at the start, about 60,000 MiB available. Three runs of one configuration, read and not graded; they agree to within 100 MiB on every row below.

`reading-3/`, MiB (GiB):

| `/proc/<pid>/status` | at listen (18 s) | after one 16-token completion | peak sampled |
|---|---|---|---|
| `RssAnon` | 172 (0.17) | 224 (0.22) | 320 (0.31) |
| `RssFile` | **13,681 (13.36)** | 13,681 (13.36) | 13,681 (13.36) |
| `RssShmem` | **12,346 (12.06)** | 12,346 (12.06) | 12,346 (12.06) |
| `VmLck` | 0 | 0 | 0 |
| `VmSwap` | 0 | 0 | 0 |
| `VmHWM` | **26,348 (25.73)** | 26,348 (25.73) | 26,348 (25.73) |
| device memory (`nvidia-smi`) | 6,942 (6.78) | | |

`Rss` summed by pathname at listen (`smaps`, `reading-3/reading.json`):

| pathname | mappings | `Rss` MiB (GiB) | what it is |
|---|---|---|---|
| `…/gemma4-26b-q4_k_m.int4.e4h.cuda.v17.giw` | 1 | 13,574 (13.26) | the sidecar, mapped read-only: `RssFile` |
| `/dev/zero (deleted)` | 126 | **12,261 (11.97)**, `rw-s` | a shared anonymous region: `RssShmem` |
| `[anon: Go: heap]` | 1 | 107 (0.10) | the Go heap: `RssAnon` |
| `/dev/nvidiactl` | 27 | 66 (0.06) | the driver's mappings |

**What it says.** The sidecar is mapped (13.26 GiB of file pages) and the process also holds a **11.97 GiB shared anonymous region in 126 mappings of `/dev/zero (deleted)`**, so the experts' bytes are in host memory twice, and `VmHWM` is 25.73 GiB for a 13.27 GiB sidecar. The Go heap is 0.10 GiB: the heap copy is gone, as #103's row says. The ledger's "12.85 GB" (decimal) is 12,254 MiB; this region is 12,261 MiB, 7 MiB more, which matches by size. A pre-permuted target registered from the mapping would remove it: up to about 12 GiB of host memory on this box.

**What the reading does not show.**
- That the region is the expert cache. `smaps` names a pathname, `/dev/zero (deleted)`, and a size; it does not name the allocator. Calling it the expert cache rests on the size match with the ledger's figure and on `upExperts`'s own comment, not on a call stack.
- Whether CUDA can register a file-backed mapping as page-locked and serve the expert slots' uploads from it at today's speed. `upExperts` records that a zero-copy direct read of a host stack was tried and mis-read by `gemv_w4a8_moe` at width; the registered-from-the-mapping plan uploads from the pages, a different question, untested here. A tier 0 probe (register a mapped file, upload slots from it, byte-compare with today's packing) is the first step and is small.
- `VmLck` reads 0, so the kernel does not count the region as locked; how the driver pins it was not checked.
- The `.giw`'s `cuda` target still writes the canonical kind, so a pre-permuted target changes that variant of the file format (version 17's "a new encoding needs no new version" applies) and every cached cuda sidecar.
- One box, an 8 GiB card, 62 GiB of RAM. On a box with less RAM the second copy decides whether the model loads at all.

**No build is started on this reading.** Whether a saving of about 12 GiB justifies an M–L change to the sidecar format is the coordinator's and the owner's call.
