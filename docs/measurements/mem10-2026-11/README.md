# MEM10: the 26B's host footprint on CUDA today (2026-11, nobara-sb01)

MEM10 (`docs/tasks/task-memory-campaign-2026-10.md`, P36 in `docs/queue-performance.md`): "a pre-permuted `cuda` target registered from the mapping; M–L". Its row says B36 (#103) covered the rest, and §1.1's problem text (the whole expert
stack in a fresh heap copy registered as pinned host memory, M26 12.85 GB) predates #95 and #103. Before an M–L build, a reading of what is left.

## MEM10-G0, a reading, registered 2026-11-05 (time from `date` at the run) before it ran

Read, not graded; it sets the size of what a build would save, or says there is nothing left.

- **Instrument:** `mem10_g0.py` here: `serve` (built from main, `CGO_ENABLED=0 go build -tags cuda ./backend/cuda/cmd/serve`) at its defaults except `--backend cuda --ctx 4096`, `~/models/gemma4-26b-q4_k_m.gguf` (local NVMe, through its `.giw` sidecar `gemma4-26b-q4_k_m.int4.e4h.cuda.v17.giw`), the exclusive timing lock. `/proc/<pid>/status` (`RssAnon`, `RssFile`, `RssShmem`, `VmLck`, `VmSwap`, `VmHWM`) when the server listens, 5 s after, after one 16-token greedy completion, and the peak sampled every 250 ms; device memory from `nvidia-smi`.
- **What it answers:** how much anonymous or pinned host memory the 26B holds beyond the mapped weights, which is what the ledger's "registered from the mapping" would remove (the pinned copy shows as `RssShmem`/`VmLck`, the heap copy as `RssAnon`).
- **Void:** the one-minute load above 8.0 at the start, or the swap tripwire closing the load. **One run.** Cost about 2 minutes.
- **Nothing is decided on this reading:** whether a build is worth its size is the coordinator's and the owner's call; this README adds the figures and the arithmetic of what could be saved.

## The first reading was void (2026-11-05 17:41 PST)

`first-reading-void/`. Started at serve's defaults with `--backend cuda --ctx 4096`, the 26B's CUDA build **declined** (`cuMemAlloc_v2: CUDA_ERROR_OUT_OF_MEMORY` in `upExperts`: the whole expert stack does not fit an 8 GB card) and the server fell back to the CPU path (`decode path: cpu (int4)`, `serve.log`). Its 15.3 GB `RssAnon` and 9.7 GB `RssFile` are the CPU model's, not the CUDA expert cache's. It measures nothing about MEM10. The cell needs `--moe-cache-experts` (the elastic expert-slot mode, `docs/cuda-backend.md`).

## MEM10-G0 again, registered 2026-11-05 17:44 PST (the time from `date` at the run) before the reading

As above with **`--moe-cache-experts`** added to the server's flags, and the script now reads `decode path:` from the log. **A run whose log says the decode path is not `cuda` is void**, not a reading. One run, the exclusive timing lock, read and not graded.


### The reading, 2026-11-05 17:43:25 to 17:43:55 PST

`reading/` (`reading.json`, `serve.log`), `reading.log`. `serve` at `1822c6f09` (main), `--backend cuda --ctx 4096 --moe-cache-experts`, `~/models/gemma4-26b-q4_k_m.gguf` through its sidecar, the exclusive timing lock (free), load 1.14, 61.5 GB available. The log reads `decode path: cuda-resident (int4)`: a valid run. One run, read and not graded.

| `/proc/<pid>/status`, MiB | at listen (21 s) | 5 s later | after one 16-token completion | peak sampled |
|---|---|---|---|---|
| `RssAnon` | 188 | 191 | 220 | 320 |
| `RssFile` | **13,681** | 13,681 | 13,681 | 13,681 |
| `RssShmem` | **12,345** | 12,345 | 12,345 | 12,345 |
| `VmLck` | 0 | 0 | 0 | |
| `VmSwap` | 0 | 0 | 0 | 0 |
| `VmHWM` | 26,345 | 26,345 | 26,345 | 26,345 |
| device memory (`nvidia-smi`) | 6,942 | | | |

**What it says.** The 26B's sidecar is 13.6 GiB and it is mapped (`RssFile` 13.7 GB). The expert cache also holds **12.3 GB of `RssShmem`**: the host copy the experts are served from (the ledger's "M26 12.85 GB pinned host"; this is the same memory, now shown as shared-memory pages rather than heap, which is why `RssAnon` is only 188 MiB). So the experts are in host memory **twice**, as the file's pages and as that copy, and the process's `VmHWM` is **26.3 GB for a 13.6 GB model**. A pre-permuted target registered from the mapping would remove the second copy: up to 12.3 GB of host memory on this box, which is the ledger's saving, **and #103 did not remove it** (its row says "B36 (#103) covered the rest"; what it removed is the heap copy).

**What this reading does not say, and the build's risks:**
- Whether CUDA can register a file-backed mapping as page-locked and serve the expert slots' uploads from it at the same speed is not known. `upExperts`'s own comment records that a zero-copy direct read of a host stack was tried and mis-read by `gemv_w4a8_moe` at width; the registered-from-the-mapping plan uploads from the pages (a copy per slot fill, as today) and is a different question, but it is untested. A tier 0 probe (register a mapped file, upload slots from it, byte-compare with today's packing) is the first step and is small.
- `VmLck` reads 0, so the 12.3 GB is not counted as locked by the kernel; whether it is pinned by the driver in a way `/proc` does not show was not checked.
- The `.giw`'s `cuda` target still writes the canonical kind, so the sidecar's bytes are not in the layout the upload wants; a pre-permuted target changes the file format's cuda variant (version 17's "new encoding needs no new version" rule applies) and every cached cuda sidecar.
- This is one run on one box, 8 GB card, 62 GB RAM. On a smaller-RAM box the second copy decides whether the model loads at all.

**No build is started on this reading.** MEM10 stays open in the ledger with this figure; whether a 12.3 GB saving justifies an M–L change to the sidecar format is the coordinator's and the owner's call.
