# MEM10: the 26B's host footprint on CUDA today (2026-11, nobara-sb01)

MEM10 (`docs/tasks/task-memory-campaign-2026-10.md`, P36 in `docs/queue-performance.md`): "a pre-permuted `cuda` target registered from the mapping; M–L". Its row says B36 (#103) covered the rest, and §1.1's problem text (the whole expert
stack in a fresh heap copy registered as pinned host memory, M26 12.85 GB) predates #95 and #103. Before an M–L build, a reading of what is left.

## MEM10-G0, a reading, registered 2026-11-05 (time from `date` at the run) before it ran

Read, not graded; it sets the size of what a build would save, or says there is nothing left.

- **Instrument:** `mem10_g0.py` here: `serve` (built from main, `CGO_ENABLED=0 go build -tags cuda ./backend/cuda/cmd/serve`) at its defaults except `--backend cuda --ctx 4096`, `~/models/gemma4-26b-q4_k_m.gguf` (local NVMe, through its `.giw` sidecar `gemma4-26b-q4_k_m.int4.e4h.cuda.v17.giw`), the exclusive timing lock. `/proc/<pid>/status` (`RssAnon`, `RssFile`, `RssShmem`, `VmLck`, `VmSwap`, `VmHWM`) when the server listens, 5 s after, after one 16-token greedy completion, and the peak sampled every 250 ms; device memory from `nvidia-smi`.
- **What it answers:** how much anonymous or pinned host memory the 26B holds beyond the mapped weights, which is what the ledger's "registered from the mapping" would remove (the pinned copy shows as `RssShmem`/`VmLck`, the heap copy as `RssAnon`).
- **Void:** the one-minute load above 8.0 at the start, or the swap tripwire closing the load. **One run.** Cost about 2 minutes.
- **Nothing is decided on this reading:** whether a build is worth its size is the coordinator's and the owner's call; this README adds the figures and the arithmetic of what could be saved.
