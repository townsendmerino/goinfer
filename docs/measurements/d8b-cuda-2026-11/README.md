# D8b-cuda: a q4k checkpoint on CUDA, loaded directly and through its `.giw` sidecar (2026-11, nobara-sb01)

The ledger row (`docs/tasks/task-memory-campaign-2026-10.md`, D8b-cuda; P36 in `docs/queue-performance.md`): D8b-build gave a native Q4_K matrix a `.giw` tensor kind (8), aliased from the mapping, and routed a q4k `.gguf`
to its sidecar by default (`docs/measurements/d8b-2026-10/README.md`; the Mac's cell D8B-G4 read the CPU heap 583.6 → 1.7 MiB). This is the CUDA cell, "not run here" there: does the CUDA resident load from the aliased
sidecar, hold the host heap down the same way, and decode the same tokens.

## The cell, registered before it runs

- **Instrument:** `d8b_cuda_cell_test.go.txt` here (copied into `backend/cuda` to build, not committed there: it is a measurement, and a test naming this checkpoint is filed by `cmd/gate`'s heavy partition). The test binary is built from main
  (`be595d7fe`, which holds D8b-build and #689); it loads `~/models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf` at `Quant: "q4k"`, `Backend: "cuda"` (local NVMe) twice, in separate processes:
  `D8B_ARM=direct` (`DirectLoad: true`: the source `.gguf`'s weights built in the heap) and `D8B_ARM=sidecar` (the default: the `.giw` sidecar, `~/models/qwen2.5-coder-0.5b-instruct-q4_k_m.q4k.e4h.cuda.v17.giw`, already built on 2026-11-03).
  After the load: `runtime.GC` and `debug.FreeOSMemory`, then Go heap in use, `RssAnon` and `RssFile`; then 16 greedy tokens through the resident after the first four of the checkpoint's seed.
- **Design:** one run each, direct first, the exclusive timing lock. The Mac's cell (D8B-G4) was one run each.
- **The decision, D8B-G4's bar carried over:**

| | |
|---|---|
| **PASS** | both arms go CUDA-resident (not declined); the 16 tokens are identical; the sidecar arm's Go heap in use is at most **0.25 ×** the direct arm's |
| **PARKED** | the tokens identical and both resident, the heap ratio above 0.25 and at most 0.5 |
| **FAIL** | a different token, either arm declined, or a ratio above 0.5 |

- **Reported, not graded:** `RssAnon` and `RssFile` (the aliased weights are mapped file pages, so `RssFile` rises where the heap falls), and the sidecar's load time.
- **Void:** the one-minute load above 8.0 at a run's start; a refused load is VOID for its arm. **Stopping rule:** one run each.
- **Cost:** about 1 minute a run. Start and finish from `date` at the run.
- **Void if:** a run outside the lock, a NOT RUN counted as a pass, a bar moved after a reading.
