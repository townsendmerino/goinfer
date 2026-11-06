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

## The first attempt was void for its sidecar arm (2026-11-05 16:57 PST)

`first-attempt-void/`. Both arms read 585 MiB of Go heap in use and the same 16 tokens, resident. That is not a result: a test binary loads a `.gguf` **directly** whatever `DirectLoad` says (`decoder/sidecar.go`'s `testing.Testing() && !sidecarInTests`, and `sidecarInTests` is unexported), so the "sidecar" arm was a second direct load. The instrument could not run the path the cell is about, and it is replaced; this attempt decides nothing.

## The cell, registered again 2026-11-05 16:58 PST before the reading, on the served path (the time from `date`; the commit before this one guessed 17:03)

- **Instrument:** `d8b_cuda_served.py` here: `serve` (built at `67bed4dbe`, which is main `be595d7fe` plus this branch's docs, `CGO_ENABLED=0 go build -tags cuda ./backend/cuda/cmd/serve`; not a test binary), `--backend cuda --quant q4k --ctx 4096`, the same checkpoint, one server per arm, direct first: `-direct-load`, then the default (the sidecar `~/models/qwen2.5-coder-0.5b-instruct-q4_k_m.q4k.e4h.cuda.v17.giw`, built 2026-11-03). Per arm: `/proc/<pid>/status` (`RssAnon`, `RssFile`, `RssShmem`, `VmSwap`) two seconds after the server listens; then one greedy completion of 16 tokens on a fixed prompt, and the process's device memory from `nvidia-smi`.
- **What replaces "Go heap in use":** `RssAnon` after load. The server does not expose its heap; MEM0's table reads the same (`docs/measurements/mem0-2026-10/README.md`). A CUDA context's own anonymous memory is a floor under the sidecar arm (MEM0 read 84 to 140 MiB after load), so the ratio is judged against the direct arm's figure.
- **Decision (D8B-G4's bar carried over to `RssAnon`):**

| | |
|---|---|
| **PASS** | both servers start and answer on the CUDA resident; the two 16-token completions are equal; the sidecar arm's `RssAnon` after load is at most **0.25 ×** the direct arm's |
| **PARKED** | the completions equal, the ratio above 0.25 and at most 0.5 |
| **FAIL** | different text, a server that did not start, or a ratio above 0.5 |

- **Reported, not graded:** `RssFile` and `RssShmem` (the aliased weights are mapped file pages), the load times, device memory.
- **Void:** the one-minute load above 8.0 at a server's start. **Stopping rule:** one run each, direct first. **Cost:** about 1 minute. **Void if:** a run outside the lock, a NOT RUN counted as a pass, a bar moved after a reading.


## The reading, 2026-11-05 16:58:44 PST: PASS

`served/` (`run.log`, `readings.json`, `serve-direct.log`, `serve-sidecar.log`). The exclusive timing lock (free), one-minute load 0.62 and 0.57, no void arm, one run each, direct first.

| | `-direct-load` | default (the sidecar) | bar |
|---|---|---|---|
| `RssAnon` after load | **656 MiB** | **61 MiB** | sidecar at most 0.25 × direct |
| ratio | | **0.093** | PASS |
| `RssFile` | 102 MiB | 554 MiB | reported (the aliased weights are mapped file pages) |
| `RssShmem` | 94 MiB | 94 MiB | reported |
| `VmSwap` | 0 | 0 | |
| device memory | 966 MiB | 990 MiB | reported |
| load time | 1.2 s | 1.2 s | reported |
| the 16-token completion | `numbers []int) int {\n    sum := 0\n    for _, number` | the same | equal |

Both servers started on the CUDA resident and answered. The sidecar arm holds 595 MiB less anonymous memory; the 0.5B checkpoint's Q4_K bytes are the difference (the sidecar's tensors are kind 8, aliased from the mapping, and the CUDA upload streams them from it). The sidecar arm's `RssAnon` of 61 MiB is under MEM0's 84 to 140 MiB after-load figures for the larger models, which had a different load path; it is a floor of this binary and this model, not a model-independent number. **Device memory is 24 MiB higher on the sidecar arm** (990 against 966): not graded, not explained here; one run each does not say whether it is noise.

**Not covered:** other q4k checkpoints and sizes (the 0.5B only; the Mac's cell used the same file), a Q5_K/Q6_K mix (the kquant lanes upload their own forms), speculative or multi-slot loads, and the first-load transcode cost (the sidecar already existed). Gemma 4 and the MoE families at q4k were not run.

**Verdict: PASS.** The ledger row D8b-cuda closes with this change (`docs/tasks/task-memory-campaign-2026-10.md`), and P36 in `docs/queue-performance.md` drops it.
