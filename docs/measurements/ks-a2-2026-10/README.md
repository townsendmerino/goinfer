# KS-A2 build: launches issued from the resident's own thread, 2026-10-23

The build for `docs/tasks/task-kernel-survey-2026-10.md` KS-A2, whose first measurement
(`docs/measurements/kernel-survey-2026-10/README.md` § KS-A2) put the 0.5B's live decode at 1.454× slower than graph replay,
with 12.9 µs of host cost a launch and 1,161 voluntary context switches a token.

**The change.**
- aikit `gpu`: `Queue.LaunchDirect` issues one `cuLaunchKernel` from the calling thread, instead of handing each launch to
  gocudrv's executor thread and waiting for it. Arguments are packed into pooled storage, with no per-launch allocation.
  `Device.BindCurrentThread` makes the context current on the caller's locked thread, once.
- A `Pipeline` now carries its `CUfunction`. gocudrv exports no accessor, so it is read by reflection at pipeline creation.
  It is the same handle, so a function attribute set through gocudrv (the dynamic shared-memory opt-in) still applies.
- goinfer `cuda`: `cudaResident.launch` calls `LaunchDirect`, and the executor goroutine binds the context at setup.
  Kernels, arguments and launch order are unchanged, so output should be byte-identical.

Machine: nobara-pc, RTX 2070 SUPER 8 GB (Turing sm_75), Ryzen 7 3700X (16 threads), NVIDIA driver 595.91.07, Go 1.27.3.
Checkpoints from `~/models` on NVMe.

## Pre-registration

Registered 2026-10-23 before any timed run. Instrument: `docs/measurements/ks-a2-2026-10/instruments/ks_a2_direct_test.go.txt`,
copied into `cuda/` as `ks_a2_scratch_test.go` for the run and removed afterwards. The run holds the timing lock
(`timing_lock.py run --label cuda-builds-ks-a2`).

- **Tier 1, in process.** One resident per model, loaded as `serve -backend cuda -quant int4` loads it, with graphs off.
  Arm A sends every launch through gocudrv's executor (`launchViaExecutor`, the old path). Arm B launches directly (the
  build). The arms are the same resident with one field toggled between blocks. A block is 64 greedy forwards over
  positions 128–191, each fed the same fixed embeddings.
- **Models:** Qwen2.5-Coder-0.5B Q4_K_M decides. gemma3-1b Q4_K_M is reported beside it and decides nothing.
- **G1, bit-identity:** over 64 positions at depth 128, the full logits from arm A and arm B are byte-equal. A single
  differing logit stops the item. Tier 0 alongside: `TestLaunchDirect_bitIdentical` (committed) holds byte-equal logits
  over 12 positions on `llama-tiny` and `qwen3moe-tiny`, and holds allocations a token on the direct path at or below the
  executor path's.
- **G2, speed:** 10 ABBA quads. Readout: direct speedup per quad, (A1 + A2) / (B1 + B2), its median, and how many quads are
  above 1.
  - **ship:** at least 9 of 10 quads above 1 (CLAUDE.md's small-gains rule: bit-identical, 9 of 10 pairs, no memory
    increase).
  - **parked, to the owner:** 6–8 of 10.
  - **killed:** 5 or fewer of 10.

  The survey bounds the median: graph replay was 1.454× the live path on this model, so a direct-launch median above that
  would be a measurement fault to explain, not a win.
- **Memory:** the change allocates no device memory (no buffer is added), and the host side is held by G1's allocation
  count. Recorded, no bar: free VRAM after load.
- **Recorded, no bar:** launches a token; voluntary context switches a token on each arm over 200 tokens; host cost a launch
  on a 1×1 grid on each arm, 20,000 launches.
- **Load rule:** every quad records the 1-minute load and `MemAvailable` at its start. A quad started with the 1-minute load
  above 8.0 is void and re-run in order, and the record lists it.
- **Stopping rule:** one run. **Cost:** two loads (~20 s) and per model 10 quads × 4 blocks × 64 tokens at ~3–6 ms, plus
  the G1 and context-switch passes: about 2 minutes in all.
- **Then:** the CUDA device suite (`scripts/cuda_device_suite.py`) on the touched tests, and the graph tests, since graph
  capture now records launches issued from a thread other than the one that began the capture.

## Readings

(not yet run)
