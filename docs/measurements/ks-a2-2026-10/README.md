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

Run 2026-10-23 11:45–11:47 PDT at goinfer 17a22510c with aikit gpu 9b938a5 (via `go.work`), under the timing lock. Logs:
`~/goinfer-logs/ks-a2-2026-10/g1g2-run1.log` (G1, G2) and `nobar-run2.log` (the no-bar counts).

| id | reading | verdict |
|---|---|---|
| G1 | 0.5B: 64 positions × 151,936 logits, **0 differ**; gemma3-1b: 64 × 262,144, **0 differ** (both runs). `TestLaunchDirect_bitIdentical`: 12 positions byte-equal on `llama-tiny` and `qwen3moe-tiny`; allocations a token 139 → 104 and 227 → 164 | **PASS** |
| G2 | 0.5B: direct speedup **median 1.4918×** (1.4620–1.6935), **10 of 10** quads above 1; 2,896.5 → 1,919.3 µs a token (345.2 → 521.0 tok/s). gemma3-1b (no bar): median 1.4647× (1.4305–1.6021), 10 of 10; 5,083.0 → 3,440.4 µs (196.7 → 290.7 tok/s). Argmax ids identical across arms | **SHIP** |
| memory | no device buffer added. Free VRAM after load: 5,101 MiB (0.5B) and 5,839 MiB (gemma3-1b) of 7,784, one resident shared by both arms. Host allocations a token are lower (G1) | no increase |

**Void quads.** Three were voided and re-run in order: 0.5B quad 0 twice (1-minute load 10.98, then 9.29) and gemma3-1b quad 0 once
(8.30). The load was btrfs writeback from another session. Every counted quad started at 6.17–7.86, with 56–57 GB available.

**The median is above the bound I registered for it, and the bound was wrong.** The pre-registration read the survey's graph
replay (1.454×) as the most a cheaper launch could buy. But graph replay does not remove launches: the graphs capture only
segments A, B and C, and rope_kv and attention still run live in the gap, through the executor. That comment is at
`cuda/resident.go:cudaResident.KVSlots`. So the graph arm still made about five executor hand-offs a layer, and the direct arm
makes none. The 0.5B's direct token, 1,919 µs, is 36 µs under the survey's graph token (1,955 µs, another session). The
mechanism explains that, not a timing fault.

**No-bar counts** (`nobar-run2.log`; its two quads a model are not part of G2):

| | 0.5B executor | 0.5B direct | gemma3-1b executor | gemma3-1b direct |
|---|---:|---:|---:|---:|
| launches a token | 195 | 195 | 341 | 341 |
| voluntary context switches a token | 1,197.7 | 328.8 | 1,893.7 | 339.0 |
| host cost a 1×1 launch (20,000 launches) | 10.42 µs | 2.26 µs | 11.19 µs | 2.28 µs |

**The first run's no-bar step failed in the instrument, not the build.** It launched `scale_vec`, which only Gemma 4 compiles,
so it hit a nil pipeline (the sticky error named `gemvQ4K`, the first zero `Pipeline` field `pipeName` found). The instrument
now launches `residual` over 0 elements. The G1 and G2 readings come before that step, and every timed token resets and
returns the launch error. They stand, and were not re-measured.

**Driver behavior found on the way.** On a queue with its own stream, the driver accepts a launch from a thread with no context
current, and runs it in the stream's context. Only the null stream refuses one (`CUDA_ERROR_INVALID_CONTEXT`). goinfer binds its
executor thread anyway (`Device.BindCurrentThread`), and aikit's test asserts the null-stream refusal.

**CUDA device suite** (`scripts/cuda_device_suite.py`, the whole `cuda` module: the change is on the launch every resident test
goes through), 12:03–12:13 PDT at 17a22510c with aikit gpu 2049dc7, under the timing lock: **GREEN, 772 passed, 0 failed, 0 device
skips, 236 other skips**. `TestLaunchDirect_bitIdentical`, `TestGraphsSafeGate`, `TestCUDAKVSlots_graphs` and the four
`TestCUDA_graph*` pass. Log: `~/goinfer-logs/ks-a2-2026-10/device-suite/`. The Gemma 4 graph bit-exact gates skip unless graphs are
forced (`GOINFER_CUDA_GRAPHS_UNSAFE=1`), so they are run forced on their own (below).

**Graph gates, capture forced** (`GOINFER_CUDA_GRAPHS_UNSAFE=1`; `-run 'TestGemma4Graphs_|TestGraphsSafeGate|TestCUDA_graph|TestCUDAKVSlots_graphs'`),
12:31 PDT at 388fe3d43 (this branch merged with main at 7b0ad9418) with aikit gpu 2049dc7, under the timing lock: **14 of 14 pass**,
including all seven `TestGemma4Graphs_bitExact_*` / `locateDivergence` / `sameModelUnderLoad` gates. Capture now records launches
issued from the resident's thread while gocudrv's executor began the capture, and replay stays bit-exact to live. Log:
`~/goinfer-logs/ks-a2-2026-10/graphs-unsafe-full.log`.
