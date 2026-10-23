# Kernel survey first measurements on nobara: KS-A2, KS-B8, KS-A3, KS-A5, KS-B2, KS-A1 (tier 0), 2026-10-23

Items from `docs/tasks/task-kernel-survey-2026-10.md`. Each is its ledger row's first measurement, and none changes product code.
Machine: nobara-pc, RTX 2070 SUPER 8 GB (40 SMs, Turing sm_75), Ryzen 7 3700X (16 threads), 62 GB, NVIDIA driver 595.91.07,
Go 1.27.2. Every checkpoint is read from `~/models` on NVMe.

**Instruments.** `docs/measurements/kernel-survey-2026-10/instruments/ks_survey_test.go.txt` holds the test-side drivers, and
`docs/measurements/kernel-survey-2026-10/instruments/ks.sh` the phase runner. For a run it is copied into `cuda/` as
`ks_survey_scratch_test.go` and removed afterwards; it is never committed under `cuda/`. Each test is gated on `GOINFER_KS=1`.
`TestPrefillDecomp` and `TestCUDA_launchCost` are the tree's own tests, run unmodified. ncu is Nsight Compute 2026.1.1
(`/opt/nvidia/nsight-compute/2026.1.1/ncu`), profiling open to users on this box (`RmProfilingAdminOnly: 0`).

## Pre-registration

Registered 2026-10-23 before any run below. Tier, instrument, readout, decision bands, stopping rule and cost for each.
The timed runs (A2, B8) and the ncu reads hold the timing lock exclusively (ncu's durations are replayed kernel times, not
the served path's wall clock, but its counters must not share the card); the build and the tier-0 count hold it shared. Every timed cell records the 1-minute load and `MemAvailable` at its start; a cell started with the
1-minute load above 8.0 is VOID and re-run in the same order, and the record lists it.

### KS-A2 · is small-model decode bound by launch cost? (timed, tier 1)

- **Instrument 1:** `TestCUDA_launchCost` (gemma4-moe-tiny): host cost per launch on a 1×1 grid and a 512-block grid,
  20,000 launches each, today's pins.
- **Instrument 2:** `TestKS_A2_graphsAB`. Qwen2.5-Coder-0.5B Q4_K_M, loaded as `serve -backend cuda -quant int4` loads it,
  twice in one process: arm A live launches, arm B CUDA graph replay (`GOINFER_CUDA_GRAPHS=1` plus
  `GOINFER_CUDA_GRAPHS_UNSAFE=1`, the only way to enable graphs on this DEFAULT-compute-mode box; its startup self-test
  still runs). Both arms warm 128 positions, run one untimed block, then 10 quads of A B B A. A block is 64 greedy forwards
  over positions 128–191, each fed the same fixed embeddings, so every block is the same work at depth 128. Readout: the
  graphs speedup per quad, (A1 + A2) / (B1 + B2); the median over 10 quads; how many quads are above 1. Also recorded:
  launches per token on the live arm, and whether the two arms' argmax ids agree.
- **Instrument 3 (no bar):** voluntary and involuntary context switches over 200 tokens on each arm, summed over the
  process's threads from `/proc/self/task/*/status`, with the four threads that sleep most. This box has no `strace` or
  `perf`, so a voluntary context switch stands in for the brief's futex count: a futex wait that sleeps is one.
- **Bands (on the median graphs speedup and the 1×1 launch cost):**
  - **confirmed:** graphs ≥ 1.10× with at least 8 of 10 quads above 1. Launch cost is then a material share of a 0.5B token,
    whatever the per-launch figure.
  - **killed:** graphs ≤ 1.03× **and** a 1×1 launch ≤ 6 µs (the survey's own kill rule).
  - **ambiguous (parked, to the owner):** anything else, including graphs ≤ 1.03× with a launch above 6 µs (the host cost is
    real but overlaps GPU time).
- **Stopping rule:** one run of 10 quads. **Cost:** two 0.5B loads (~20 s), 42 blocks × 64 tokens at ~3 ms, 400 tokens for
  instrument 3: about 1 minute.

### KS-B8 · wider prefill passes (timed, tier 1, no code)

- **Instrument:** `TestPrefillDecomp` on Qwen2.5-Coder-1.5B Q4_K_M at `GOINFER_DECOMP_K=2048,3900`, with the pass width set by
  `GOINFER_PREFILL_CHUNK` (read once at Load) to 512, 1024 and 2048, one process per width, in the order
  512, 1024, 2048, 2048, 1024, 512. Each process reports, per depth, the unprofiled wall of a warm pass and the best of 3
  profiled passes split into GEMV / attention / glue.
- **Readout:** attention speedup at width w = attention ms at 512 ÷ attention ms at w, per pass of the order and per depth;
  the wall ratio the same way. The survey's simulation: 1.16–1.26× at 1,024 rows and 1.37–1.47× at 2,048 (1.5B, attention only).
- **Bands (attention, K = 3,900, the mean of the two passes; each pass must agree in direction):**
  - **confirmed:** 2,048 rows ≥ 1.25×, with each pass ≥ 1.15×.
  - **killed:** ≤ 1.05× at both 1,024 and 2,048.
  - **ambiguous:** anything else.
  K = 2,048 is reported but decides nothing. The wall ratio is reported beside it, since a wider pass also reuses each weight
  over more rows in the GEMM.
- **Stopping rule:** the six processes once. **Cost:** six 1.5B loads and 2 depths × 4 passes each: about 3–4 minutes.

### KS-A3 · `topk_select` (ncu, tier 0 kernel read)

- **Instrument:** `TestKS_A3A5_decodeUnderNcu` under `ncu -k regex:'^(topk_select|fused_rms_gu)'`, sections SpeedOfLight,
  LaunchStats, Occupancy and MemoryWorkloadAnalysis, default cache control. The 0.5B decodes 8 tokens through
  `ForwardTopK` at k = 256 (vocab 151,936); gemma3-1b decodes 8 at k = 1,024 (vocab 262,144). Readout: `gpu__time_duration`
  (median of 8) and the grid size at each shape. The survey: 102 µs a token on one SM, against the argmax's 20 µs.
- **Bands:** **confirmed** if (152k, 256) takes ≥ 60 µs on a single-block grid; **killed** if both shapes take ≤ 30 µs (under
  about 1% of a 0.5B token); **ambiguous** otherwise.

### KS-A5 · `fused_rms_gu` at (896, 4864) (ncu, tier 0 kernel read)

- **Instrument:** the same ncu run, the 0.5B's gate/up launches (hidden 896, inter 4,864; `fused_rms_gu` or its rows-per-warp
  twin, whichever the build dispatches; the record names it). Readout: `dram__throughput.avg.pct_of_peak_sustained_elapsed`
  and `gpu__time_duration`, median over the profiled launches.
- **Bands:** **killed** at DRAM ≥ 80% of peak (the survey's rule); **confirmed** (headroom) at ≤ 65%; **ambiguous** between.

### KS-B2 · `attn_fused_bm128_hd128` staging share (ncu, tier 0 kernel read)

- **Instrument:** `TestKS_B2_prefillUnderNcu`: one 3,900-token `PrefillLast` on the 1.5B loaded as serve loads it, under
  `ncu -k regex:attn_fused_bm128_hd128`, skipping 168 launches to the seventh pass (rows 3,072–3,583, the deepest full 512-row pass: 48 blocks on 40 SMs; the
  grid size in LaunchStats confirms the pass) and profiling its first four layers, sections
  WarpStateStats, SchedulerStats, SpeedOfLight, Occupancy, LaunchStats, ComputeWorkloadAnalysis, MemoryWorkloadAnalysis, plus
  `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active`, `smsp__sass_inst_executed_op_global_ld.sum` and
  `l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum`. Readout: the share of warp stall cycles per issued instruction spent in
  `long_scoreboard` + `lg_throttle` (waiting on, or throttled issuing, global loads: the staging) out of all stall reasons;
  the tensor pipe's active share; and global-load bytes per thread per load instruction (the survey: scalar loads only).
- **Bands:** **confirmed** (staging-bound) if `long_scoreboard` + `lg_throttle` ≥ 40% of stall cycles and the tensor pipe is
  active ≤ 20% of peak; **killed** if `long_scoreboard` + `lg_throttle` ≤ 15%; **ambiguous** otherwise. The barrier share is
  reported beside it and decides nothing (warps waiting at a barrier for another warp's staging are staging-bound too, but so
  are warps waiting out an MMA phase).
- **Read at** today's `origin/main` (d6214db6b), after KV4a's paged reads (`attn_fused` now stages a key tile from at most two
  blocks' first rows). If KV4b (#157) merges before this read, it is read on that main as well.

### KS-A1 · plan bytes per token for phi3-mini (tier 0 count, no GPU A/B)

- **Instrument:** the GGUF tensor table of `~/models/phi3-mini-4k-gguf/Phi-3-mini-4k-instruct-q4.gguf`, and the CUDA upload
  plan (`planWeight`) of every projection and the head of that file loaded at `--quant q4k`: device bytes by kind (q4k native,
  int8 + scales) and by source GGUF type. Readout: goinfer's weight bytes read per decode token against the file's own
  (llama.cpp's), embeddings and norms excluded. The survey: 2,711 against 2,336 MB.
- **No bar:** a count; it informs the owner's decision KS-O1 and the row stays `decision`.

### Item 0 · MEM11-int8-MLA-T3 (not part of the survey)

`TestDeepseekV2LiteReal_gate` and `TestDeepseekMoonlightReal_gate` at int8int8 against the HF bf16 goldens: the bars are
I8M-G5's, already registered in `docs/tasks/task-memory-campaign-2026-10.md` (argmax, greedy continuation, cosine ≥ 0.99). The
reading goes in `docs/measurements/mem11-int8-2026-10/mla.md`.

## Readings

Not yet run.
