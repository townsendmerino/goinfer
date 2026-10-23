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

All at `origin/main` d6214db6b plus this record's commits (no product code), 2026-10-23. Logs are under
`~/goinfer-logs/kernel-survey-2026-10/`, one per phase.

| id | reading | verdict |
|---|---|---|
| KS-A2 | graphs ÷ live **1.454× median** (1.414–1.626), **10 of 10 quads above 1**; live 2,874 µs a token (347.9 tok/s), graphs 1,955 µs (511.6 tok/s); 195 launches a token, so 14.7 µs a launch; `TestCUDA_launchCost` 12.92 µs on a 1×1 grid, 10.95 µs on 512 blocks; argmax ids identical across arms | **confirmed** |
| KS-B8 | attention at K = 3,900: 340.9 / 340.4 ms at 512 rows, 276.8 / 276.4 at 1,024, 218.9 / 218.9 at 2,048: **1.231× and 1.556×**, each pass within 0.2% of the other; whole prefill 933 → 840 → 748 ms (**1.25×** at 2,048) | **confirmed** |
| KS-A3 | `topk_select` is **one block** of 1,024 threads (33.5 KB static shared memory, 64 registers); SM throughput 1.3–1.7% of the device. **101.7 µs** median at (152k, 256), **211.5 µs** at (262k, 1,024) | **confirmed** |
| KS-B2 | `attn_fused_bm128_hd128`, the seventh pass (48 blocks): `long_scoreboard` + `lg_throttle` **59.7%** of stall cycles (barrier 0.7%); tensor pipe active **8.8%**; global loads **4.03 bytes** per thread per instruction (scalar); 180 registers, one block per SM, 25% of warps active; 2.75 ms a launch | **confirmed** |
| KS-A1 | goinfer's q4k plan reads **2,710.96 MB** a token (q4k native 1,302.33, int8 with scales 1,408.63) against the file's own **2,336.28 MB**: +373.7 MB, all of it the int8 copies of the Q5_K and Q6_K tensors | count matches the survey; stays `decision` (KS-O1) |

### KS-A2

Run 08:41–08:43. The first quad was voided six times while the 1-minute load came down from 19.6 (another session's loaders
had just released the lock); every quad that counted started at 7.39–7.44, under the 8.0 bar, with 57 GB available.

| quad | live µs/token | graphs µs/token | graphs ÷ live |
|---|---:|---:|---:|
| 0 | 3,381.7 | 2,079.3 | 1.6264 |
| 1 | 2,863.1 | 1,944.8 | 1.4722 |
| 2 | 2,873.3 | 1,938.1 | 1.4825 |
| 3 | 2,749.9 | 1,943.8 | 1.4147 |
| 4 | 2,873.0 | 1,944.2 | 1.4777 |
| 5 | 2,819.2 | 1,945.9 | 1.4488 |
| 6 | 2,809.0 | 1,941.8 | 1.4466 |
| 7 | 2,726.7 | 1,928.2 | 1.4141 |
| 8 | 2,807.4 | 1,937.8 | 1.4487 |
| 9 | 2,837.2 | 1,944.0 | 1.4595 |

- **What it says.** The graph arm is steady at 1.93–1.96 ms a token; the live arm moves 2.62–3.78 ms between blocks. 195 launches
  at 12.9 µs of host cost each is 2.5 ms, more than the GPU-bound token, so the live 0.5B is limited by issuing launches, not
  by the kernels. The survey's "GPU-bound near 2.0–2.1 ms" reads 1.95 ms.
- **Context switches** (instrument 3, whole process, 200 tokens): live 232,286 voluntary, **1,161 a token** (about 6 a
  launch), 399 involuntary; graphs 150,854 voluntary, 754 a token. The four busiest threads took 44,713 / 23,669 / 23,508 /
  23,012 on the live arm. Those are threads sleeping and waking a thousand times a token: the hand-offs between locked
  executor threads the survey inferred. A futex count by syscall was not available (no `strace` or `perf` on this box).
- **Caveat:** graphs are forced past the tenancy gate here (`GOINFER_CUDA_GRAPHS_UNSAFE`); this is the size of the launch
  cost, not a proposal to ship graphs. The survey's change is a cheaper launch path (raw launches from the caller's thread,
  pre-packed arguments), and graphs bound what it can buy: up to 1.45× on the 0.5B.

### KS-B8

Run 08:43–08:44, one process per width in the order 512, 1024, 2048, 2048, 1024, 512; 1-minute load 4.7–7.1 at each start,
58 GB available. Best of 3 profiled passes; "wall" is the warm, unprofiled pass.

| width | K | GEMV ms | attention ms | glue ms | wall ms |
|---:|---:|---:|---:|---:|---:|
| 512 | 2,048 | 272.8 / 272.9 | 98.0 / 97.9 | 45.2 / 45.0 | 432 / 431 |
| 1,024 | 2,048 | 254.0 / 254.7 | 70.7 / 70.8 | 37.5 / 38.9 | 393 / 385 |
| 2,048 | 2,048 | 244.0 / 244.1 | 52.3 / 51.9 | 34.6 / 34.7 | 355 / 352 |
| 512 | 3,900 | 528.7 / 528.9 | 340.9 / 340.4 | 89.5 / 88.7 | 933 / 932 |
| 1,024 | 3,900 | 493.5 / 492.7 | 276.8 / 276.4 | 74.9 / 75.7 | 840 / 838 |
| 2,048 | 3,900 | 461.7 / 461.7 | 218.9 / 218.9 | 66.7 / 67.0 | 748 / 748 |

(first / second pass of the order.)

- **Attention** at K = 3,900: 1.231× at 1,024 rows, **1.556× at 2,048** (the simulation said 1.16–1.26 and 1.37–1.47). At
  K = 2,048: 1.38× and 1.88×.
- **The rest moves too.** A wider pass reuses each weight over more rows: GEMV 528.7 → 461.7 ms (1.145×) and glue 89.5 → 66.7
  ms at K = 3,900, so the whole prefill gains **1.247×** at 2,048 rows (933 → 748 ms) and 1.11× at 1,024; 1.22× at K = 2,048.
- **Not measured here:** scratch and admission at 2,048 rows (the survey's 276 MiB on the 1.5B), and models other than the 1.5B.

### KS-A1

`TestKS_A1_planBytes`, a CPU load of the file at `--quant q4k` and `planWeight` over every projection and the head; the GGUF
tensor table read with the `gguf` Python reader. Per token, embeddings and norms excluded:

| tensors | file type | file MB | goinfer kind | goinfer MB | extra MB |
|---|---|---:|---|---:|---:|
| `attn_qkv` × 32 (split into q, k, v at load) | Q5_K | 622.85 | int8 | 907.14 | +284.29 |
| `ffn_down` × 16 | Q6_K | 330.30 | int8 | 402.85 | +72.55 |
| `output` | Q6_K | 80.80 | int8 | 98.63 | +17.83 |
| `ffn_down` × 16 | Q4_K | 226.49 | q4k | 226.49 | 0 |
| `ffn_up` × 32 (split into gate, up) | Q4_K | 905.97 | q4k | 905.96 | 0 |
| `attn_output` × 32 | Q4_K | 169.87 | q4k | 169.87 | 0 |
| **total** | | **2,336.28** | | **2,710.96** | **+374.7** |

The survey's 2,711 against 2,336 MB, and its +283 / +72 / +18 split, hold to the megabyte. The GPU A/B the row also names (a
Q5_K GEMV against `gemv_w8a8_g32`) waits on KS-O1.

### KS-A3

`TestKS_A3A5_decodeUnderNcu` under `ncu -k regex:^topk_select$` (run 08:47, timing lock exclusive), every launch: 8 on the 0.5B at
k = 256, then 8 on gemma3-1b at k = 1,024, each after 8 greedy tokens.

| shape | µs per launch (8 launches) | median |
|---|---|---:|
| vocab 151,936, k = 256 | 101.4, 128.3, 129.2, 126.5, 101.9, 101.0, 101.1, 101.1 | 101.7 |
| vocab 262,144, k = 1,024 | 152.3, 151.9, 152.1, 215.0, 211.3, 214.3, 211.6, 211.9 | 211.5 |

The grid is a single block in every launch and DRAM throughput stays under 2.5% of peak (the row is L2-resident after the head),
so one SM does all of the work. The survey's 102 µs reads 101.7 at the 0.5B shape; at gemma3's it is 152–215 µs, above the
survey's +125 µs preset cost. Each shape shows two levels (about 101 and 127 µs; 152 and 213 µs); this read does not say why.

### KS-B2

`TestKS_B2_prefillUnderNcu` under `ncu -k regex:attn_fused_bm128_hd128 --launch-skip 168 -c 4` (run 08:48, timing lock
exclusive): one 3,900-token `PrefillLast` on the 1.5B loaded as serve loads it, 512-row passes; the four profiled launches
are layers 0–3 of the seventh pass (grid 48 = 4 row tiles × 12 heads, as registered).

| metric | layer 0 | 1 | 2 | 3 |
|---|---:|---:|---:|---:|
| duration, ms | 2.748 | 2.761 | 2.740 | 2.763 |
| `long_scoreboard`, % of stall cycles | 59.05 | 59.18 | 59.1 | 59.0 |
| `lg_throttle`, % | 0.63 | 0.63 | 0.6 | 0.6 |
| barrier, % | 0.70 | 0.72 | 0.7 | 0.7 |
| tensor pipe active, % of peak | 8.83 | 8.77 | 8.82 | 8.78 |
| warps active, % of peak | 25.0 | 25.0 | 25.0 | 24.8 |

The rest of the stall cycles (layer 0): `wait` 13.4%, `selected` 8.2, `math_pipe_throttle` 7.0, `branch_resolving` 4.3,
`short_scoreboard` 3.1. Global loads: 1,347,456 warp instructions for 173.66 MB, **4.03 bytes per thread per load**: every
staging load is a scalar 32-bit word. Registers (180) limit the kernel to one 256-thread block per SM.

- **The finding holds after KV4a.** KV4a's paged reads changed how a tile finds its keys (a block-table lookup per tile),
  not how it stages them: the warps wait on global loads for 59% of their stalled cycles while the tensor pipe idles 91% of
  the time.
- **This load holds f32 KV.** The serve-shaped 1.5B load reports `kvF16 false`, so this is the f32 twin. KV4b (#157, f16 KV per
  group) was still open at this read; if it merges, the `_kv16` twin (2-byte elements) is the one to read next, with the same
  instrument.
