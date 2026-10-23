# Task: kernel survey, speed and memory candidates in goinfer and aikit (2026-10)

> **Status: SURVEY. Nothing in it is built or measured.** Filed 2026-10-22 at the owner's request: read every kernel in both
> repos for speedups and memory savings.
>
> - **Read at** goinfer `5e98ca9` and aikit main at v1.72.0, the version goinfer requires since c446a7b2 (aikit's `linalg/`
>   and `embed/` are unchanged from v1.71.0). Re-checked against goinfer `0af51451`: the commits since (MEM7, KV8a on Metal,
>   MEM4(b)) change none of the findings. MEM7 sizes the Metal prefill block to the memory guard's budget but leaves KS-H1's
>   exclusions as they were.
> - **How:** eight static reviews in parallel, one per area: CUDA decode, CUDA prefill and families, Metal decode, Metal
>   prefill and families, WebGPU, CPU, aikit retrieval and towers, and memory across backends. Each review searched the
>   docs for prior attempts before listing a candidate. Each review's top claims were then re-checked against the source
>   and, for CUDA, against the shipped PTX in `cuda/testdata/`.
> - **What a number here is:** an estimate from arithmetic, labeled with its basis. Every item names the cheapest first
>   measurement that confirms or kills it. Nothing here is a claim until that measurement reads, and CLAUDE.md's
>   measurement rules apply unchanged (tiers, timed runs by day through the timing lock, pre-registration).

<!-- ledger -->
| id | state | next | evidence |
|---|---|---|---|
| KS-A1 | decision | KS-O1, then nobara: plan bytes per token (tier 0), a Q5_K GEMV against `gemv_w8a8_g32` | — |
| KS-A2 | open | nobara: `TestCUDA_launchCost`, the 0.5B graphs A/B, a futex count (minutes) | — |
| KS-A3 | open | nobara: ncu of `topk_select` at (152k, 256) and (262k, 1024) | — |
| KS-A4 | open | nobara: launch counter (tier 0), then an in-process A/B on gemma3-1b | — |
| KS-A5 | open | nobara: ncu DRAM percent of `fused_rms_gu` at (896, 4864) | — |
| KS-A6 | open | nobara: host timestamps around upload, readback and hand-off | — |
| KS-A7 | open | nobara: launch counter, then an A/B on phi3-mini | — |
| KS-A8 | open | nobara: an A/B after A2 reads | — |
| KS-B1 | open | nobara: ncu on the gate and k launches at M=512; the int-to-float swap alone, bit-gated | — |
| KS-B2 | open | nobara: ncu staging share of `attn_fused_bm128_hd128`, then the attention A/B | — |
| KS-B3 | open | nobara: MoE category and launch count on M26 at M=512 | — |
| KS-B4 | open | nobara: attention/GEMV/recurrence split on Qwen3.5-9B at 8k and 16k | — |
| KS-B5 | open | nobara: MoE category and cache misses on M35 and gpt-oss-20b | — |
| KS-B6 | open | nobara: served TTFT for phi3-mini at q4k against Ollama | — |
| KS-B7 | open | nobara: a prototype against `gemv_w4a8_rn` at the 1.5B gate shape | — |
| KS-B8 | open | nobara: `TestPrefillDecomp` at pass widths 512/1024/2048 (no code) | — |
| KS-B9 | open | nobara: launch and grid counters, then the GEMV-category A/B | — |
| KS-B10 | open | nobara: sync and launch counters in the generic expert-major Phase 1 | — |
| KS-B11 | open | nobara: `PrefillScratchBytes` delta, then a glue+GEMM A/B | — |
| KS-B12 | open | nobara: per-token TTFT on Nemotron-H as the baseline | — |
| KS-B13 | open | aikit: a microbenchmark at SigLIP shapes, bit-gated | — |
| KS-B14 | open | nobara: host round trips per image (55 → 0), then a tower A/B | — |
| KS-B15 | decision | KS-O2 | — |
| KS-C1 | open | Mac: time `gemv_q4k_f32` and `gemv_w8_f32` at Phi-3 shapes (tier 0) | — |
| KS-C2 | open | Mac: a masked down-projection prototype, byte-equal at both shapes | — |
| KS-C3 | open | Mac: the int8 GEMVs in `TestMetalGEMVS0` at the 1.5B shapes | — |
| KS-C4 | open | Mac: P33's decomposition split by pipeline | — |
| KS-C5 | open | Mac: the decomposition's attention arm at 128 and 512 keys | — |
| KS-D1 | open | Mac: E2B at 4,096 tokens with and without the attention dispatch | — |
| KS-D2 | open | Mac: `TestMetalPrefillDecomp` on the 0.5B at K=512 and 3900 | — |
| KS-D3 | done | done 2026-10-22: bit-identical (D3-G1, G2 PASS); `delta_rule_seq` at M=512 27.2 → 19.7 ms GPU, median 0.7245, 10 of 10 pairs (D3-G3 SHIP, run at night on the owner's word); `delta_conv_seq` 0.53 of its time | `docs/measurements/ksd3-2026-10/README.md` |
| KS-D4 | open | Mac: the S17 whole-tower A/B on the Gemma 4 tower, plus its gate | — |
| KS-D5 | open (measured 2026-10-22) | next: SwiGLU fused into the gate\|up GEMM, behind a tiny-fixture byte compare. Measured: one 3,900-row pass on the 1.5B counts 261.9 MB of scratch, `guF` 139.9 MB (53.4%) | `docs/measurements/ks-night-2026-10/README.md` |
| KS-D6 | done | done 2026-10-23: the grid towers allocate the unfused attention's scratch and `f2` only where read, and stage `pt` in `f1`; Qwen3-VL-2B at 4K holds 2,622.5 → 1,515.4 MiB; outputs byte-equal on MEM2-G1's seven towers (D6-G1, G2 PASS) | `docs/measurements/ks-night-2026-10/README.md` |
| KS-D7 | measured (KILL) | closed 2026-10-23: the half2 dequant is bit-identical (all 1,048,576 cases; the 1.5B's 3,900-token prefill byte-equal) and makes that prefill 1.1% faster (median 0.9888, 10 of 10 pairs), under the 2% kill line. The first read (0.9983) was an A/A, withdrawn: its switch never reached the prefill library. Not merged | `docs/measurements/ks-night-2026-10/README.md` |
| KS-D8 | open | Mac: sequential prefill tok/s on Nemotron-Nano-9B-v2 at K=512 | — |
| KS-D9 | open | Mac: `TestG4LayerMajor_M26AB` driving the generic pager | — |
| KS-D10 | open | aikit: GPU timestamps in `TestMetalANNCrossover` (no consumer today) | — |
| KS-E1 | open | either box: `TestW4A8_parity_and_bandwidth` GB/s, then an A/B | — |
| KS-E2 | open | either box: byte equality, then `TestDecode_dispatchProfile` deltas | — |
| KS-E3 | open | either box: `TEncode` from `TestDecodeTWE_split` on today's binding | — |
| KS-E4 | open | either box: the projections class of `TestPrefillInt4_profile` | — |
| KS-E5 | open | either box: the attention class at 2k, 8k and 16k | — |
| KS-E6 | open | either box: `TestPrefillInt4_profile` at 4,096 rows | — |
| KS-E7 | open | either box: KV bytes per slot and KV equality past the window | — |
| KS-E8 | open | either box: a WebGPU twin of `cuda/lora_cost_measure_test.go` | — |
| KS-E9 | open | either box: `TestSpike_capabilities` | — |
| KS-E10 | open | either box: dispatch count, then an A/B | — |
| KS-E11 | open | either box: `deepseek-tiny`, then V2-Lite | — |
| KS-F1 | open | either box: a fork/join counter and bit-identity on the tiny MoE fixtures | — |
| KS-F2 | open | either box: bit-identity on the tiny gpt-oss/qwen35/granite/lfm2 fixtures and KV rows held | — |
| KS-F3 | open | either box: `GOINFER_DELTANET_TIMING=1` with M35 (tier 1, minutes) | — |
| KS-F4 | open | nobara: per-fan-out timestamps inside a real 0.5B token | — |
| KS-F5 | open | either box: `AllocsPerRun` and trig calls per token on `deepseek-tiny` | — |
| KS-F6 | open | nobara: a kernel microbenchmark at depth 4096, then `TestR9_groupedDepthSweep` | — |
| KS-F7 | open | nobara: an in-process ABBA on the vetted phi3 q4k checkpoint | — |
| KS-G1 | open | either box: the 10 s window at `GOMAXPROCS=1` against the default (seconds) | — |
| KS-G2 | open | aikit: an int8 `BenchmarkHNSWBuild` | — |
| KS-G3 | open | aikit: time `topHits` inside the existing benchmarks | — |
| KS-G4 | open | aikit: a stage-by-stage benchmark of `Gemma4Preprocess` | — |
| KS-G5 | open | aikit: `BenchmarkQwenViT` with the matmuls timed apart | — |
| KS-G6 | open | aikit: `EncodeBatch` at concurrency 1, 2, 4, 8, A/B | — |
| KS-G7 | open | aikit: a histogram of tf over the test corpora | — |
| KS-G8 | open | aikit: a cold-cache HNSW query benchmark | — |
| KS-G9 | open | aikit: time the final loop in `BenchmarkFlatBinaryQuery` | — |
| KS-G10 | open | aikit: an offline block-max simulation counting documents scored | — |
| KS-H1 | open (512, G5) | done 2026-10-22 for head dim ≤ 256 and E-models on their batched pass: blocked prefill byte-equal to whole, admission prices one pass; KH1-G1 to G4 PASS. Left: G5 (Qwen3.5-9B at 20K, by day) and head-dim-512 models (the tile choice must key on the whole prompt's length) | `docs/measurements/ksh1-2026-10/README.md` |
| KS-H2 | open | nobara: `cuMemGetInfo` around the 7B build; free VRAM after `kvAllocPools` | — |
| KS-H3 | open | nobara: `TestKernelLocalMemoryCensus` and `TestMoERouteFirstLaunchReservation` after the move | — |
| KS-H4 | open (measured 2026-10-22) | next: release the tower scratch above a ceiling after the forward. Measured: Qwen3-VL-2B holds 2,622.5 MiB after a 4K forward and still 10 s later | `docs/measurements/ks-night-2026-10/README.md` |
| KS-H5 | open | either box: bytes allocated at build | — |
| KS-H6 | decision | KS-O3, after KS-H2 and KS-H3 read | — |
| KS-O1 | decision | owner: native Q5_K/Q6_K on the GPU q4k lanes (§2) | — |
| KS-O2 | decision | owner: the 64 KB shared-memory opt-in and past-bound prompts (§2) | — |
| KS-O3 | decision | owner: the two CUDA 384 MiB margins (§2) | — |
| KS-O4 | decision | owner: gpt-oss onto the batched attention kernel with a sink term (§2) | — |
| KS-DOC | done | — | notes 1–3, 5 and 7–10 corrected in their docs, 2026-10-22, in the commit that closed this row |
| KS-DOC-code | open | the next change to `cuda/resident.go` or aikit's `encoder/`: notes 4 and 11 are code comments (a product-code push); note 6 done 2026-10-22 with KS-H1 (`PrefillBlockRows`' comment) | — |

## 0. Where to start

By expected value for the effort, per box. Sizes: S up to a day, M a few days, L a week or more.

**nobara (CUDA)**
1. **KS-A2, minutes to measure.** The 0.5B and gemma3-1b may be limited by the cost of each kernel launch (about 14.6 µs
   each), not by the kernels. If the measurement confirms it, launching from goinfer's own thread could be worth up to
   about +30% on small-model decode.
2. **KS-B8, no code.** Wider prefill passes: 1.16–1.47× on the 1.5B's attention by simulation.
3. **KS-B2, then KS-B1**, bit-identical:
   - `attn_fused` has no vector loads in its PTX.
   - The fast GEMM runs at about 12% of int8 tensor-core peak and costs more per token than Ollama's whole token at
     512–1024.
   - Together, about 1.3–1.5× time to first token.
4. **KS-H3, KS-H5, KS-H2**, VRAM on the 8 GB card:
   - 131 MiB back on every MoE load;
   - about 430 MiB on the 7B (about +7,900 f16 positions);
   - about 140 MiB plus 135 MiB of real margin on Qwen3-8B at 16k.
5. **KS-B3, S for its first half.** Gemma 4 26B prefill on the grouped `moe_rows` kernels: 1.5–2.5×.
6. **KS-A1, needs KS-O1.** phi3-mini's whole gap to llama.cpp is 375 MB per token of int8 copies of Q5_K/Q6_K tensors.
7. **KS-B4, L, bit-identical.** A row-tiled exact attention for Qwen3.5 and Gemma long prompts: Qwen3.5-9B at 31k from
   about 310 s to 120–150 s.

**Mac (Metal)**
1. **KS-H1, S.** Exact-attention and E-model prompts run as one unblocked pass. At 20k tokens Qwen3.5-9B takes 4.8 GiB of
   transient scratch, and admission prices it 2.7 GiB short. Blocking it like the plain pass brings that to about 0.5 GiB.
2. **KS-C1, M.** The q4k GEMV does about 20 instructions and 4.5 loads per weight. A rebuild projects phi3-mini on Metal
   from 0.54× to about 0.9–1.15× Ollama, behind the existing gates.
3. **KS-D3, S–M, bit-identical.** The DeltaNet prefill scan keeps its state in device memory; CUDA's same fix took 1.23 s
   to 38 ms.
4. **KS-C2, KS-C3, KS-C4**, P33:
   - the down projection in R18b's form;
   - native int8 decode in R18's form;
   - the single-threadgroup quantizers.

   With the parked int4 head, these reach about 0.81–0.88× of mlx-lm on the 7B. They do not reach parity.
5. **KS-D2 (M) and KS-D1 (L).** Steel attention for head dims other than 128: about 2.2× on the 0.5B at 3900, and
   3.6–4.4× on Gemma 4 E2B.

**Either box**
1. **WebGPU decode (KS-E2, KS-E1, KS-E3, all bit-identical):** about 25% of a 1.5B token by arithmetic.
2. **CPU (KS-F1, KS-F2, KS-F3):** MoE fan-outs, gpt-oss attention and the hybrids' mixer core.
3. **aikit (KS-G1, KS-G4):**
   - G1: an S14.6 verdict read CPU-sample share as wall time, and a seconds-long check settles it.
   - G4: Gemma 4 image preprocessing is single-threaded on every backend.

## 1. Patterns across backends

- **Per-call overhead dominates the small models on every backend.**
  - CUDA: launches go through a thread-locked executor that yields (KS-A2).
  - CPU: every fan-out spawns fresh goroutines, with three per MoE expert (KS-F4, KS-F1, KS-F7).
  - WebGPU: token t+1 is encoded only after token t returns (KS-E3).
  - Metal and WebGPU: activation quantizers run as one threadgroup on one core (KS-C4, KS-E2).
- **Tiled kernels stage with scalar loads.** This applies to the CUDA prefill GEMM and fused attention (KS-B1, KS-B2) and
  WebGPU's int4 GEMM (KS-E4). The exact magic-number int-to-float conversion (`bits(c + 0x4B400000) − 12582912.0`, exact for
  |c| < 2²²) applies to the first and third.
- **Proven designs not yet ported to every place they fit:**
  - steel attention at head dims other than 128 (KS-D2);
  - `moe_rows` on the Gemma 4 path (KS-B3);
  - R18 rows on native int8 (KS-C3);
  - CUDA's LoRA fixes on WebGPU (KS-E8);
  - S17's tower kernels on the Gemma 4 towers (KS-D4);
  - the grouped acc64 attention on amd64 (KS-F6);
  - CUDA's KV window on WebGPU (KS-E7).
- **phi3-mini at q4k reads int8 copies of its Q5_K and Q6_K tensors on both GPUs** (KS-A1, KS-C1): about 375 MB per token,
  which is CUDA's whole gap to llama.cpp.
- **Families that still prefill one token at a time:**
  - phi3/phi4 at q4k on CUDA (KS-B6);
  - Mamba-2 on CUDA and Metal (KS-B12, KS-D8);
  - MLA.

## 2. Decisions for the owner

- **KS-O1, native Q5_K/Q6_K on the GPU q4k lanes (KS-A1, KS-C1 option c).**
  - Lever 2 (native Q6_K) was parked on 2026-09-27, on qwen 1.5B/7B at depth, where attention binds
    (`docs/tasks/task-int4-weight-quality-2026-09.md`).
  - phi3-mini at depth 128 is byte-bound, and Q5_K (283 of the 375 MB) was never priced.
  - On the CPU it stays int8: a native Q6_K CPU kernel is a recorded dead end in aikit (≤1.22×).
- **KS-O2, the 64 KB shared-memory opt-in (KS-B15).**
  - gocudrv v0.3.2 exposes `Function.SetAttribute`, which moves the exact attention kernels' bound from 12,160 keys to
    16,256.
  - Prompts in between could stay past-bound (exact GEMM).
  - Or they could regain the fast GEMM: 1.37× on Gemma-3-4B, but it re-exposes the measured cosine drift of 0.9865–0.990.
- **KS-O3, the two CUDA 384 MiB margins (`cuda/resident.go:ctxCapMarginBytes`, `cuda/resident.go:slotMarginBytes`).**
  - After KS-H2 and KS-H3, a margin built from measured parts could return about 100–200 MiB (estimate): 700–1,400
    positions on Qwen3-8B, or 1–2 expert slots on the 26B.
  - The parts are the ~144 MiB allocation floor, one prefill pass's scratch, and decode transients.
  - This is a guard threshold, so it is yours.
- **KS-O4, gpt-oss onto the batched acc64 attention with a sink term (KS-F2).**
  - It changes output.
  - It unblocks the KV ring, f16/int8 KV, the batched walker and MC3c for gpt-oss on the CPU.
- **Context, not a proposal:** the Metal int4 LM head (MEM8(a)/D5, parked "defer" 2026-10-21) is about 9% (1.5B) and 16%
  (7B) of P33's gap by KS-C's arithmetic (0.33 and 1.37 ms per token).

## 3. Corrections to existing docs and comments (KS-DOC)

The survey found eleven places where a doc or comment no longer matches the code.

**Corrected 2026-10-22:** notes 1–3, 5 and 7–10, each as a dated note or an in-place correction in its own doc. Notes 4 and
11 are code comments, left for the next change to those files (row KS-DOC-code). Note 6 is KS-H1's code change. Also
noticed: `docs/queue-performance.md` has two entries numbered P24 (`attn_fused`, and decode falling off with KV depth).

1. **`docs/tasks/task-kv-architecture-2026-10.md` §1's table** lists gpt-oss as "CPU ring". But
   `decoder/arch.go:Architecture.kvRingsOK` admits only `attnBatchedAcc64`, and gpt-oss runs `attnSinksF64`, so its sliding
   layers keep append-forever f32 KV (KS-F2).
2. **The memory campaign's MEM13-copies-CUDA row** ("remove CUDA's dense host pack") is out of date.
   - The dense pack is already gone (`cuda/pack_stream.go`).
   - What remains is whole-tensor packing for shared experts, Mamba-2 projections and drafter weights.
   - That is about 4 GB of int8 host copies on Nemotron-H 9B by arithmetic.
3. **KV4a's record** (`docs/measurements/kv4-cuda-2026-10-21.md` §1, §4) says the pool VRAM is priced. The 2 MiB rounding of
   each layer's pool buffer is not (KS-H2).
4. **The comment on `cuda/resident.go:singleBlockAttnShmemLimit`** says per-kernel attribute support does not exist. gocudrv
   v0.3.2 has `Function.SetAttribute`; aikit's `gpu` layer does not pass it through (KS-B15).
5. **`docs/tasks/task-multimodal-support-2026-10.md` S14.6** concludes "Both are matmul-bound" from CPU-sample share, which
   sums over threads. The serial share of wall time is much larger (KS-G1).
6. **`metal/backend.go:metalResident.PrefillBlockRows`** excludes exact-attention models for a 4,096-key decline. The tiled
   twin removed that decline (KS-H1).
7. **`docs/tasks/task-forward-architecture-2026-10.md` §6** says "decode already replays CUDA graph segments". Graphs are
   opt-in and off by default.
8. **`docs/tasks/red-october.md` §2.3** calls the CUDA GEMM done ("Nothing left to fund there"). It compared against a
   whole-curve Ollama figure that the 2026-09-21 per-interval marginals supersede (KS-B1).
9. **P24's 1.72% of tensor peak and 12.6% occupancy** describe the retired 64×64 kernel. The shipped 128-row kernel has no
   ncu reading (KS-B2).
10. **P32's plan, "raise `ATTN_MAXHD` to 512",** does not fit Apple7. The fused kernel's accumulator alone would be 32 KB at
    head dim 256 and 64 KB at 512, against a 32 KB threadgroup limit (KS-D1).
11. **aikit:** the comment on `matmulBTQ8Into` in `encoder/` still says amd64 takes the dequant fallback; it no longer does.

## 4. CUDA decode (KS-A)

**Per-token accounting (RTX 2070 SUPER, depth 128; bytes from the code, rates from P31):**

| | 0.5B `int4` | gemma3-1b `int4` | phi3-mini `q4k` |
|---|---|---|---|
| launches per token | 195 | 341 | 419 |
| weight bytes per token, goinfer | 338 MB | 696 MB | 2,711 MB |
| llama.cpp Q4_K_M (from memory for the first two) | ~392 MB | ~799 MB | 2,336 MB |
| measured greedy tok/s | 350.9 | 200.6 | 113.8 |
| share of the 448 GB/s byte floor | 26.5% | 31.1% | 68.9% |
| token time ÷ launches | 14.6 µs | 14.6 µs | 21.0 µs |

The two small models read fewer bytes than llama.cpp and still trail it. phi3-mini's time outside weight streaming
matches llama.cpp's (1.46 against 1.50 ms), so its gap is bytes.

**KS-A2 · Launches may cost about 14.6 µs each on the small models (measure first).**
- **Where:** `cuda/resident.go:cudaResident.launch` → aikit `gpu` `Queue.Launch` (an argument slice per launch) → gocudrv
  v0.3.2 `Function.LaunchOn` (a lock per buffer argument) → `Executor.DoJob`.
- **Evidence, three readings that agree:**
  - The 0.5B and gemma3-1b cost the same 14.6 µs per launch, and their token-time ratio (1.749) equals their launch-count
    ratio (1.749); bytes would predict 2.06.
  - `glu_quant` at 1024 threads moved the 0.5B only +0.5–0.7%, where the D7 record's arithmetic predicted +4.8%.
  - The flash lane's one extra launch per layer cost the 0.5B 0.37 ms; 24 × 14.6 µs is 0.35 ms.
- **Mechanism (inferred):**
  - Both goinfer's executor and gocudrv's are goroutines locked to OS threads. They spin-poll and call `runtime.Gosched()`
    every 128 polls.
  - A Gosched on a thread-locked goroutine hands the thread off, which can mean a futex sleep and wake.
  - The August spike ran the 0.5B at ~504 tok/s in process on gocudrv v0.2.0; served today it runs at 351.
- **Change:**
  - A raw launch path in aikit modeled on `Queue.UploadAsyncAt`, which already issues raw driver calls from the caller's
    thread.
  - Arguments pre-packed at build time through gocudrv's `Pack` / `LaunchPackedOn`.
  - Or an upstream fix: no Gosched from a spin loop on a locked thread.
- **Gain (est.):** if confirmed, the 0.5B becomes GPU-bound near 2.0–2.1 ms, up to +30–35%. gemma3's GPU time is unmeasured.
- **Class:** output unchanged.
- **Size:** S to measure, M to build.
- **First measurement:**
  - `TestCUDA_launchCost` on today's pins.
  - An in-process interleaved A/B of the 0.5B at depth 128 with CUDA graphs on (`GOINFER_CUDA_GRAPHS=1` plus the unsafe
    tenancy switch). This was never measured on the 0.5B; the 1.5B read 1.01×.
  - A futex count over 200 tokens.
  - Kill it if graphs give ≤1.03× and the launch cost is ≤6 µs.
- **Prior art:** extends `docs/completed/cuda-decode-headroom-audit.md` §3 (a de-hop was proposed, not built) and
  `docs/cuda-graphs-investigation.md`.

**KS-A1 · phi3-mini: keep the GGUF's Q5_K and Q6_K tensors native under `q4k` (needs KS-O1).**
- **Where:** `decoder/weightmat.go:streamQuantizedRow4` ("the rest of quantQ4K is int8 W8A8"), `decoder/gguf.go:q4kMat`
  (Q4_K only), and `rowQ4K` / `projRow` in `cuda/actgroup.cu`.
- **Waste:** the 32 Q5_K `attn_qkv` tensors and 17 Q6_K tensors (16 `ffn_down` plus the head) run as int8. That adds
  +283, +72 and +18 MB per token, for 2,711 MB against llama.cpp's 2,336.
- **Change:**
  - Per-32 `rowQ5K` (rowQ4K plus the high bit; 176-byte blocks are 16-byte aligned) and `rowQ6K` (subtract 32, then
    `dp4a`, then the int8 sub-scale × d × activation scale).
  - An aligned Q6_K layout of the same size at upload.
  - A native head.
- **Gain (est.):** at today's 309 GB/s, ~132 tok/s, about 1.03× llama.cpp (0.89× today). Frees 375 MB of VRAM.
- **Class:** changes output toward the file (exact Q5_K/Q6_K instead of an int8 requant); goes through the q4k agreement
  gates.
- **Size:** M.
- **First measurement:**
  - Plan bytes per token at tier 0 (2.711 → 2.336 GB).
  - A Q5_K GEMV against `gemv_w8a8_g32` at N=9216, K=3072. It must reach ≥0.95× that kernel's GB/s.

**KS-A5 · Load each short GEMV row in one round (output unchanged).**
- **Where:** `fused_rms_gu` and `fused_rms_qkv` in `cuda/fused_qkv.cu`, their rows-per-warp twins, and aikit's
  `gemv_w4a8_fwd` (`gpu/gemv_quant.cu`).
- **Waste:** a K=896 row is three serialized load rounds of 64, 32 and 16 words, and the next row waits for the reduction.
  Committed ncu times track row length: `fused_rms_gu` runs at 200 GB/s on the 0.5B shape against 355–368 GB/s at
  K ≥ 3072. The 0.5B's gate/up kernel is 0.59 ms of a 2.85 ms token.
- **Change:**
  - Issue all of a row's words at once (≤4 per lane for K ≤ 1024), with each lane's words and order unchanged.
  - Prefetch the next row before the reduction.
  - Issue the first row's weight loads before the norm prologue.
- **Gain (est.):** ~0.3 ms per token (~10%) on the 0.5B and ~0.2 ms on gemma3, once decode is GPU-bound (KS-A2).
- **Size:** S–M.
- **First measurement:** ncu DRAM percent of peak at (896, 4864). Kill it at ≥80%.
- **Prior art:** new. The recorded 4× unroll null result added rounds for 112-word rows; this removes them.

**KS-A3 · Spread `topk_select` over many blocks (output unchanged).**
- **Waste:** one SM samples, sorts, compacts and scans for Z: 102 µs per token, against the argmax's 20 µs. This is most of
  P31's preset-over-greedy cost (+62 µs on the 0.5B, +125 µs on gemma3), where llama.cpp is flat.
- **Change:**
  - Multi-block compaction and global max with atomics.
  - The same Z partial sums, in the same order, spread over 32 blocks.
  - K = top_k when 1 ≤ top_k ≤ 256 (today K is 256 or 1024 even for top_k 40).
- **Gain (est.):** ~75 µs (2.6%) on the 0.5B preset and ~140 µs on gemma3.
- **Size:** S–M.
- **Prior art:** new; R7 shipped the single-block kernel.

**KS-A4 · Gemma: fuse the post-norm with its residual add, and QK-norm with `rope_kv` (output unchanged).**
- **Change:**
  - An `rmsnorm_f32_add` with explicit `__fadd_rn` / `__fmul_rn`, so nothing contracts.
  - A per-head `qk_norm_rope_kv`; at head dim 256 each rope pair is what one `qk_norm` thread already holds.
- **Gain (est.):** removes 78 of gemma3's 341 launches per token, 0.16–0.23 ms of GPU time (3–4.5%), and more if KS-A2
  holds. It also serves Gemma 2/4, Olmo and Qwen3.
- **Size:** S.
- **Prior art:** new; `docs/tasks/task-forward-architecture-2026-10.md` §4.5 names the cost.

**KS-A6 · Per-token host plumbing (output unchanged).**
- **Waste:**
  - The embedding upload is synchronous: a context sync, a pageable copy, then another sync.
  - The readback is pageable, into a per-token slice.
  - goinfer's executor parks between tokens; the spike measured that hand-off at 15.3 µs.
- **Change:**
  - A pinned ring with an async upload, as `kvFlushTabs` already does.
  - Mapped host memory for results.
  - A short spin before parking.
- **Gain (est.):** 30–75 µs per token, 1–2.6% on the 0.5B, on every sampling path.
- **Size:** S.
- **First measurement:** kill it under 20 µs. HL (a CUDA token chain) covers greedy and temperature-only sampling, not the
  top-K preset.

**KS-A7 · phi3: one launch for Q|K|V and one for gate|up.**
- **Change:** keep the GGUF's fused `attn_qkv` / `ffn_up` as single buffers with offset views. That is 96 fewer launches
  per token.
- **Gain (est.):** about 0.19 ms (~2.2%), stacking with KS-A1.
- **Size:** S.

**KS-A8 · Greedy argmax over many blocks.**
- **Change:** a 64-bit `atomicMax` over (value, inverted index), which keeps "lowest index wins", or a fold into the head
  GEMV as Metal's `gemv_w8a8_amax` does.
- **Gain (est.):** ~16 µs (0.6%) on the 0.5B.
- **Size:** S.

## 5. CUDA prefill, batched and family kernels (KS-B)

**Where the marginal gap sits (1.5B; `docs/measurements/peer-prefill-cuda-2026-09-21.md` split by
`docs/measurements/attn-fused-tile128-default-2026-09-21.md`):**

| interval | goinfer ms/token | GEMM | attention | glue | Ollama ms/token |
|---|---|---|---|---|---|
| 512–1024 | 0.189 | 0.133 | 0.033 | 0.023 | 0.111 |
| 2048–3900 | 0.289 | 0.139 | 0.130 | 0.020 | 0.160 |

**KS-B1 · The fast-lane GEMM is latency-bound at 12% of int8 peak (bit-identical rework).**
- **Where:** `gemm_w4a8_mma` in `cuda/gemm_w4a8_mma.cu`, launched from `cuda/prefill.go:cudaResident.bGemvB`.
- **Waste:**
  - It runs at 19.3 TOPS (156 TOPS peak).
  - Per-token floors: tensor 16.8 µs; the I2F conversion, one per output per 32-group at quarter rate on sm_75, 33.6 µs. The
    kernel takes 136 µs.
  - Each group iteration loads its weight fragments and scales (behind branches) and then computes, unpipelined.
  - A-panel staging is one 4-byte load plus a runtime integer `div` per iteration.
  - The PTX has 0 vector global loads, 32 `cvt.rn.f32.s32` and a `div`.
  - k and v (N=256) each run as a 32-block grid on 40 SMs.
- **Change:**
  - 16-byte staging loads with a compile-time thread count and no `div`, prefetched one K step ahead.
  - Group g+1's fragments loaded while g computes.
  - `__int_as_float(c + 0x4B400000) − 12582912.0f` for the conversion (|c| ≤ 32,512).
  - q‖k‖v as one launch with a segmented N.
- **Gain (est.):** 1.4–2× on the GEMM. 1.5B time to first token: 1.3–1.5× at K=512 (84 → 56–65 ms) and 1.2–1.4× at 3900.
- **Size:** M.
- **Risk:** about 24 more registers, so 4–5 blocks per SM instead of 6.
- **First measurement:** ncu on the gate and k launches at M=512 (occupancy limiter, stall split, tensor and XU pipes),
  then the conversion swap alone, gated on `math.Float32bits` equality.
- **Prior art:** reopens red-october §2.3's "done" (§3, note 8).

**KS-B2 · `attn_fused` stages K/V with one scalar load in flight per thread (bit-identical).**
- **Where:** the staging loops of `attn_fused_impl` in `cuda/attn_fused_bm.cu` and `cuda/attn_fused.cu`.
- **Waste:**
  - Every `attn_fused*` PTX module has zero vector global loads; `decode_fa*` has 279–288.
  - That is about 77K cycles per (block, 64-key tile) against an ~8.2K-cycle MMA floor.
  - 256 threads at ~178 registers allow one block per SM.
  - The V^T store has a 4-way bank conflict.
  - R5's 2.39× from the taller tile fits this mechanism.
- **Change:**
  - 16-byte loads issued before any convert or store.
  - Register prefetch of the next tile.
  - A key-major lane map (or `ldmatrix.trans`) for V^T.
  - No mask arithmetic on interior tiles.
- **Gain (est.):** 2.5–4× on attention at depth (K=3900: 337 → 85–135 ms), so 1.27–1.36× time to first token on the 1.5B
  at 3900. With KS-B1, the deep marginal lands near 0.13–0.15 ms per token against Ollama's 0.160.
- **Size:** S–M.
- **First measurement:** R5's open ncu staging-share read, then the attention A/B under the 456-shape `Float32bits` sweep.
- **Prior art:** extends P24 and R5.

**KS-B8 · Wider prefill passes on few-head models (bit-identical, no code to measure).**
- **Waste:** a 512-row pass gives the 1.5B 48 blocks of fused attention on 40 SMs at one block per SM. That is 60–64%
  efficiency by simulation at K=2048–8000.
- **Change:** 1024- or 2048-row passes where scratch allows (138 / 276 MiB on the 1.5B; admission prices them).
- **Gain (simulated, attention only):** 1.5B 1.16–1.26× at 1024 rows and 1.37–1.47× at 2048; the 7B 1.02–1.09×.
- **First measurement:** `TestPrefillDecomp` with the pass width at 512/1024/2048 and K=2048/3900.
- **Prior art:** reopens the `cuda/prefill.go:prefillDefaultChunk` note, which was measured on the old grid.

**KS-B3 · Gemma 4 26B (M26) expert-major prefill never got option A (bit-identical).**
- **Where:** `cuda/moe_expert_major_gemma4.go:cudaResident.prefillGemma4ExpertMajorRun`; it has no reference to the grouped
  kernels that `cuda/moe_expert_major.go` uses.
- **Waste:**
  - Phase 2 is three launches per (row, rank) pair, 12,288 per layer per chunk.
  - Phase 1 is 11 launches per row, including the dense FFN as M=1 GEMVs that re-read ~10 MB per row.
  - Each distinct expert's upload synchronizes the context twice.
- **Change:**
  - Phase 2 onto `moe_rows` (hidden 2816 and moeInter 704 are multiples of 32).
  - Phase 1 through the exact batched twins.
- **Gain (est.):** 1.5–2.5× on M26 prefill (~16 → 7–11 ms per token).
- **Size:** S (Phase 2), M (Phase 1).
- **First measurement:** the MoE category and a launch count at M=512, then `TestMoEExpertMajorGemma4CUDA_bitIdentical`.

**KS-B4 · Exact attention, row-tiled (bit-identical; reopens P35 option A).**
- **Where:** `attn_batched` (`cuda/prefill_batched.cu`) and the `splitkv_*_rows` kernels.
- **Waste:**
  - Each (head, row) streams its whole key range with no reuse across rows, at 0.43 TFLOPS.
  - This path serves Gemma 2/3/4, the Qwen3.5/3.6 attention layers, phi3, E-models, prompts under 512, speculative verify
    and past-bound prompts.
  - Qwen3.5-9B: 310 s at 31.4k, about 220 s of it attention (inferred from its flat GEMV).
- **Change:**
  - R rows per block stream 128-key K/V tiles through shared memory.
  - Pass 1 computes the row max; pass 2 recomputes the same dots, accumulates into the same strided slots and applies the
    same 128-wide ladder, folding V in ascending key order.
  - No score row is stored: no 12,160-key bound and no 64 MiB scratch.
- **Gain (est.):** 4–7× on exact attention. Qwen3.5-9B from 310 to about 120–150 s at 31.4k, and 38.5 to about 27 s at 8k.
- **Size:** L.
- **Risk:** the shared-memory budget at head dims 256/512, plus windows, sinks, softcap and image blocks.

**KS-B5 · Expert-major for shared-expert MoE (M35) and gpt-oss (bit-identical).**
- **Where:** `cuda/moe_expert_major.go:cudaResident.prefillMoEExpertMajorEligible` declines `hasShared`, bias tables and the gpt-oss route.
  Those models fall back to per-row FFN, the shape P20 found 2.3–2.7× slower on M26.
- **Change:**
  - The shared expert through the batched twins, plus a row twin of `shared_gate_combine`.
  - For gpt-oss: bias tables by expert id and route/GLU row twins.
- **Gain:** unmeasured.
- **Size:** M.
- **Prior art:** extends P20 ("not attempted"). Confirm `hasShared` on a real M35 load first.

**KS-B6 · phi3/phi4 at their default `q4k` prefill one token at a time.**
- **Where:** `cuda/prefill.go:cudaResident.prefillStaticDecline` (per-32 activations) and `cuda/prefill.go:nonBatchableKind` (`q4k`).
  About 35 s at K=3900.
- **Change:** batched row twins of the per-32 quantizers in `cuda/actgroup.cu`, plus a batched `gemv_q4k_g32`.
- **Gain (est.):** 5–9× time to first token. Exact-class twins.
- **Size:** L.
- **Prior art:** the unowned remainder of `docs/tasks/task-actquant-pergroup-2026-09.md`.

**KS-B7 · An exact-class tensor-core GEMM (bit-identical to `gemv_w4a8_rn`; reopens `docs/ollama-chase.md`'s "no tensor-core
tiling reproduces").**
- **Change:**
  - An m8n8k16 IMMA with half the A fragment zeroed gives each 8-element word's exact int32 sum.
  - Each lane keeps 32 residue partials per output and replays rn's fold and shuffle tree.
- **Gain (est.):** 2.5–4× on the exact GEMM, which is 93% of Qwen3.5-9B's prefill: 1.78 → 0.55–0.8 s at 621 tokens.
- **Size:** L.
- **First measurement:** a prototype at the 1.5B gate shape.

**KS-B9 · One segmented-N launch per projection group.**
- **Waste:** k and v (16 blocks each) cost as much as q in `gemv_w4a8_rn`, whose per-warp time does not depend on N, and
  DeltaNet's two small projections run as 2-block launches.
- **Gain (est.):** 8–12% of the GEMM/GEMV category.
- **Class:** bit-identical. **Size:** S–M.

**KS-B10 · Generic expert-major plumbing.**
- **Where:** `cuda/moe_expert_major.go:cudaResident.prefillMoEExpertMajorRun`.
- **Waste:** Phase 1 launches three tiny kernels per row (1,536 per layer per chunk), including the single-thread
  `moe_route`. Each distinct expert costs two context syncs, and the expert DMA is synchronous.
- **Change:** row twins of the router and route, slot ids passed by value, and `dmaExpertSlot` with events.
- **Gain (est.):** 1.15–1.25× on Mellum-class prefill.
- **Class:** bit-identical. **Size:** M.

**KS-B11 · SwiGLU into the gate/up GEMM epilogue (bit-identical).**
- **Gain:**
  - −224 KB of activation traffic per token per layer on the 1.5B (~4.5% of the shallow marginal).
  - −35 MiB of pass scratch on the 1.5B and −74 MiB on the 7B: 51% of pass scratch.
- **Size:** M.
- **Prior art:** extends the filed `dScrB` alias (task-kv-architecture, KV3).

**KS-B12 · Mamba-2 row-batched prefill (Nemotron-H; bit-identical).**
- **Waste:** today every prompt token runs a decode forward.
- **Change:** row twins of `mamba_conv`, `mamba_ssm` and `mamba_gnorm`, following the DeltaNet recipe (5.7× there).
- **Gain (est.):** 4–6× time to first token.
- **Size:** M.

**KS-B13 · aikit's f32 tower GEMM (`gemm_f32_bias` in `gpu/vit.cu`).**
- **Waste:** 3.56 TFLOPS (39% of fp32 peak) on a 4×4 micro-tile, with 12 shared-memory reads per 16 FFMA.
- **Change:** a 128×128 block with 8×8 micro-tiles, `LDS.128` and double buffering, keeping each output's ascending-k chain.
- **Gain (est.):** 1.5–1.8× on the GEMMs (40–53% of each f32 tower); SigLIP 1.97 → ~1.5 s.
- **Class:** bit-identical. **Size:** M, aikit.

**KS-B14 · The int8 SigLIP fallback tower.**
- **Waste:** `cuda/vision_encoder.go:addInPlaceHost` makes ~55 host round trips per image (~0.5 s of 4.08 s), and the int8
  GEMV runs at ~1 TOPS.
- **Change:** device residual adds (`residual_batched`), and an exact int8 `mma.sync` GEMM.
- **Gain (est.):** the tower to about 1.2–1.5 s. Matters only when the f32 tower does not fit.
- **Class:** bit-identical. **Size:** S + M.

**KS-B15 · The 64 KB shared-memory opt-in (needs KS-O2).**
- **Waste:** the needed call exists in the pinned gocudrv, but aikit does not pass it through.
- **Change:** pass the attribute through aikit and set it on the exact attention kernels: bound 12,160 → 16,256 keys.
- **Class:** bit-identical attention; the decision is about the GEMM that comes with it. **Size:** S.

## 6. Metal decode (KS-C)

**Where P33's gap sits, by arithmetic (depth 128; mlx-lm's 4-bit head is from memory):**

| | goinfer bytes per token | mlx-lm bytes | goinfer vs mlx-lm time | effective GB/s |
|---|---|---|---|---|
| 1.5B | 975 MB (737 body + 234 int8 head) | ~872 MB | 10.87 vs 7.24 ms | 90 vs 121 |
| 7B | 4,223 MB (3,671 + 546) | ~3,984 MB | 33.67 vs 25.00 ms | 125 vs 159 |

- **Where goinfer is slow:**
  - The down projection is the slowest large GEMV (85 / 111 GB/s).
  - qkv and o run at 64–110 GB/s.
  - About 1–2 ms per token is fixed cost. Part of that is built into W4A8 on Metal: a per-tensor int8 activation scale
    needs a global max, which is why the quantizers run as single-threadgroup kernels. MLX runs f16 activations.
- **Why candidates alone can't close it:** KS-C2, C4, C5 and the parked int4 head bring the 7B to about 0.81–0.88× of mlx-lm
  and the 1.5B to about 0.73–0.77×. The rest needs P33's two-sided trace.
- **P33's goinfer-side first step, tier 1, about 5 minutes per model:**
  - Re-run `TestMetalDecodeDecomp` at depth 128 on today's build; its record predates R18, R18b, C-B01 and B-P03.
  - Split its "rope, kv store, act-quant, ctx-quant" category by pipeline.
  - Add the greedy chain's head as its own category.

**KS-C1 · Rebuild the q4k lane's GEMVs (phi3-mini on Metal, 0.539× Ollama).**
- **Where:** `metal/model.go:resident.q4kGemv`; `gemv_q4k_f32` (`Q4K_F32_BODY`), `W8_F32_BODY` and `gemv_w8f32_head` in
  `metal/kernels/kernels_msl.go`; the step twins in `metal/batch_rows.go`.
- **Waste:**
  - Per weight, about 20 instructions and 4.5 loads. Every lane repeats `q4k_scale_min` for each sub-block, loads one byte
    of weights and 8 bytes of f32 activation, and the int4 rows kernel needs about 3.8 ALU operations and 0.07 loads.
  - Per token, 2,760 MB: 1,302 Q4_K, 1,407 of int8 copies of Q5_K/Q6_K, and 50 of KV. llama.cpp reads 2,387.
  - The Q4_K GEMVs run near 44–48 GB/s by subtraction.
  - OP6's 8-row step ran only 1.985× the sequential loop, so per-row work, not bytes, sets the cost.
- **Change:**
  - (a) A Q4_K GEMV in llama.cpp's shape (from memory): a `uint4` of quants per lane per super-block, 2–4 rows per
    simdgroup with the activations in registers, scales decoded once per row and super-block, and the min term as
    dmin·m·Σa.
  - (b) The int8-f32 GEMV at 4 rows per simdgroup, with each row's arithmetic unchanged.
  - (c) Native Q5_K/Q6_K (KS-O1).
- **Gain (est.):**
  - (a)+(b): 37–48 tok/s, 0.89–1.15× Ollama.
  - (c): about 2.5 ms more per token and −373 MB per token. `q4kLaneBuf` copies all 2.76 GB on Phi-3, where the M1 cell
    voided three times on memory.
- **Class:** fidelity-gated by the existing G-Q1 (`TestQ4KLaneKernels`) and G-Q2 (`TestQ4KLane_realPhi3NonInferiority`).
  `q4kRowsSrc` is regenerated from the same body.
- **Size:** M for (a)+(b), L for (c).
- **First measurement:** time the shipped kernels at the real Phi-3 shapes with `TestMetalGEMVS0`'s method. Kill (a) if
  Q4_K already reads ≥100 GB/s.

**KS-C2 · The down projection in R18b's masked form, in coal's per-word order (bit-identical).**
- **Where:** `gemv_w4a8_resid_staged<R>` (and `gemv_w4a8_coal_staged<R>`) in `metal/kernels/kernels_msl.go`.
- **Waste:** it still runs `int gi = (int((x)&0xF)-8)*int(a0.x) + …` with `acc[r] += float(gi) * …`: about 5.4 operations per
  weight against R18b's 3.8.
- **Change:**
  - Half-precision staged activations, pre-scaled as `sa_rows_acc` does.
  - Two masked half-word dots per word, plus Σa, in the same lane and word order.
  - Integer sums stay exact (|gi| ≤ 8,128), so each lane's float chain is unchanged.
  - The 7B's 37.9 KB of halves needs int8 staging or two K chunks split at word 1,184.
- **Gain (est.):** R18b gave 1.19–1.235× on gate|up. At 1.10–1.25× on down: −0.23 to −0.51 ms on the 1.5B and −0.88 to −1.93
  ms on the 7B. `mc3_btd` stays valid.
- **Size:** S–M.
- **First measurement:**
  - A `resid_mk4` prototype beside `resid_st4` in `metal/gemv_r18_test.go`, byte-equal at both shapes.
  - Then `TestR18InSequence`, 7 paired reps.
  - Kill it under 1.03× or on any differing byte.

**KS-C3 · Native int8 decode GEMVs get R18's rows and staging (bit-identical).**
- **Waste:** `W8A8_SA_BODY` / `W8A8_COAL_BODY` run one row per simdgroup with 4-byte loads, and the down projection re-reads
  activations from device memory in every lane. The 1.5B at int8int8 decodes at 70.6 tok/s, with its int8 body near
  132 GB/s (inferred).
- **Change:** R=2/4 rows per simdgroup, `uint4` loads, and activations staged per threadgroup. int32 sums are exact.
- **Gain (est.):** −1.7 to −2.0 ms (12–14%) if the body reaches 160–165 GB/s. This is `-backend auto`'s dense int8int8 path,
  including the goinfer-chat bundles.
- **Size:** S–M.
- **Prior art:** `docs/tasks/task-metal-int8-2026-10.md` slice 1, "can follow", never built.

**KS-C4 · Fold the single-threadgroup quantizers into their neighbors (designed bit-identical).**
- **Waste:** `swiglu_quant` and `quant_vec` are dispatched as one 256-thread threadgroup (one core of 16), and
  `swiglu_quant` makes two passes, computing `exp` twice.
- **Change:**
  - The gate|up epilogue computes h and writes per-threadgroup max partials; the row mapping is index math only.
  - The down projection's staging reduces the partials (max is exact) and quantizes with `swiglu_quant`'s own statements.
  - The same pattern for ctx before the o-projection.
- **Gain (est.):** 0.3–0.5 ms (3–5%) on the 1.5B and 0.5–0.8 ms on the 7B, if these are half to two thirds of the category.
- **Size:** M.
- **Risk:** this file has a history of fast-math drift across code shapes, so prove it with the snapshot golden and an
  in-sequence byte check.
- **First measurement:** kill it if the two pipelines total under 0.25 ms on the 1.5B.

**KS-C5 · Legacy decode attention below 1024 keys (bit-identical).**
- **Waste:** at 128 keys it costs 0.67 / 0.78 ms per token, with serial 8-wide V-load batches (16 dependent round trips) and
  about 16 barriers.
- **Change:** 16/32-wide V batches in the same order, and `simd_shuffle_down` for tree levels ≤16, keeping
  `tgReduceAttn` at 128.
- **Gain (est.):** −0.1 to −0.2 ms (1–2%); low confidence.
- **Size:** S.

## 7. Metal prefill and family kernels (KS-D)

**The prefill GEMM is near its practical roof.** R16 runs 2.63–2.99 TFLOPS, 81–88% of Apple's MPS f16 GEMM on the same shapes
with no int4 dequant to pay. At K=3900 the 1.5B pass is 83% GEMM, so the 0.90× against mlx-lm there is most likely GEMM
efficiency (an inference; there is no MLX-side breakdown). KS-D7 is the one kernel-level lever found.

**KS-D1 · Gemma 4 (head dims 256/512): a steel kernel with the head dim split across simdgroups.**
- **Where:** today `metal/prefill.go:resident.prefillAttnKernels` sends every layer with head dim >128 to the exact
  `attention_prefill`. That kernel uses one threadgroup per (row, head), a scalar dot per key and a serial P·V scan per
  output dim, at ~0.06–0.07 TFLOPS.
- **Why P32's plan does not fit:** `attention_prefill_fused` keeps `oScr[ATTN_SGPT][8*ATTN_MAXHD]` floats in threadgroup
  memory. That is 32 KB at head dim 256 and 64 KB at 512, against Apple7's 32 KB.
- **Change:**
  - A head-dim template on R19's steel kernel, with 8 simdgroups each owning a 128-wide slice (4 row groups × 2 slices at
    256; 2 × 4 at 512).
  - Partial scores exchanged through threadgroup memory and summed in one fixed order, so every slice holds the same
    softmax state.
  - K/V staging ≤21 KB.
- **Gain (est.):** attention is about 80% of E2B's 4,096-token prefill, so 28.4 s → 6.4–7.8 s (3.6–4.4×), and about 6–7× at
  8,192. A head-dim-256-only kernel gives ~1.3×; the 512 global layers hold 68% of the attention FLOPs.
- **Class:** fidelity-gated (R19's class).
- **Size:** L.
- **First measurement:** a test seam that skips the attention dispatch, timed on E2B at 4,096 (about 3 minutes). Kill it
  under 50%.

**KS-D2 · Steel prefill attention for head dims 64, 80 and 96.**
- **Where:** `metal/prefill.go:resident.prefillAttnKernels` takes steel only for head dim 128. Others run
  `attention_prefill_fused`, which R19 measured 7.2× slower.
- **Affected:** Qwen2.5-0.5B, Llama-3.2-1B and SmolLM2 (64), and Phi-3 at int4 (96).
- **Gain (est.):** attention is ~65–70% of the 0.5B's pass at 3900, so 3.4–3.8 s → ~1.6 s (~2.2×), and ~1.2× at 512.
- **Class:** fidelity-gated.
- **Size:** M.
- **First measurement:** `TestMetalPrefillDecomp` on the 0.5B. Kill it if attention is under 30% at 3900.

**KS-D3 · DeltaNet prefill scan with the state on chip (bit-identical).**
- **Where:** `delta_rule_seq` in `metal/kernels/deltanet_kernels_msl.go`, from
  `metal/prefill_deltanet.go:resident.encodePrefillDeltaMixer`.
- **Waste:** each thread's 512-byte state row (`device float* S`) is read and written in device memory for every prompt row,
  and lanes are 512 B apart. CUDA's register version, `delta_rule_rows_128`, took 1.23 s to 38 ms, bit-identical.
- **Change:**
  - A 32-thread threadgroup holds 32 rows in threadgroup memory (16 KB, laid out without bank conflicts), or holds the row
    in registers as CUDA does.
  - Load once, run the same loop in the same order, store once.
  - Also keep `delta_conv_seq`'s window in registers.
- **Gain (est.):** the rule ÷3–10, so 1.1–1.4× on Qwen3.5/3.6 prefill.
- **Gate:** `TestDeltaNetSeqKernels_matchDecodeBitwise`.
- **Size:** S–M.
- **Prior art:** a new form. D-B05 reorders the sums and waits on O2; this needs no decision.

**KS-D3 gates, registered 2026-10-22 before the code (D3-G).** Re-checked on main at 14eea5f8: `delta_rule_seq` still reads and
writes each thread's `hk`-float state row in device memory for every prompt row, with 128 threads per threadgroup and rows `hk`
floats apart. `delta_conv_seq` still shifts its window through device memory. The build:
- **`delta_rule_seq_tg`:** a 32-thread threadgroup holds its 32 state rows in threadgroup memory (16 KB at `hk` = 128, element `kd` of
  lane `l` at `kd*32 + l`, so the 32 lanes hit 32 banks). It loads once, runs the same loop in the same order, and stores once.
  It is taken when `hk` ≤ 128.
- **`delta_conv_seq_reg`:** the conv window in registers when `K` ≤ 5, the same order of operations.
- The current kernels stay as the fallback and as the in-process A/B's old arm.

| gate | tier | instrument | PASS iff |
|---|---|---|---|
| D3-G1 kernel identity | 0 | `TestDeltaNetSeqKernels_matchDecodeBitwise` with the new kernels, Qwen3.5 geometry (`hk` = `hv` = 128, `nv` = 48), M = 1, 7, 33; and the same at a tiny fixture's geometry | every stage's output, the window and the state equal the decode kernels' bit for bit (as the current seq kernels do) |
| D3-G2 model identity | 0 | Metal resident at int4 on qwen35-tiny, qwen3_5-tiny, qwen3_5_moe-tiny and olmo_hybrid-tiny (the Gated-DeltaNet fixtures, #163's included): a 40-token batched prefill's last logits and 8 decode steps after, new kernels against the current ones in one process | byte-equal on every fixture |
| D3-G3 speed | timed, Mac, by day after 07:00, timing lock | GPU time (GPUEnd − GPUStart) of `delta_rule_seq` alone at M = 512, Qwen3.5 geometry, the current kernel against `delta_rule_seq_tg`, interleaved A B B A, 10 pairs in one process; `delta_conv_seq` the same, read only | SHIP if the median new/old ratio is ≤ 0.98 and at least 9 of 10 pairs are below 1; PARK at 0.98 to 1.00; OFF above 1.00 (no change to the default) |

Planted defect, run once on the committed tree and reverted: the new rule kernel's two dot products in a different order
(`kvdot` summed in reverse). D3-G1 goes red. Stopping rule: one pass each; a FAIL is diagnosed, not re-run. Cost: seconds; D3-G3
about a minute.

**KS-D4 · Gemma 4 vision and audio towers and EmbeddingGemma 2 onto S17's kernels.**
- **Where:** `metal/embeddinggemma2.go:eg2Ops.gemm`.
- **Waste:** these towers run aikit's `gemm_f32_sg_big` (1.19 TFLOPS) and unfused attention, against `tower_gemm_w32` / `w16`
  (2.02–2.22) and `tower_attn`. The Gemma 4 tower is recorded at 1.3–2.3 s per image.
- **Gain (est.):** −0.6 to −1.5 s per E2B/E4B image turn (6.2 s recorded). w16 halves the tower weights (604 → 302 MB).
- **Class:** w32 is expected to be bit-identical (confirm by bytes); w16 and fused attention take the tower gate.
- **Size:** S–M.

**KS-D5 · Fuse SwiGLU into the gate|up GEMM and drop `guF` (bit-identical).**
- **Waste:** `guF` exists only to carry gate and up to `swiglu_f16`. It is 52–55% of the 1.5B's and 7B's per-row prefill
  scratch.
- **Gain:**
  - A 3,900-token pass goes 255 → 116 MB on the 1.5B and 563 → 268 MB on the 7B.
  - 2.1× the rows per MEM7 block.
  - Speed neutral to +1%.
- **Size:** M.
- **Risk:** fast-math may compile the inlined activation differently; a byte compare decides.

**KS-D6 · Grid towers allocate scratch they never read (bit-identical).**
- **Where:** `metal/grid_vision.go:gridVAccel.grow`.
- **Waste:**
  - `qh`, `kh`, `vh`, `sc`, `vt` and `ob` serve the old unfused attention, which no shipped grid tower takes.
  - `f2` serves only gated MLPs.
  - `pt` is dead after the patch embed.
  - Qwen3-VL-2B on a 4K image: 970 MB unread (35% of MEM2-G4's 2,622.5 MiB), plus 199 MB from aliasing `pt`.
- **Size:** S.

**KS-D6 gates, registered 2026-10-23 (05:20) before the code (D6-G), on the owner's plan of 5:16 am.** The first measurement
(`docs/measurements/ks-night-2026-10/README.md`) showed 918.7 of 2,622.5 MiB unread on Qwen3-VL-2B at 4K, and a NaN fill of those
buffers changed no output. The build in `gridVAccel.grow`:
- the unfused attention's `qh`, `kh`, `vh`, `sc`, `vt`, `ob` only when that attention can run (no fused kernel for the head
  dim, or the `oldAttn` seam);
- `f2` only for a gated MLP (`gvKind.gated`);
- `pt` (the patch rows, dead after the patch embed) as a view of `f1` when `f1` is at least as wide, else its own buffer.
`gridScratchAllForTest` restores main's allocation for the A/B.

| gate | tier | instrument | PASS iff |
|---|---|---|---|
| D6-G1 identity | 0 | MEM2-G1's seven tower fixtures (gemma4-vision, gemma4-audio, embeddinggemma2, qwen3.5 and glm-ocr grids, siglip, qwen2.5-vl), three inputs each, new allocation against main's in one process; `TestGridVisionMetal_matchesCPU`; S17's fused/unfused A/B (the `oldAttn` path still runs) | every output byte-equal; the grid tests pass |
| D6-G2 allocated size | 1, Mac, memory read, timing lock shared | `TestKSD6H4_gridTowerScratch` on Qwen3-VL-2B at 4K, Metal's allocated size net of the build after the forward | at most 1,520 MiB (main 2,622.5; the unread 918.7 and `pt`'s 188.4 removed), and the NaN-fill check still shows 0 outputs differing |

Planted defect: `pt` aliased to `x` (which the patch embed writes while reading `pt`). D6-G1 goes red on the grid towers. Stopping
rule: one pass each. Cost: seconds; D6-G2 about 2 minutes.

**KS-D7 · Cheaper nibble dequant in the prefill GEMM's staging.**
- **Change:** `((word >> s) & 0x000F000F) | 0x64006400` read as half2, minus half2(1032), times half2(scale). That is two
  weights in ~5 operations instead of 14.
- **Gain (est.):** 0–5% time to first token at 3900.
- **Class:** bit-identical only if an exhaustive device check passes (16 nibbles × 65,536 half scales).
- **Size:** S. Kill it under 2%.

**KS-D7 gates, registered 2026-10-23 (05:05) before the change (D7-G); the reading is in the record, and the change was not
merged.** D7-G1, model identity: llama-tiny and Qwen2.5-Coder-1.5B, a batched prefill's last logits byte-equal, old form against
new. D7-G2, speed: Qwen2.5-Coder-1.5B, a 3,900-token `PrefillLast`, two residents in one process, A B B A over 10 pairs; SHIP iff
the median new/old is ≤ 0.98 and at least 9 of 10 pairs are below 1, otherwise the row closes as measured.

**KS-D8 · Batched Mamba-2 prefill on Metal (Nemotron-H).**
- **Waste:** `metal/model.go:resident.recurrentPrefillOK` declines Mamba, so every prompt token is a decode step.
- **Gain (est.):** the DeltaNet precedent was 6.7–8.4×, so 5–8×.
- **Class:** fidelity-gated, like D-B01.
- **Size:** M–L. Pairs with KS-B12.

**KS-D9 · Batched phase 2 for the generic paged MoE prefill.**
- **Waste:** `metal/prefill_g4paged.go:resident.moeLayerMajorRows` runs experts per row and re-reads the shared expert for
  every prompt row (19.5 MB per layer on Qwen1.5-MoE).
- **Change:** reuse `moe_batch_gemv`.
- **Gain (est.):** ~1.2×.
- **Class:** bit-identical. **Size:** M.

**KS-D10 · aikit annmetal kernels are overhead-bound.**
- **Waste:** single query 0.12× the CPU at N=100k; batch about 7% of the int ALU roof.
- **Change:** lane-per-row or grid-stride.
- **Size:** S. Nothing imports annmetal today, so this waits for a consumer.

## 8. WebGPU (KS-E)

Baseline (records): the 1.5B int4 resident decode runs at 7.25 ms per token in process, with 366 dispatches recorded per
token into one compute pass. Device setup requests no optional features. `dot4I8Packed` works on the 2070 but only the
prefill GEMMs and int8 KV attention use it.

**KS-E1 · int4 decode GEMV: `nib4` with `dot4I8Packed` (bit-identical).**
- **Where:** `dot8` in `gpu/gemv_w4a8.go:gemvW4A8ShaderWGSL`; the same in `moeExpertGEMVW4WGSL` and `gemmRowsW4A8ShaderWGSL`.
- **Waste:** eight `(i32((ww >> k) & 0xFu) - 8) * a` terms per word, about 55 integer operations. It is recorded as ALU-bound
  at 250–280 GB/s against int8's ~360.
- **Change:**
  - `dot4I8Packed(nib4(ww & 0xFFFFu), alo) + dot4I8Packed(nib4(ww >> 16u), ahi)` when DP4A is present. `nib4` exists, and
    `TestW4Multi_matchesSingleRow` already holds the tiled int4 GEMM bit-equal to this GEMV.
  - Later, several rows per workgroup.
- **Gain (est.):** 0.55–1.1 ms (8–15%) on the 1.5B and 2.5–5 ms on the 7B.
- **Size:** S.
- **Risk:** on the Mac, naga's MSL `dot4I8Packed` is likely emulated, so gate per backend.
- **Prior art:** reopens the "~0 gain" verdicts, which were read on the W8A8 GEMV and on time to first token.

**KS-E2 · Widen six single-workgroup decode quantizers from 64 to 256 lanes (bit-identical).**
- **Where:** `gpu/decodefuse.go:swigluQuantWGSL`, plus `gegluQuantWGSL`, `rmsnormQuantWGSL`, `quantizeShaderWGSL`,
  `relu2QuantWGSL` and `gptossGluQuantWGSL`, each dispatched (1,1).
- **Waste:** `swigluQuant` measures 30.3 µs for 143 KB.
- **Change:** 256 lanes with the same per-element expressions. f32 max is exact in any order, and the norm's sum stays on
  lanes 0–63 in today's order.
- **Gain (est.):** about 0.78 ms of the 7.25 ms token on the 1.5B (G35's measured per-class costs), 1.7 ms on the 7B.
- **Size:** S.
- **Prior art:** extends G35.

**KS-E3 · Encode token t+1 while the GPU runs token t (bit-identical).**
- **Where:** `gpu/decoderunner.go:DecodeRunner.Run`.
- **Waste:** recording starts only after the previous token's Poll, so about 0.62 ms of CPU encode is exposed per token.
- **Change:** record after the submit. Drop the pending buffer on `SetAdapter`, `UseKVSlot` or a mode switch.
- **Gain (est.):** ~0.6 ms (~8%). Metal's version measured +4.4%.
- **Size:** S–M.

**KS-E4 · int4 prefill GEMM (`gpu/gemm_w4a8.go:matmulTiledW4A8KernelWGSL`; bit-identical).**
- **Waste:** 1.18–1.29 ms per row (~2.1 TOPS), against the int8 RB64 kernel's 0.377 on the same tile. `BK = 8` means two
  barriers per 128 `dot4`, plus 16 int-to-float conversions per group per thread.
- **Change:** two groups per barrier pair, and the exact magic-number conversion.
- **Gain (est.):** 1.4–2× on projections: a 1,783-row pass 2.78 → 1.7–2.2 s.
- **Size:** M.

**KS-E5 · Decode attention at depth.**
- **Where:** `gpu/attention.go:attnKeysShaderWGSL`.
- **Waste:** only nH workgroups, with a serial V-sum. 1.14 µs per position of context on the 1.5B, against CUDA's 0.924 and
  Ollama's 0.168.
- **Change:** (a) a bit-identical V-load unroll; (b) split-KV, fidelity-gated, like CUDA's parked V-sum split.
- **Gain (est.):** (b) the 1.5B at 16k 26.1 → 12.4 ms per token.

**KS-E6 · Prefill attention with several rows per workgroup.**
- **Where:** `gpu/prefillrunner.go:attnKeysBatchedShaderWGSL`.
- **Change:** each K/V load serves R rows. This needs the workgroup storage limit raised (KS-E9).
- **Gain (est.):** 22–30% at 4,096 rows.
- **Class:** bit-identical. **Size:** L.

**KS-E7 · A KV ring for sliding-window layers (bit-identical).**
- **Where:** `gpu/residency.go:webgpuBackend.BuildResident` allocates every layer at the full context.
- **Gain:** per slot at 16,384 positions: Gemma 3 4B 4.25 → 0.97 GiB, Mistral-7B v0.1 4.0 → 1.13 GiB.
- **Size:** M.
- **Prior art:** precedent in `cuda/kvwindow.go`.

**KS-E8 · LoRA: the P-10 and P-11 fixes CUDA and Metal got (bit-identical).**
- **Waste:** `loraDeltaDownWGSL` runs on a (1,1) grid (`gpu/lora_resident.go:DecodeRunner.rebuildSteps`), and every bind
  re-uploads A and B.
- **Gain:** on CUDA these fixes took adapter overhead from 122% to 42% of a token.
- **Size:** S.

**KS-E9 · Request the features the adapter offers.**
- **Where:** `gpu/gpu.go:New`.
- **Change:** request timestamp queries (per-pass GPU time for every item here), subgroups where exposed, and a larger
  workgroup storage limit.
- **Size:** S. Mainly an enabler.

**KS-E10 · Fold the int4 q/k/v bias into `qkvFinalize`.**
- **Gain (est.):** 84 fewer dispatches, ~0.15 ms (~2%).
- **Class:** bit-identical without an adapter.

**KS-E11 · MLA decode attention (`gpu/mla.go:mlaAttnWGSL`).**
- **Waste:** about 9 barriers per key.
- **Change:** port G36's key split.
- **Class:** fidelity-gated. **Size:** M.

## 9. CPU kernels (KS-F)

**What limits x86 decode, by arithmetic** (bytes per token at the fitted 26.03 GB/s against the served time):
- About 6.2 ms (0.5B) and 3.9 ms (1.5B) of each token stream nothing; the 7B is bandwidth-bound.
- A 0.5B token has 73 fork/joins and a 1.5B token 141, each 25–70 µs by the records (the τ fit, R-06).
- aikit's `parallelSpawnCols` (`linalg/linalg.go`) starts fresh goroutines per call, and the caller waits without
  working.
- The kernels are not the limit (gate+up streams at 86% of the all-thread rate).
- ggml's persistent spinning threads (from memory) pay about 1 µs per barrier.

**KS-F1 · A MoE layer's experts in one or two fork/joins instead of three per expert (bit-identical).**
- **Where:** `decoder/mlp.go:swiGLUExpert` (gate, up and down each a separate fan-out), `decoder/mlp.go:moeMLPWith`,
  `decoder/forward_gptoss.go:Model.gptOssMoEWith` and `decoder/forward_gemma4_moe.go:gemma4MoEFFNWith`.
- **Waste:**
  - **M35:** each expert matmul is exactly `int4ParThreshold` MACs, a 0.59 MB stream behind a 25–70 µs barrier, so
    1,080 fork/joins per token.
  - **gpt-oss:** a serial f64 activation, two allocations per expert, and h re-quantized for gate and up.
- **Change:** quantize h once per layer; one fan-out over (expert, column chunk) of gate‖up with the activation inside the
  worker; one fan-out over the down projections; accumulate in routing order.
- **Gain (est.):** M35's CPU fallback 1.12–1.41× (25–70 ms of 239); gpt-oss 1.09–1.20×.
- **Size:** M.

**KS-F2 · gpt-oss and `attendQueryP` attention: one thread, one scalar chain per key.**
- **Where:** `decoder/forward_gptoss.go:attendSinksToken` (`scores := make(...)` per call, `for qh := range nH`, one f64 dot
  chain per key) and `decoder/attention.go:attendQueryP` (Qwen3.5/3.6 attention layers, Granite-4.0-H, Nemotron-H, LFM2,
  Llama 4).
- **Waste:** gpt-oss spends ~15–17 ms of a 133 ms token at depth 128 and ~140 ms at 2048. M35 spends ~190 ms at 4096.
- **Change, bit-identical:**
  - Heads fanned out over `headWorkerScratch`.
  - 8 keys scored at once as 8 independent f64 chains, keeping the f64 scale.
  - AV vectorized with each arch's own contraction.
  - Scores from `decodeScratch`.
- **Step 2, memory:** read gpt-oss's 12 window-128 layers through the KV ring. Today they keep append-forever f32 KV:
  49,152 B per position, 1.6 GB at 32k and 6.4 GB at 131k, to serve a 6.3 MB window (§3, note 1).
- **Gain (est.):** 1.12× at depth 128 on gpt-oss (about 0.87–0.96× of Ollama with KS-F1), ~1.5× at 2k.
- **Size:** S–M.
- **Option, changes output:** KS-O4.

**KS-F3 · DeltaNet, Mamba-2 and KDA mixer cores run serially (bit-identical; reopens the parked CPU recurrence task).**
- **Where:** `decoder/deltanet.go:deltaNetRecurrence` (`for headV := range nv` on one goroutine), `decoder/deltanet.go:deltaNetCore`,
  `decoder/mamba2.go:mamba2Step` and `decoder/kda.go:kdaMixerStep`.
- **Why reopen:**
  - Gate D0 measured ~73 ms per token in this core and parked it at 8% of a ~620 ms token.
  - M35's CPU token is now 239 ms, so at the old rate this core is ~30%.
  - The threading half is S-sized and goinfer-only.
- **Change:** fan out over value heads, which have disjoint state; the same for the gated norm and the conv channels.
- **Gain (est.):** 1.2–1.35× on M35's CPU fallback.
- **First measurement:** `GOINFER_DELTANET_TIMING=1` on today's tree.

**KS-F4 · A decode-scoped worker team on amd64 (reopens aikit's killed worker pool, measure first).**
- **What changed since the kill:**
  - The serial stretches that sank it roughly halved: SwiGLU moved into the workers (09-23), and the activation quantizer
    went AVX2 (09-29).
  - Its 15 spinners always shared the caller's SMT sibling.
  - The τ fit and R-06 came later.
- **Steps:**
  1. Per-fan-out timestamps inside a real 0.5B token on amd64 (the S-02 harness never ran there).
  2. If wake latency dominates: ≤7 workers plus the caller, one per other physical core, spinning ≤20 µs inside a forward
     pass. Lower the threshold so the 0.5B's q/k/v and o fan out too (they run serial today, under `int4ParThreshold`).
  3. If the tail dominates: dynamic chunks in `decoder/cpu_gateup_fused.go:gatedMLPFusedGateUpAct`.
- **Gain (est.):** if the barrier cost halves, the 0.5B 1.18–1.26× and the 1.5B 1.05×.
- **Class:** bit-identical. **Size:** L; high risk.

**KS-F5 · MLA decode re-normalizes and re-ropes every cached latent, every token (bit-identical).**
- **Where:** `decoder/forward_deepseek.go:Model.mlaAttentionAbsorb`. `AppendLatent` stores the raw kv_a output, so each token
  redoes a 512-wide RMSNorm and 64 trig calls per cached position per layer, on one thread.
- **Waste:**
  - The f32 projections fall under `linalg`'s 1<<24 threshold and run single-threaded.
  - V2-Lite: ~37 ms per token at depth 128, ~290 ms at 1024.
- **Change:** store the normalized, roped latent at append, as the WebGPU and CUDA paths already do; score with heads in
  lanes and keys split across workers.
- **Gain (est.):** ~1.25× at depth 128, ~2.5–3× at 1024.
- **Size:** S–M.

**KS-F6 · An AVX2 port of the grouped acc64 attention, and the head-worker width.**
- **Waste:**
  - aikit's `linalg/attn_acc64_group_other.go` returns 0 blocks off arm64, so amd64 widens K/V once per query head.
  - `decoder/scratch.go:maxAttnWorkers` is 6, the M1 Pro's P-core count, applied unchanged on the 8-core Ryzen.
- **Gain (est.):** 1.5B at 4k: attention 42 → 23–28 ms (1.15–1.2× on the token).
- **Class:** bit-identical. **Size:** M + S.
- **Prior art:** extends R13.

**KS-F7 · phi3 at q4k on CPU: gate and up as one fan-out with SwiGLU inside (bit-identical).**
- **Waste:** `decoder/mlp.go:gatedMLPAct` runs gate and up as two fan-outs, each quantizing h, then a serial 8192-element f64
  SwiGLU (~145 µs per layer).
- **Change:** an aikit pre-quantized entry for `matmulQ4K`.
- **Gain (est.):** 1.04–1.055×, phi3 q4k from 0.763 to about 0.80 of Ollama.
- **Size:** S + S.

## 10. aikit: retrieval, encoders and towers (KS-G)

**KS-G1 · The Qwen3-ASR and Gemma 4 audio encoders: the serial work is most of the wall time (reopens S14.6).**
- **Where:** aikit `audio/qwen3asr_encoder.go` and `audio/gemma4_encoder.go`.
- **Waste:**
  - Attention is a scalar loop per (head, query, key) with float64 `math.Exp`.
  - GELU uses float64 `math.Erf`, about 29M calls per 10 s clip.
  - LayerNorm is float64.
  - Bias adds are serial.
  - Only `MatmulBT` runs on more than one core.
- **Why S14.6 misread it:** it called these "matmul-bound" from CPU-sample share. Solving wall = serial + parallel ÷ cores
  puts the serial share at 70–83% of Qwen3-ASR's wall time and 60–69% of Gemma 4 audio's.
- **Change:**
  - Step A: split the passes by rows and the attention by (head, query), with the math unchanged (bit-identical).
  - Step B: S14.5's contract kernels, fidelity-gated by the existing cosine gates.
- **Gain (est.):** Qwen3-ASR 1.72 s → ~0.5–0.75 s per 10 s window after step A; Gemma 4 audio 1.92 → ~0.65–1.0 s.
- **Size:** S (A), M (B).
- **First measurement:** the 10 s window at `GOMAXPROCS=1` against the default. Matmul-bound shows ≥5×;
  serial-dominated about 2.2×.

**KS-G4 · Gemma 4 image preprocessing runs on one goroutine, on every backend (bit-identical).**
- **Where:** aikit `vision/gemma4_preprocess.go`: `toNRGBA` (a whole-image `draw.Draw`), a per-pixel copy, then a
  single-goroutine `resizePass` (`vision/bicubic.go`).
- **Waste:** ~0.27–0.38 s per 12 MP photo, ~50–75 ms per 1080p (counted). The July campaign removed `toNRGBA` from
  `vision.Preprocess` only.
- **Change:** direct YCbCr → RGB by rows (item 32's proven method) and a row-split resize.
- **Gain (est.):** ~0.05 s per 12 MP photo. Also helps the LFM2-VL and Pixtral resizes.
- **Size:** S.

**KS-G3 · `FlatI8` selects its top-k on one goroutine after a parallel scan (bit-identical).**
- **Where:** aikit `ann/flat_i8.go` (`topHits`) and the per-query loop in `ann/backend.go`.
- **Waste:** `QueryBatch` CPU is flat at 5.3k queries/s for batch 64 and 256 at N=100k; the serial selection is about 7.7 of
  each 12.1 ms batch (counted).
- **Change:** per-worker top-k with an exact merge, as `Flat` does.
- **Gain (est.):** `QueryBatch` ~2.4×; a single query 15–20% at 100k.
- **Size:** S.

**KS-G2 · int8 HNSW builds never use the 8-row int8 kernel (bit-identical).**
- **Where:** `simIDsInto` in aikit `ann/hnsw_select.go` guards `if h.scoreUnbatched || h.int8 || ...` and sends int8
  similarities to per-pair `DotI8`.
- **Change:**
  - Route int8 through `scoreInto` with `DotI8x8` (exact integer arithmetic, same float64 order).
  - Step 2: an arm64 SDOT 8-row kernel. Today `dotI8x8` on arm64 is eight `dotI8` calls, so M-21's 1.42× never reached the
    Mac.
- **Gain (est.):** ~15% on amd64 int8 builds.
- **Size:** S, then M.

**KS-G5 · The CPU vision towers' bias, residual, clamp and RoPE passes are serial (bit-identical).**
- **Waste:** Qwen2.5-VL ~0.1–0.15 s of a 0.55–1.07 s forward; SigLIP-so400m ~0.7 s of ~8 s. Gemma 4's tower makes about 3 GB
  of fresh copies per image.
- **Gain (est.):** 5–15% of CPU tower time. Matters on the CPU backend and fallback only.
- **Size:** S–M.

**KS-G6 · The WordPiece cache takes a read lock per word (bit-identical).**
- **Where:** `embed/tokenize.go`.
- **Evidence:** dead-ends §3.1 measured a sharded read-write lock 30.9% slower under `EncodeBatch` with 8 workers.
- **Change:** a lock-free read path.
- **Gain:** unknown, perhaps 4–8%.
- **Size:** S.

**KS-G7 · BM25 postings store tf as int32 (memory).**
- **Waste:** `bm25/index.go`'s posting is 8 bytes; at 1M chunks that is 888 MB of postings.
- **Change:** separate arrays, tf as uint8 with an exception table. Needs a v2 format.
- **Gain (est.):** −333 MB, about 9% of ken's index footprint at 1M chunks.
- **Size:** M.

**KS-G8 · HNSW queries have only been measured warm.**
- **Waste:** every benchmark repeats one query over 50k vectors, and the prefetch trial was warm too.
- **Change:** benchmark cold first, then prefetch if cold is ≥1.5× slower.
- **Size:** S, then M.

**KS-G9 · The binary prefilter's final pass over all distances is serial.**
- **Waste:** ~70–80 µs of a 578 µs query at 200k (~12%).
- **Change:** per-worker collection, keeping the order.
- **Class:** bit-identical. **Size:** S.

**KS-G10 · Block-max WAND for BM25.**
- **Change:** per-128-posting bounds, ~0.8% extra.
- **Gain:** speculative; the literature (from memory) reports 2–3× over plain WAND.
- **Class:** bit-identical. **Size:** M–L.
- **First measurement:** an offline simulation counting documents scored.

## 11. Memory the kernels need, all backends (KS-H)

**KS-H1 · Metal: block the exact-attention and E-model prompts like the plain pass (bit-identical at head dim ≤256).**
- **Where:** `metal/backend.go:metalResident.PrefillBlockRows` returns 0 for `prefillExactAttn()` and `emodelLayerMajor()`,
  so those prompts embed and prefill in one pass. Admission then prices them with the CPU formula
  (`decoder/prefill_budget.go:prefillScratchBytes`).
- **Why the exclusion is stale:** it rests on the exact kernel declining past 4,096 keys. The tiled twin removed that decline.
- **Row scratch per row** (`metal/prefill.go:resident.prefillScratchBytes`): Qwen3.5-9B 254,468 B, E2B 134,660, Gemma-3-4B
  83,972, gemma3-1b 51,204.
- **What it costs:**
  - Qwen3.5-9B at 20,189 tokens: 4.78 GiB of transient scratch, priced at 2.10 GiB.
  - E2B at 20,975 tokens: 2.63 GiB, priced at 1.21.
  - That is a never-swap exposure on the 16 GB Mac.
- **Change:**
  - Drop the exclusion for head dim ≤512 and block like the plain pass.
  - Key the tiled-kernel choice on the whole prompt's length, so every block takes the kernel the whole pass would.
  - Admit E-models after their own chunk-invariance gate.
- **Gain:**
  - About 0.5 GiB of scratch per pass: −4.3 GiB on Qwen3.5-9B at 20k, −2.1 on E2B, −1.1 on Gemma-3-4B, −1.06 on gemma3-1b
    at 32k.
  - Admission becomes exact.
  - The host embeddings shrink with the blocks (Qwen3.5-9B 316 → 32 MiB).
- **Class:** bit-identical at head dim ≤256 by existing tests (`TestAttentionPrefillTiled_bitIdenticalToTheExactKernel`,
  `TestMC5_prefillChunkInvariance`, `TestDB01_chunkedPrefillMatchesWhole`); at 512 once the tile choice keys on the prompt
  length.
- **Size:** S; S–M with E-models.
- **First measurement:** tier 0 on `llama-tiny` with the exact kernel forced and small block and tile sizes. Blocked against
  whole must be byte-equal, and the scratch seen must stay at or under the block's figure. Then the qwen35 and E-model tiny
  fixtures, then a 20k-token prompt on Qwen3.5-9B under `mem0_darwin.py`.
- **Prior art:** new. I8's "what this leaves" listed these routes, and no ledger row owns them.

**KS-H1 gates, registered 2026-10-22 before the code (KH1-G).** Re-checked on main 0e620108 (after P3b-2 and MEM7): the
finding holds. `PrefillBlockRows` still returns 0 for `prefillExactAttn()` and `emodelLayerMajor()`, the tile choice keys on
each pass's own end (`exactAttnTile(startPos+M)`), and such a resident is still priced by the CPU formula.

**The build.**
- `metal/backend.go:metalResident.PrefillBlockRows` stops excluding exact-attention models and Gemma 4 E-models whose widest
  exact-attention head dim is at most 256. There the exact and tiled kernels are bit-identical, so a block may take either.
- The decoder then splits their prompts in blocks, and admission prices one pass (`residentPrefillScratch`).
- A model whose exact-attention head dim is 512 (Gemma 4's global layers) stays unblocked. Blocking it needs the tile choice
  keyed on the whole prompt's length, which the resident does not know today; asked as a decision, not built.

| gate | tier | instrument | PASS iff |
|---|---|---|---|
| KH1-G1 identity, llama | 0, Mac | `llama-tiny`, Metal, the exact kernel forced (`GOINFER_METAL_FUSED_ATTENTION=0`), a small tile (`exactAttnTileForTest` 32): a 300-row prompt prefilled whole against in 64-row blocks; and each block pass's scratch (`prefillScratchSeen`) against `PrefillScratchBytes` of its rows | last logits and every layer's K/V bytes byte-equal; each pass's scratch at or under its figure |
| KH1-G2 identity, qwen35 and E-model | 0, Mac | `qwen35-tiny` and `gemma4-emodel-tiny` (window raised in a copy, MEM6's helper), Metal, the batched route: a 600-row prompt whole against in blocks of 128 | byte-equal, as KH1-G1 |
| KH1-G3 admission | 0 | `Model.residentPrefillScratch` for a 20,000-token prompt on the exact-forced `llama-tiny` resident | reports a blocked price: `PrefillScratchBytes` of the longest pass plus its embeddings, not the CPU formula (main: not ok, so the CPU formula) |
| KH1-G4 unchanged | 0 | `touched.py`; the Metal prefill, MEM7 (BLK), I8 and MC5 tests; a fused-attention model's `PrefillBlockRows` and logits | unchanged |
| KH1-G5 measure | 1, Mac, by day | Qwen3.5-9B (`~/models`, its Metal sidecar), a 20K-token prompt, transient prefill scratch and RSS under `mem0_darwin.py`, main against the branch | read, not gated; the survey's estimate is 4.8 → about 0.5 GiB. Run only after its fit check passes on the day |

Planted defect, run once on the committed tree and reverted: an exact-attention block pass attending from position 0 instead of its
start (`uStartPos` 0; G1 red). Stopping rule: one pass each. Cost: G1–G4 seconds; G5 about 10 minutes.

**KS-H2 · CUDA: suballocate weights, KV pools and expert slots from arenas (bit-identical).**
- **Where:** one `cuMemAlloc` per weight buffer (`cuda/resident.go:cudaResident.upW`, `cuda/pack_stream.go`), per layer's
  K and V pool (`cuda/kvpool.go:cudaResident.kvAllocPools`), and four per MoE layer
  (`cuda/resident.go:cudaResident.allocSlots`). Each rounds up to `cuda/resident.go:allocQuantumBytes`, 2 MiB.
- **Waste:**
  - The weights' rounding is measured (G-M1): 139 / 122 / 406 / 257 / 136 / 142 MiB on the 0.5B / 1.5B / 7B /
    Gemma-3-4B / Qwen2.5-VL-3B / E2B. It is priced, not removed.
  - The KV pools' rounding is unpriced: Qwen3-8B at 16k, 1 slot, 135 MiB; the 7B at 32k 108.5 MiB.
  - The 26B's expert slots: ~83 MiB.
- **Change:** the plan knows every size before the first allocation. Use one arena for the weights, one per KV group and one
  for the slots, with views.
- **Gain:**
  - The 7B: +406 MiB, about +7,400 f16 positions on the 8 GB card.
  - Qwen3-8B at 16k: +110 MiB (about +780 positions), and 135 MiB of the 384 MiB margin becomes real.
  - The 26B: +0.9 slots per layer.
- **Size:** M.
- **Risk:** an out-of-bounds read lands in a neighbor, not in slack; run compute-sanitizer once.
- **First measurement:** `cuMemGetInfo` around the 7B build (expect −406 MiB); free VRAM after `kvAllocPools` on Qwen3-8B at
  16k (expect +135 MiB today).

**KS-H3 · CUDA: move `moe_route`'s and `route_gptoss`'s per-thread arrays to shared memory (bit-identical).**
- **Waste:**
  - `float score[512]; float sel[512];` plus two 64-entry arrays (4,416 B per thread) in a kernel that runs one thread.
  - That sizes the device-wide local-memory pool: `TestMoERouteFirstLaunchReservation` measured 138,412,032 B retained,
    with a 289 MB peak demand at first launch.
- **Change:** `__shared__` arrays (4 KB static); regenerate `moe.ptx` per `cuda/testdata/REGEN.md`.
- **Gain:**
  - −131 MiB on every CUDA MoE load, and −289 MB of transient demand.
  - The 26B gets +1.37 expert slots per layer.
- **Size:** S.
- **First measurement:** `TestKernelLocalMemoryCensus` reads 0 B for `moe_route`, and the reservation falls to ~1.3 MiB.
- **Prior art:** new. The cost was recorded as "the price of the router cap".

**KS-H4 · Metal towers keep their largest image's scratch until unload (bit-identical).**
- **Where:** `metal/grid_vision.go:gridVAccel.grow` only grows; the same holds in the other tower accelerators and aikit's
  `qwenmetal` / `visionmetal`.
- **Waste:** MEM2-G4: Qwen3-VL-2B holds 2,622.5 MiB after a 4K forward until `Close`.
- **Change:** release above a ceiling after the forward, or MEM2's optional patch ceiling.
- **Gain:** up to 2.6 GiB back on the 16 GB Mac.
- **Size:** S.

**KS-H5 · Buffers only speculative verify reads, allocated at build (bit-identical).**
- **Waste:**
  - CUDA's `faBuf` (`cuda/flash_decode.go:cudaResident.loadFlashDecode`, sized by `cuda/flash_decode.go:faMaxRows` = 16
    rows): 31 MiB unused at M=1 on Qwen3-8B (≈220 positions), 27 MiB on the 7B.
  - phi3-mini: 26.2 MiB that can never be used (head dim 96 is never eligible).
  - Metal's `metal/model.go:resident.ensureBatchCap(16)` at build: 9.3–16 MiB. A growth strands the old pair until `Close`.
- **Change:** allocate the multi-row part on first verify, or when speculation or a drafter is on; skip it when no layer is
  eligible; release on growth.
- **Size:** S.

**KS-H6 · Re-derive the two CUDA 384 MiB margins (needs KS-O3).**
- The margins can be rebuilt from measured parts after KS-H2 and KS-H3.
- **First measurement:** minimum free VRAM over a 20k prompt on Qwen3-8B at 16k and on the 26B.

## 12. Already covered (not re-proposed)

**CUDA**
- Graphs: 1.5B 1.01×, the 26B 1.339×; opt-in by tenancy.
- The cooperative megakernel: rejected.
- K2 (`quant_vec` into the o-projection): ~0%.
- `glu_quant` at 1024 threads: shipped.
- Rows-per-warp, the wave rule and the 4× unroll: 2026-09-21 record.
- The warp-parallel per-32 quantizer: 2% slower.
- Fused per-32 kernels above 1536: off by measurement.
- Narrow-row q4k: killed.
- The flash lane below 2048 keys: measured slower.
- Split-KV thresholds: their own campaign.
- The BM=32 tile: killed (2.86× slower).
- Fused attention at 256/512: P35 option C and P32.
- The fast GEMM at depth on Gemma 3: KV3 option (b), not chosen.
- The fast GEMM for DeltaNet: rejected on fidelity.
- Chunkwise (WY) DeltaNet: fidelity class, and the recurrence is 2%.
- Expert-major and option A on the generic path: shipped.
- The `dScrB` alias and phase aliasing: KV3.
- The kv16 V-sum: fixed (`9a1a0f7e`).
- Tower attention: S17 lever A.

**Metal**
- The int4 LM head: MEM8(a)/D5, parked.
- The down projection to SA order: B-P05 = O4, open.
- SA lane balance: B-P04, killed.
- Rows-kernel geometry: B-P06, killed.
- The f32 and masked SA forms: R18/R18b shipped; R1 measured +2–4%.
- `rmsnorm_quant` register cache: B-P09.
- Concurrent dispatch, untracked buffers, ICB, unretained references, megakernel and Stage B: recorded negatives.
- Device top-k: E-P04 = O6.
- Decode attention generality: B-P01..B-P03 done; B-P07 open; B-P08 parked.
- MC3 step kernels: E-P02/E-P05/E-P08 shipped; E-P03 killed.
- Prefill GEMM work: A-P01/D-B02 shipped; split-K not recommended; double-buffering lost 12%.
- R19 steel at 128: shipped; A-P03 parked.
- D-B01: shipped. D-B05: waits on O2.
- f16 tower activations and the 32×32 f32 tile: failed.

**WebGPU**
- Device argmax: G38.
- Megakernel and K1 fusion: G35, no-go.
- The `quantize` serial reduce: fixed.
- Per-key attention reduction: G36, S12.4c.
- The qk-norm fold: −2.7%, reverted.
- The W8A8 bias fold and qkvFinalize fusion: shipped.
- Merging independent GEMVs: +1.5%.
- RunN fence amortization: ≤4%.
- f16/int8 KV: KV precision P5.
- RB64: shipped.
- The W8A8 decode GEMV with dot4: at its roof.
- Vision towers: S12.3.
- Paging: KV6/D3.

**CPU**
- 2Acc/4Acc W4A8: noise.
- Split-half AVX2: +2.1%, parked off.
- VPMADDUBSW W4A8: not built; gate+up is already at 86%.
- VNNI: no box.
- S-05 fold, quantizers, acc64 kernels, R-06, fused gate+up, f16 scales, R-13/R-14: shipped.
- Activation fan-out on amd64: off by measurement.
- SMT, prefetch and page size: dead.
- A native Q6_K CPU kernel: a dead end.
- Native MXFP4: filed (task-fp4-formats A).
- CPU f16 KV way B: P27, parked.
- f32 islands: MEM11.
- GC headroom: MEM12.
- Gemma 4 decode re-gather: P-09.

**aikit retrieval**
- Similarity hoists, the `topk.Push` check and the BM25 impact precompute.
- Sparse WAND and MaxScore.
- HNSW heap pooling.
- MEM14–MEM16.
- Model2Vec presum and gather.
- The WordPiece arena and the tokenizer interner.
- CPU FlashAttention: 8.6–10× slower.
- SIMD layerNorm: rejected.
- M-18..M-21: shipped.
- Overlapping ANN and BM25 queries: slower.
- MEM17 and MEM4: owned.

**Memory**
- Windowed layers at full context on Metal and CUDA: KV5.
- Slots × context: KV4.
- f32 KV families on CUDA: KV4b and the precision plan.
- int8 KV on CUDA: B39 / precision P2.
- The drafter forcing f32: precision P6.
- The 512 MiB block: MEM7 (merged since this survey read).
- C′ pinned stacks: MEM10.
- Fixed expert slots: KV11.
- MLA f32 latents: MEM11.
- CPU rings: MEM12-ringres.
- Host embeddings: MEM6.
- Tower host copies: MEM4.
- KV7 snapshots and KV8a checkpoints: by design.
- WebGPU whole-runner slots: KV6.

## 13. Checked, nothing to gain

These read at or near their roof:
- **CUDA:**
  - aikit's int8 head GEMV `gemv_w8a8_fwd`: ~420 GB/s.
  - The fused gate/up decode GEMVs at K ≥ 3072: 355–368 GB/s.
  - Prefill glue (norms, residual, quantizers): ~0.022 ms per token, near DRAM.
  - The DeltaNet recurrence: 2% of prefill.
- **Metal:**
  - The int8 head `gemv_w8a8_coal`: 161–163 GB/s against a 178–182 ceiling.
  - The prefill GEMM: 81–88% of MPS f16; KS-D7 aside.
  - Steel attention at 128: 2.1 TFLOPS.
  - Prefill norms, RoPE and KV store: ≤0.2% of a pass.
  - MoE routing, gather and scatter: <1% of a layer.
- **WebGPU:**
  - The W8A8 decode GEMV: ~83% of bandwidth.
  - RB64: ~6.9 TOPS.
- **CPU (aikit):**
  - The W4A8 M=1 kernels: 91–97% of the per-core issue ceiling.
  - `dotQ4KAVX2`: ~17.6 GB/s per thread.
  - The f32 dots: ~95% of FMLA peak on the M1.
  - The acc64 kernels and the activation quantizers.
- **aikit retrieval:**
  - anncuda single query: 384 of 412 GB/s.
  - Hamming on arm64: 52–55 GiB/s.
  - The BM25 term-at-a-time accumulator.
  - `Flat` (already the fused shape).
  - The audio FFT front ends: ~0.1 s per 30 s window.
