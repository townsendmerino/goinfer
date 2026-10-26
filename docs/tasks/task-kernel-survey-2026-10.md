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
| KS-A1 | done (GO) | done 2026-10-24 (A1-G0): a prototype Q5_K GEMV reaches **0.985×** `gemv_w8a8_g32`'s GB/s at 9216×3072 on the 2070 SUPER (bar 0.95), 0.697× its time summed over Phi-3; Q6_K 0.832×. The build is KS-O1 on CUDA. Tier 0 read 2026-10-23: the q4k plan reads 2,710.96 MB a token against the file's 2,336.28 | survey, "A1-G0 read" |
| KS-A2 | done | done 2026-10-23 (#191, aikit `gpu/v0.34.0`): the CUDA resident launches from its own executor thread (aikit `Queue.LaunchDirect`), byte-identical. The 0.5B at depth 128 **1.492×** median, 10 of 10 ABBA quads (345 → 521 tok/s); gemma3-1b 1.465× (197 → 291); 10.4 → 2.3 µs a launch. Unblocks KS-A5 (decode now GPU-bound) | `docs/measurements/ks-a2-2026-10/README.md` |
| KS-A3 | done | done 2026-10-23 (#195): the top-K as `topk_compact` (32 blocks) + `topk_finish` (select K, sort K) + `topk_z` (32 blocks, same partials); every output word equal to the one-block kernel on 608 cases; a sampled 0.5B token **1.0263×** median, 10 of 10 ABBA quads (2,007.5 → 1,955.4 µs); +36.9 KB device scratch | `docs/measurements/ks-a3-2026-10/README.md` |
| KS-A4 | open | nobara: launch counter (tier 0), then an in-process A/B on gemma3-1b | — |
| KS-A5 | killed | killed 2026-10-23 by its pre-registered bands: one-round row loads plus a next-row prefetch in `fused_rms_qkv` / `fused_rms_gu` (code at 6ababd5e3) were bit-identical but made 0.5B decode **4.5% slower** (median 0.9548×, 0 of 10 ABBA quads; 523.7 → 499.9 tok/s). gemma3-1b and the 1.5B dispatch the unchanged rows twin. Cause not read (registers unread); a no-prefetch variant would need its own registration | `docs/measurements/ks-a5-2026-10/README.md` |
| KS-A6 | open | nobara: host timestamps around upload, readback and hand-off | — |
| KS-A7 | open | nobara: launch counter, then an A/B on phi3-mini | — |
| KS-A8 | open | nobara: an A/B after A2 reads | — |
| KS-B1 | open | nobara: ncu on the gate and k launches at M=512; the int-to-float swap alone, bit-gated | — |
| KS-B2 | done | done 2026-10-23 (#192): 16-byte staging loads in the six `attn_fused_bm*` entry points, byte-identical (1,080 shapes a KV type old = new; the 456-shape sweep; 3,900-token prefill logits). The 1.5B's 3,900-token prefill **1.263×** at f32 (937 → 742 ms) and **1.275×** at f16, 10 of 10 ABBA quads each; attention 2.44× / 2.79× | `docs/measurements/ks-b2-2026-10/README.md` |
| KS-B3 | open | nobara: MoE category and launch count on M26 at M=512 | — |
| KS-B4 | open | nobara: attention/GEMV/recurrence split on Qwen3.5-9B at 8k and 16k | — |
| KS-B5 | open | nobara: MoE category and cache misses on M35 and gpt-oss-20b | — |
| KS-B6 | open | nobara: served TTFT for phi3-mini at q4k against Ollama | — |
| KS-B7 | open | nobara: a prototype against `gemv_w4a8_rn` at the 1.5B gate shape | — |
| KS-B8 | open | nobara-ks-cuda, after #230 merges: 2,048-row CUDA prefill passes by default where scratch allows, priced by admission (decided A, Francis, 2026-10-25). Measured 2026-10-23 on the 1.5B at K=3,900: attention **1.556×** at 2,048 rows (1.231× at 1,024), whole prefill **1.247×** (933 → 748 ms) | `docs/measurements/kernel-survey-2026-10/README.md` |
| KS-B9 | open | nobara: launch and grid counters, then the GEMV-category A/B | — |
| KS-B10 | open | nobara: sync and launch counters in the generic expert-major Phase 1 | — |
| KS-B11 | open | nobara: `PrefillScratchBytes` delta, then a glue+GEMM A/B | — |
| KS-B12 | open | nobara: per-token TTFT on Nemotron-H as the baseline | — |
| KS-B13 | open | aikit: a microbenchmark at SigLIP shapes, bit-gated | — |
| KS-B14 | open | nobara: host round trips per image (55 → 0), then a tower A/B | — |
| KS-B15 | done: KILL | nobara-ks-cuda: the 64 KB shared-memory opt-in on the exact attention kernels (bound 12,160 → 16,256 keys) is bit-identical but slower: a 14,336-key pass ran 0.9715× of today's row-batched route, 0 of 10 pairs faster ("KS-B15 build read"); the opt-in stays unset | — |
| KS-C1 | done (a)+(b) | done 2026-10-24: the q4k lane's GEMVs on `gemv_q4k_f32_rows<8>` and `gemv_w8_f32_rows<2>`; the Q4_K GEMVs 1.75–2.03× in isolation (C1-G0); G-Q1 and the step identity PASS (C1-G1, G2); G-Q2 98.66% against the reference's 93.99% (C1-G3 PASS); Phi-3 mini q4k decode 22.3 → 33.0 tok/s, median 0.6741, 10 of 10 pairs (C1-G4 SHIP). (c) done on Metal as KS-O1 (#215; the head stays int8); its CUDA half is KS-A1 | survey, "KS-C1 read" |
| KS-C2 | done | done 2026-10-23: the staged coal kernels (the int4 decode down projection; DeltaNet's coal projections) take each word's sum in R18b's masked form; bit-identical (C2b-G1); the down GEMV 1.48× in sequence; the 1.5B's int4 Metal decode 10.75 → 10.02 ms a token, median 0.9332, 10 of 10 (C2b-G2 SHIP) | survey, "KS-C2 wiring read" |
| KS-C3 | done | done 2026-10-23: a native int8 resident's dense decode GEMVs take int8 twins of R18's rows kernels; bit-identical (C3-G1); the int8 body at the 1.5B's shapes 135 → 157 GB/s (C3-G2); the 1.5B at int8int8 decodes 13.83 → 12.60 ms a token, median 0.9109, 10 of 10 pairs (C3-G3b SHIP). The 0.5B's C3-G3 did not resolve (median 1.0021) | this doc, "KS-C3 read" |
| KS-C4 | parked | parked 2026-10-24: folding `swiglu_quant` into the gate\|up epilogue and the down staging is not bit-identical on real data (a rounding tie at fast math: one of 4,864 values in one layer, C4-G1 (b) red) and is 4.9% slower (median 1.0489, 1 of 10). Code on branch `ksc4-fold-parked-2026-10-24`. The quantizers are 0.875 ms a token; a multi-threadgroup quantizer is the untried shape | survey, "KS-C4 read" |
| KS-C5 | parked | parked 2026-10-24: 16-wide V loads and shuffle reductions in the legacy decode attention, bit-identical (C5-G1 PASS); the 1.5B at 385–512 keys median 1.0108, 4 of 10 pairs (C5-G2), with both arms swinging 10.8–12.9 ms within the run: unresolved, ceiling about 0.2 ms. Code on branch `ksc5-attn-parked-2026-10-24` | survey, "KS-C5 read" |
| KS-D1 | done | done 2026-10-23: `attention_prefill_steel_split` (head dim split over simdgroups) at 256 and 512; E2B's 4,096-row prefill 28.3 → 6.13 s (4.61×, D1-G3 SHIP); G-S9d and a 2,048-token cell PASS (D1-G2); kernel error at the hd-64 steel's size (G1). Follow-ups: a model-level check on Gemma 3 (head dim 256, ungraded); KS-H1's 512 blocking now applies to E2B and E4B; a scratch re-read of their blocked pass is owed | survey, "KS-D1 read" |
| KS-D2 | done | done 2026-10-23: `attention_prefill_steel` templated on the head dim and admitted at 64; the 0.5B's prefill 2.88× at 3,900 rows and 1.33× at 512 (D2-G4 SHIP); the 128 instance byte-identical (G1); kernel error equal to the fused kernel's at 64/80/96 (G2); set A on Q05 SHIPS, closer to the reference than fused (G3). Left as its own row: 80 and 96, each after a model-level gate | `docs/measurements/ks-night-2026-10/README.md` |
| KS-D3 | done | done 2026-10-22: bit-identical (D3-G1, G2 PASS); `delta_rule_seq` at M=512 27.2 → 19.7 ms GPU, median 0.7245, 10 of 10 pairs (D3-G3 SHIP, run at night on the owner's word); `delta_conv_seq` 0.53 of its time | `docs/measurements/ksd3-2026-10/README.md` |
| KS-D2b | blocked | 2026-10-24: Phi-3 at int4 is not Metal-resident (its activation outliers, H2), and on Metal it runs only the q4k lane, which has no batched prefill; no other head-dim-96 model is on this Mac. Reopens with a Metal-resident model at 96 (or 80) | survey, "KS-D2b read" |
| KS-D4 | done | done 2026-10-23 (w32): the EmbeddingGemma 2 encoder and the Gemma 4 vision and audio towers run their GEMMs on `tower_gemm_w32`, byte-identical to aikit's (D4-G1, G2); the E4B vision tower 1.61× faster, median 0.6192, 10 of 10 (D4-G3 SHIP). Open: w16 weights and the fused attention, each under the tower gate | survey, "KS-D4 read" |
| KS-D5 | done | done 2026-10-23 (handoff): the dense int4 prefill fuses the gate\|up GEMM with SwiGLU (`gemm_w4f16_swiglu`, its half tile in the staging arrays' memory) and drops `guF` where every layer takes the 64 × 64 tile; bit-identical; the 1.5B's 3,900-row pass scratch 261.9 → 121.9 MB; prefill median 0.9881 of two-step, 8 of 10 pairs (D5b-G1 to G3). The first build (an extra 8 KB tile) was 10.4% slower, parked, then fixed. Not covered: the small-M tiles, the int8 lane and the MoE expert GEMMs, which keep `guF`; a fused form of those is the next idea | `docs/measurements/ks-night-2026-10/README.md` |
| KS-D5c | done | done 2026-10-23: the native int8 lane's dense prefill fuses SwiGLU into the gate\|up GEMM too (`gemm_w8f16_swiglu`); bit-identical (D5c-G1); the 0.5B's 3,900-row int8 pass scratch 143.9 → 68.0 MB (D5c-G2); median 0.9931, 7 of 10 pairs (D5c-G3 SHIP). Still keeping `guF`: the small-M tiles and the MoE expert GEMMs, neither worth a build (registration) | `docs/measurements/ks-night-2026-10/README.md` |
| KS-D6 | done | done 2026-10-23: the grid towers allocate the unfused attention's scratch and `f2` only where read, and stage `pt` in `f1`; Qwen3-VL-2B at 4K holds 2,622.5 → 1,515.4 MiB; outputs byte-equal on MEM2-G1's seven towers (D6-G1, G2 PASS) | `docs/measurements/ks-night-2026-10/README.md` |
| KS-D7 | done | done 2026-10-23, shipped by owner decision over the 2% bar: the prefill GEMM stages two nibbles per half2 operation; bit-identical (all 1,048,576 cases; the 1.5B's 3,900-token prefill byte-equal); 1.1% faster on that prefill (median 0.9888, 10 of 10 pairs); scratch and allocated memory not higher (D7-G3). The first read (0.9983) was an A/A, withdrawn | `docs/measurements/ks-night-2026-10/README.md` |
| KS-D8 | done | done 2026-10-24: batched Mamba-2 prefill, the relu² MLP and mixer-only layers on Metal; the 9B's 512-token prefill 5.14× (D8-G4 SHIP); G1, G2, G5, G6 PASS; D8-G3 PASS on its reference route (against a CPU bf16 reference the pass reads mean KL 0.046 and top-1 7 of 8, the sequential path 0.102 and 4 of 8) | survey, "KS-D8 read", "D8-G3 reference-route read" |
| KS-D9 | done | done 2026-10-24: the generic pager's phase 2 batched expert by expert, its scratch allocated once per call and priced in the plan; D9-G1 bit-identical on 7 paged cells (against sequential and against the per-row phase 2); Qwen1.5-MoE-A2.7B's 512-row paged prefill median 0.9050, 0.8241 and 0.9131 of per-row, 3 of 3 pairs each, 0 logits differ (D9-G2 SHIP) | survey, "KS-D9 read", "D9-G2 read", "After the review of #224" |
| KS-D10 | open | aikit: GPU timestamps in `TestMetalANNCrossover` (no consumer today) | — |
| KS-E1 | open (measured) | Mac 2026-10-25: the int4 decode GEMV moves 53.7 GB/s (144 µs for 8960×1536), slower per matrix than the production int8 GEMV (~113–145 GB/s from the decode profile): compute-bound. Next: a same-kernel A/B with the change built | `docs/measurements/ks-e-profiles-2026-10-25/README.md` |
| KS-E2 | done | done 2026-10-24: the six single-workgroup WebGPU quantizers and the prefill row forms on 256 lanes, rmsnorm's sum kept on 64; bit-identical (E2-G1, and every Context builder wired through quantSrc); the 1.5B's WebGPU decode 24.7 → 21.8 ms a token, median 0.8810, 10 of 10 pairs, with an A/A control (1.0041) in the same run and busy cores read per arm (E2-G2 SHIP) | survey, "KS-E2 read" |
| KS-E3 | done | done 2026-10-24, kept on main by Francis's decision: the WebGPU DecodeRunner records the next token's command buffer while the GPU runs this one; bit-identical (E3-G1); E3-G2 on its full guard (busy cores, GPU contention, vanished clients) passes, median 0.9166, 10 of 10 pairs; the size varies with the machine's state (a quiet earlier session read 0.9916, also 10 of 10) | survey, "KS-E3 read" |
| KS-E4 | open (measured) | Mac 2026-10-25, Qwen3-VL-2B int4 on WebGPU: projections are 84% / 80% / 54% of the batched pass at 256 / 1,024 / 4,096 rows | `docs/measurements/ks-e-profiles-2026-10-25/README.md` |
| KS-E5 | open (measured) | Mac 2026-10-25, the 1.5B on WebGPU: decode attention is 62% / 87% / 93% of a token at 2k / 8k / 16k positions, about 15 µs a position | `docs/measurements/ks-e-profiles-2026-10-25/README.md` |
| KS-E6 | open (measured) | Mac 2026-10-25: prefill attention is 43% of a 4,096-row pass (5% at 256, 16% at 1,024) | `docs/measurements/ks-e-profiles-2026-10-25/README.md` |
| KS-E7 | open | either box: KV bytes per slot and KV equality past the window | — |
| KS-E8 | open | either box: a WebGPU twin of `cuda/lora_cost_measure_test.go` | — |
| KS-E9 | open | either box: `TestSpike_capabilities` | — |
| KS-E10 | open | either box: dispatch count, then an A/B | — |
| KS-E11 | open | either box: `deepseek-tiny`, then V2-Lite | — |
| KS-F1 | open | either box: a fork/join counter and bit-identity on the tiny MoE fixtures | — |
| KS-F2 | active | #239: gpt-oss's attendSinksToken fans its heads out and scores eight keys at once, bit-identical, default on for gpt-oss (Francis, 2026-10-25 10:16; limited to gpt-oss at 10:35). gpt-oss-20b at depth 128 10 of 10 pairs faster, 1.025×; then KS-O4 | `docs/measurements/ks-f2-2026-10/README.md` |
| KS-F2b | done | default on 2026-10-25 by Francis's call (14:01, option (c)) for every attendQueryP family: Qwen3.5/3.6, Granite-4.0-H, Nemotron-H, LFM2 and Llama 4. Bit-identical (768 function-level runs, 132 fanned; gate identity IDENTICAL for all five at 640 steps). Qwen3.5-0.8B on the Mac CPU 1.113× at depth 128 and 1.833× at 1,024, 10 of 10 each; the other four ship untimed by his call. LFM2.5-2.6B's confirming run is queued on the Mac's night runner | `docs/measurements/ks-f2b-2026-10-25/README.md` |
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
| KS-H1 | open (512) | done 2026-10-22 for head dim ≤ 256 and E-models on their batched pass: blocked prefill byte-equal to whole, admission prices one pass; KH1-G1 to G4 PASS. G5 read 2026-10-24: Qwen3.5-9B at 20K, transient prefill scratch 4.86 → 0.58 GiB, peak footprint −70%. Left: head-dim-512 models (the tile choice must key on the whole prompt's length) | `docs/measurements/ksh1-2026-10/README.md` |
| KS-H2 | done | read 2026-10-25: the 7B's weights round up by 405.8 MiB; Qwen3-8B's 16k f16 pools by 135.0 MiB; rounding explains the card's drop to within 14.4 MiB; free after `kvAllocPools` 1,821.1 MiB (7B) and 391.1 MiB (8B at 16k) | §11, "KS-H2 / KS-H3 readings, read" |
| KS-H3 | open | baseline read 2026-10-25 before the move (`moe_route` 4,416 B a thread, 132.0 MiB retained); the move (the per-thread arrays to shared memory) is not on `main`, so the reading after it is still to do | §11, "KS-H2 / KS-H3 readings, read" |
| KS-H4 | done | done 2026-10-23: the grid tower releases scratch over 256 MiB after 5 s idle; Qwen3-VL-2B 10 s after a 4K forward 1,515.4 → 0.0 MiB; outputs byte-equal; a repeat call 0.9996 of release-off, the first call after a release +67 ms on 50 s (H4-G1 to G3 PASS). Other tower accelerators and aikit's `qwenmetal`/`visionmetal` not covered | `docs/measurements/ks-night-2026-10/README.md` |
| KS-H5 | open | either box: bytes allocated at build | — |
| KS-H6 | blocked | nobara-ks-cuda, after KS-H2 and KS-H3 read: re-derive both margins from measured parts, the 2,048-row pass's scratch included, and bring the numbers to Francis before changing either guard (KS-O3 decided A, Francis, 2026-10-25) | — |
| KS-O1 | done (Metal and CUDA, owner overrides) | native Q5_K/Q6_K on the Metal q4k lane. O1-G1–G6 PASS (G-Q2 on Phi-3 98.39%, +4.39 points; KL to the file 0.7638 of int8; Phi-3 −357 MB on the device); O1-G7 KILL (re-run on the fixed harness: Phi-3 0.9743, 10 of 10; the 1.5B 1.0127, 1 of 10), shipped on Francis's override of 2026-10-24 11:47 am. The head stays int8. **CUDA: done 2026-10-24** (owner's option A on O1C-G5): Phi-3 mini q4k 114.9 → 127.3 tok/s (0.9025, 10 of 10), the 1.5B 0.9786 (10 of 10); −374.7 MB on the device, head included; KL to the file 0.39× (Phi-3), 0.91× (0.5B); G-Q2 +0.13 points. Also fixes main's concurrent `noteKQuant` map write | survey, "Owner override of O1-G7" |
| KS-O1-F | done | closed 2026-10-25 by Francis (20:31) on the (c) finding, read 2026-10-25 (D1, D3, D2a, (c)): the native lane's gap on 110 × "the " (p10 0.8826 against int8's 0.9401) is specific to that prompt. On the same input the native qkv projection is closer to f32 than int8's at every layer, and its kernels are exact. On "and ", "apple ", "hello " and a 110-token natural text, the native lane is the closer end to end (KL 0.27–0.73× int8's). No fix indicated | survey, "KS-O1-F diagnostic"; "KS-O1-F (c) read" |
| KS-O1a | done | done 2026-10-24: the q4k lane's Q4_K scale decode without a branch (`q4k_scale_min2`), bit-identical in 192 kernel cells and 24 Phi-3 runs (O1a-G1, G2); Phi-3 mini q4k Metal decode 33.4 → 41.7 tok/s, median 0.8044, 10 of 10 pairs (O1a-G3 SHIP) | survey, "KS-O1a read" |
| KS-O2 | moved | KS-B15: decided A (Francis, 2026-10-25): the opt-in on the exact attention kernels, band prompts on the exact GEMM; the fast-GEMM option is not approved (it would need its own fidelity gate and a new decision) | §2 |
| KS-O3 | moved | KS-H6: decided A (Francis, 2026-10-25): re-derive the margins after KS-H2 and KS-H3, numbers to Francis before either guard changes | §2 |
| KS-O4 | moved | KS-F2 (mac-kapitsa): decided B, approved gated (Francis, 2026-10-25): a fidelity gate (KL and agreement against today's gpt-oss path, what each outcome means) registered before any code | §2 |
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

**Decided 2026-10-25 (Francis, 6:33–6:38 am, relayed by the coordinator).**
- **KS-B8: A.** 2,048-row CUDA prefill passes by default where scratch allows, priced by admission.
- **KS-O2 / KS-B15: A.** The 64 KB opt-in goes through aikit onto the exact attention kernels (bound 12,160 → 16,256 keys).
  - Prompts in that band stay on the exact GEMM, which is bit-identical. The gain is measured.
  - The fast-GEMM option is not approved; it would need its own fidelity gate and a new decision.
- **KS-O3 / KS-H6: A.** After KS-H2 and KS-H3 read, the two 384 MiB margins (`cuda/resident.go:ctxCapMarginBytes`,
  `cuda/resident.go:slotMarginBytes`) are re-derived from measured parts, the wider (2,048-row) pass's scratch included.
  - The numbers come to Francis before either guard changes.
- **KS-O4: B, approved gated.** gpt-oss moves onto the batched acc64 attention with a sink term.
  - A fidelity gate (KL and agreement against today's gpt-oss path, and what each outcome means) is registered before any code.
- **Who:** the three CUDA items (KS-B8, KS-B15, KS-H6) go to nobara-ks-cuda after #230 merges; KS-F2 and KS-O4 go to mac-kapitsa.

The options as they were put to the owner:

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

**KS-A1 on CUDA: the first measurement's gate, registered 2026-10-24 before any kernel is written or timed (A1-G0), on the
owner's word of 4:37 pm ("#2 then #1": KS-A1's CUDA half, then KS-O1 on CUDA).**
- **Prototype kernels** (`cuda/ks_a1_proto.cu`, test-only, a module of its own; nothing shipped loads it): one warp per output row
  group, 8 warps a 256-thread block, the shipped per-32 int8 activations (`quantG32`'s codes, scales and group sums) and the
  explicit `__fmaf_rn` / `__fmul_rn` form of `rowQ4K`.
  - `gemv_q5k_g32` (Q5_K, 176-byte blocks, 16-byte aligned in the file's own layout): (A) `rowQ4K`'s lane map, lane l on pair
    l mod 4 of block l / 4, plus the block's 32 `qh` bytes; (B) 8 lanes a block, 4 blocks in flight, lane l on 16 bytes of pair
    (l mod 8) / 2's codes and the matching 16 `qh` bytes, the min term added once a group (the low sub-block's by the even lane,
    the high one's by the odd lane).
  - `gemv_q6k_g32` (Q6_K): the 210-byte block is not word-aligned, so the prototype reads segment-wide planes of the same total
    size (`ql`, `qh`, int8 scales, `d`), O1's production layout on Metal. Lane k of a block takes half k / 4 and elements
    8·(k mod 4)…+7 of each of the half's four 32-element groups; the two Q6_K sub-scales of a group fold in integer arithmetic,
    then × d × the group's activation scale.
  - Each at R = 1, 2, 4 rows a warp (the activation loads shared by the R rows).
- **Instrument:** `TestKSA1_gemvTiming` (`cuda/ks_a1_test.go`, `-tags cuda`), the timing lock held exclusively. The shipped
  `gemv_w8a8_g32` comes from the embedded `actgroup.ptx` itself. Per shape and kernel: CUDA-event time of 50 launches, median of 7
  batches after 3 warm launches, the weights rotated over copies of at least 64 MiB per arm (16× the 4 MiB L2), beside a
  streaming-read ceiling over the same bytes. GB/s is the kernel's own weight bytes (int8: N·K plus the f32 row scales; Q5_K: the
  file's N·K/256·176; Q6_K: N·K/256·210) over its time. Load average and free memory are logged at the start, and CLAUDE.md's
  void rule applies (load above 8.0 on nobara, or free VRAM under the run's need plus 2 GB: about 0.6 GB here).
- **Correctness first:** every prototype, at every R and shape, against a float64 host reference that dequantizes the same blocks
  by ggml's formulas and takes the same int8 activations: every row within G-Q1's bar (|err| ≤ 1e-5 of Σ|w·a|). A kernel that
  misses it is not timed and the reading is void for its kind.
- **Shapes (Phi-3 mini q4, the graded ones):** Q5_K qkv 9216×3072 (×32 layers); Q6_K `ffn_down` 3072×8192 (×16) and the head
  32064×3072. **Read, not graded:** Qwen2.5-Coder-1.5B q4_k_m's Q6_K shapes, `ffn_down` 1536×8960 and `attn_v` 256×1536 (O1-G7's
  KILL on Metal came from that model), and the shipped `gemv_q4k_g32` at Phi-3's gate|up 16384×3072 for context.
- **The pick:** per kind, the fastest variant and R summed over Phi-3's tensors of that kind; the smaller R on a tie within 2%.

| outcome | PASS iff | what it means |
|---|---|---|
| **KS-A1 GO** (the survey's bar) | the best Q5_K kernel's GB/s at 9216×3072 is ≥ 0.95× `gemv_w8a8_g32`'s at the same shape (its time then ≤ 0.724× the int8 kernel's) | the survey's speed case stands (about 1.03× llama.cpp by its arithmetic); KS-O1 on CUDA builds Q5_K |
| **KS-A1 short of the bar** | that ratio is under 0.95, but Q5_K's summed time is ≤ 0.95× the int8 kernel's (O1-G0's kill rule on Metal) | the survey's speed estimate is withdrawn and the record says so; KS-O1 on CUDA still builds Q5_K by the rule Metal's build used, its own gates deciding |
| **KS-A1 NO-GO** for Q5_K | Q5_K's summed time is > 0.95× the int8 kernel's | no speed case for native Q5_K on CUDA; whether to build it for memory and fidelity alone (Metal's option C) is the owner's decision, so the work stops there with DECISION NEEDED |
| Q6_K (KS-O1's build rule, read in the same run) | the best Q6_K kernel's time summed over Phi-3's 16 `ffn_down` and the head is ≤ 0.95× the int8 kernel's sum | built in KS-O1 on CUDA; otherwise killed for speed, and the same owner decision as Q5_K's NO-GO |

Stopping rule: one pass. A kind that fails its rule may have at most one amended variant, registered here before it runs, as O1-G0
did on Metal; that reading is final. Expected cost: the PTX build and the correctness check seconds, the timed pass about 1 min.

**A1-G0 amended before it ran (2026-10-24, 16:49):** Phi-3's hidden size (3072) is over `fusedG32MaxHidden`, so on CUDA its q, k
and v run as three separate `doG` launches of 3072×3072, not one launch of 9216 rows. A read-only Q5_K line at 3072×3072 is added.
The graded shape, the bars and the rule are unchanged.

**A1-G0 read 2026-10-24, by day: KS-A1 GO; Q6_K built.** Run 17:05 on nobara by `night.py serve` (the timing lock held
exclusively) at b21a1f950, NVIDIA driver 595.91.07. At the start: 1-minute load 0.86 (the void line is 8.0), 57.9 GB available,
7,277 MiB of VRAM free, only the desktop compositor's context on the card. Every prototype passed the correctness check first: worst
|err|/Σ|w·a| 1.2e-7 to 2.4e-7 against the 1e-5 bar. The streaming ceiling read 416–419 GB/s. Medians of 7 batches of 50 launches,
in ms (GB/s):

| shape | `gemv_w8a8_g32` | best native | native / int8 time | native / int8 GB/s |
|---|---:|---:|---:|---:|
| Phi-3 qkv 9216×3072, Q5_K (graded) | 0.0705 (401.9) | B, R = 1: 0.0492 (396.0) | 0.697 | **0.985** |
| Phi-3 q, k or v 3072×3072, Q5_K (read only, as dispatched) | 0.0250 (378.4) | B, R = 1: 0.0180 (360.8) | 0.720 | 0.954 |
| Phi-3 `ffn_down` 3072×8192, Q6_K | 0.0626 (402.0) | R = 1: 0.0523 (394.9) | 0.835 | 0.982 |
| Phi-3 head 32064×3072, Q6_K | 0.2374 (415.4) | R = 1: 0.1950 (414.3) | 0.821 | 0.997 |
| 1.5B `ffn_down` 1536×8960, Q6_K (read only) | 0.0361 (381.9) | R = 2: 0.0300 (375.9); R = 1: 0.0319 | 0.833 / 0.886 | 0.984 / 0.925 |
| 1.5B `attn_v` 256×1536, Q6_K (read only) | 0.0057 (69.7) | R = 2: 0.0058; R = 1: 0.0069 | 1.020 / 1.217 | 0.802 / 0.672 |
| Phi-3 gate\|up 16384×3072, shipped `gemv_q4k_g32` (context) | 0.1223 (412.0) | 0.0722 (392.2) | 0.590 | 0.952 |

- **KS-A1: GO.** The best Q5_K kernel reaches 0.985× the int8 kernel's GB/s at the graded shape (bar 0.95), so the survey's speed case
  stands by its own arithmetic. Every variant and R cleared it (0.969–0.985).
- **Summed over Phi-3 (ms a token):** Q5_K, 32 qkv launches: 2.257 int8 → 1.573 native (B, R = 1), **0.697**. Q6_K, 16 `ffn_down`
  and the head: 1.240 → 1.032 (R = 1), **0.832**. Both are under O1-G0's 0.95, so KS-O1 on CUDA builds both.
- **The pick:** every Q5_K variant and R is within 2% of the fastest, and the smaller R wins a tie. At R = 1, B (0.6970) is 0.9% faster
  than A (0.7030), so B is built at R = 1. Q6_K: R = 1 (R = 4 is within 2%; the smaller R wins). `kquantGemvRows` is 1.
- **What it does not show:** a kernel-only reading gives direction, not size (CLAUDE.md: its served effect ranged 0.05–1.72× of it).
  The Phi-3 whole-token saving O1C-G7 can expect is at most about 0.9 ms (0.68 + 0.21) of the projections' time. On the 1.5B, the
  tiny `attn_v` is slower native at R = 1 (1.22×), but on CUDA it runs inside `fused_rms_qkv_g32` at that hidden size, not standalone;
  its `ffn_down` is 0.886× at R = 1. O1C-G7 reads the 1.5B as a whole token.
- **Log:** `docs/measurements/ks-a1-2026-10/a1-g0-2026-10-24-1704.log` (from `~/goinfer-logs/ks-a1-2026-10/`).

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
- **Decided (Francis, 2026-10-25): A.** 2,048-row passes by default where scratch allows, priced by admission. nobara-ks-cuda,
  after #230.
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
- **Decided (Francis, 2026-10-25, KS-O2): A.** The band prompts stay on the exact GEMM, and the fast GEMM is not approved.
  nobara-ks-cuda, after #230.
- **Waste:** the needed call exists in the pinned gocudrv, but aikit does not pass it through.
- **Change:** pass the attribute through aikit and set it on the exact attention kernels: bound 12,160 → 16,256 keys.
- **Class:** bit-identical attention; the decision is about the GEMM that comes with it. **Size:** S.

**KS-B15 build: gates, registered 2026-10-26 before any code (B15-G).** On aikit `gpu` v0.35.0
(`Pipeline.SetMaxDynamicSharedBytes`, `Pipeline.MaxDynamicSharedBytes`).
- **The change:**
  - At load, the three exact prefill attention kernels opt in to the most dynamic shared memory each can take: `attn_batched`,
    `attn_img_batched` and `attn_span_batched`.
  - The prefill's span limit becomes the smallest of what they got. On the RTX 2070 SUPER that is 64 KiB less each kernel's
    static shared memory, which moves the span bound from 12,160 keys to at most 16,256.
  - In that band a layer that took the row-batched kernels (`prefillSplitKVAttn`) takes `attn_batched` or its image twins
    instead. Past the new bound, the row kernels serve as today.
  - A failed opt-in keeps 48 KiB, which is today's behaviour.
- **Unchanged:**
  - Decode's single-block attention and its split-KV choice keep the 48 KiB bound (`splitKVRequired`), and so does the drafter.
  - A band prompt stays on the exact GEMM: `prefillPastBoundExact` keeps the 48 KiB bound (owner decision A; the fast GEMM is
    not approved).
- **Memory:** a band pass no longer allocates the row kernels' score scratch (up to 64 MiB plus its inverses,
  `PrefillSpanScratchBytes`). Nothing is added.

| gate | tier | instrument | PASS iff |
|---|---|---|---|
| B15-G1 the opt-in | 0 (device, tiny fixture) | `llama-attnfa-tiny` on the card | each of the three kernels reports a limit above 48 KiB, and the prefill's limit is their minimum. A 16-row pass whose last row attends the new bound in keys runs on `attn_batched` (no row-batched group), and one attending a key more takes the row kernels. Decode's `splitKVRequired` is unchanged at 12,161 keys. Plant: with the opt-in skipped, the prefill limit is 48 KiB and a band span takes the row kernels |
| B15-G2 bits, tiny | 1 | `llama-attnfa-tiny` at context 17,000, f32 and f16 KV, fast levers off. Prompts of 12,400, 14,000 and 16,000 tokens (in the band), and 16,384 (past it). Arms: the opt-in, and today's limit set back to 48 KiB on the same resident. Also `TestKV3Tiled_batchedEqualsPerToken_tiny` at 12,400 and 16,384 | in the band, 0 differing logits and 0 differing K/V elements between the arms. The opt-in arm has no row-batched group and the other arm some, and neither runs a tensor-core launch. Past the band both arms run the row kernels and are equal. The batched prefill equals the per-token path (decode on split-KV), bit for bit, at both lengths |
| B15-G3 bits, real | 1 | Gemma-3-4B Q4_K_M, f32 KV, default levers: a 13,824-token lead, then a 512-row pass whose last row attends 14,336 keys, each arm once. The load is the fit check: G3 runs first, and if the f32 load is refused (about 3.9 GB of KV beside the weights on 8 GB), G3 and G4 both run at f16 KV | 0 differing logits and 0 differing K/V rows across the pass; the opt-in arm has no row-batched group; neither arm runs a tensor-core launch |
| B15-G4 speed | 1 (timed, the timing lock) | the same load and pass: off and on interleaved A B B A, 10 pairs after 2 warm runs each. The load average and free memory are recorded per pair, and a pair started above 8.0 is void and re-run. Second reading: one whole 14,336-token `PrefillLast`, off then on, twice each | **SHIP** iff the pass's median off/on ≥ 1.00 with at least 8 of 10 pairs faster. **KILL** at a median < 0.98: the band keeps the row kernels and the opt-in stays unset. **Ambiguous** (0.98–1.00, or fewer than 8 faster): parked, to Francis |

- **Why G4's bar is 1.00, not the standing merge rule's 1.05:**
  - Francis decided A; this records the gain rather than earning a decision.
  - A change that is bit-identical and no slower still frees the band's score scratch.
  - It also lets an image span pass on a sliding layer run in the band, where it declines today.
  - The whole prompt is a reading, not the rule. Of a 14,336-token prompt's 28 passes at 512 rows, only the four or five whose
    last row is past 12,160 keys can move, so the whole-prompt ratio is diluted by the 23 passes that cannot.
- **Stopping rule:** each gate once. **Cost:** G1 seconds; G2 about 5 min (two per-token references at 12,400 and 16,384
  rows, plus eight batched passes); G3 about 2 min; G4 about 5 min (24 passes of about 0.2–0.5 s and four whole prompts).
- **Review:** it touches kernel routing, so Gemini reviews it before merge.

**KS-B15 build read 2026-10-26, 02:55–03:12, by day: KILL.**
- **Setup:** run by `night.py serve` (the timing lock held exclusively) at 410c6ee87, from
  `docs/measurements/ks-b15-2026-10/run-b15.sh` (the G4 phase from c758a83da, the same tree with the script's G4-only phase
  added).
  - NVIDIA driver 595.91.07.
  - At G4's start the 1-minute load was 0.62, and 58.2 GB was available.
  - The code is on branch `ks-b15-shared-optin-2026-10-26` (pushed, unmerged).
  - Logs: `docs/measurements/ks-b15-2026-10/b15-2026-10-26-0255.log` (G1–G3) and `b15-2026-10-26-0307.log` (G4).
- **B15-G1: PASS.**
  - `attn_batched`, `attn_img_batched` and `attn_span_batched` each opted in to 65,536 B. None has static shared memory.
  - So the prefill bound is 16,256 keys.
  - A 16-row pass whose last row attends the bound ran on `attn_batched`, and one key more took the row kernels.
  - Decode's bound is unchanged.
  - Plant: with the opt-in skipped, the limit stayed at 48 KiB and a 12,400-key span took the row kernels.
- **B15-G2: PASS.** At 12,400, 14,000 and 16,000 tokens, in f32 and f16 KV, the opt-in arm ran no row-batched group (today's
  route ran 6, 18 and 34). 0 of 256 logits and 0 K/V elements differed. At 16,384 both arms ran the row kernels and were equal.
  `TestKV3Tiled_batchedEqualsPerToken_tiny` passed at 12,400 (now on `attn_batched`) and 16,384.
  - The run's own exit was red only because two extra families tests skipped subtests whose Gemma 4 fixtures are not on this
    box (`gemma4-dense-twogeom-tiny`, `gemma4-dense-scaled`). Those are not registered gates.
- **B15-G3: PASS.** Gemma-3-4B Q4_K_M, f32 KV (the load fitted), a 512-row pass at 14,336 keys: 0 of 262,144 logits and 0 of
  35,651,584 K/V elements differed, the opt-in arm ran no row-batched group (today's ran 20), and neither ran a tensor-core
  launch.
- **B15-G4: KILL.** The opt-in is slower on every pair: off 1,854.7–1,861.0 ms, on 1,911.9–1,914.3 ms, **median off/on
  0.9715**, 0 of 10 pairs faster, no void. That is under the 0.98 kill line.
  - The whole 14,336-token prompt (a reading) took 40.99 and 41.00 s off, 41.26 and 41.25 s on.
  - By the registration, the band keeps the row-batched kernels and the opt-in stays unset. Nothing merges.
  - aikit `gpu` v0.35.0's `Pipeline.SetMaxDynamicSharedBytes` stays available, and goinfer does not call it.
  - Why `attn_batched` is slower at these spans was not investigated.

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

**KS-C1 (a)+(b) gates, registered 2026-10-23 before the code (C1-G), on the owner's word of 1:55 pm (KS-O1 option A: build (a)+(b),
decide O1 once they read).** The build:
- (b) `gemv_w8_f32_rows<R>` (plain, residual, bias): R int8 rows a simdgroup sharing each lane's activation loads, every row's lane
  loop, `simd_sum` and epilogue statement for statement `W8_F32_BODY`'s, so each output is `gemv_w8_f32`'s bits.
- (a) `gemv_q4k_f32_rows<R>` (plain, residual, bias): all 32 lanes on one super-block, lane l on byte group l mod 8 of 64-weight
  chunk l / 8 (one `uint` of quants a row), its two `float4` activations loaded once for R rows; d·sc·dot(q, a) per sub-block half
  and the min term as dmin·m·Σa. A new order of the same products: not bit-identical, graded by G-Q1's bar and G-Q2.
- `q4kGemv` launches them at ceil(rows / R) simdgroups; the batched step's `mc3_gemv_q4k_f32_rows<B>` takes (a)'s per-lane loop, so
  the step stays decode's bits. R is chosen by C1-G0.

| gate | tier | instrument | PASS iff |
|---|---|---|---|
| C1-G0 measure | timed, Mac | the survey's first measurement: `gemv_q4k_f32` and `gemv_w8_f32` at Phi-3's shapes (o, gate\|up, down Q4_K; qkv, down's Q6_K layers int8), beside the streaming ceiling and the new kernels at R = 1, 2, 4, SLC-defeating rotation, median of 7 | read, not gated; R per family is the fastest, the smaller R on a tie within 2% |
| C1-G1 kernels | 0 | G-Q1 (`TestQ4KLaneKernels`) on the new Q4_K kernels, its bar (per row within 1e-5 of Σ\|w·a\|) and its planted defects; the new int8 kernels against `gemv_w8_f32` at G-Q1's int8 shapes, every variant | (a) every row within the bar and every plant red; (b) byte-equal |
| C1-G2 step | 0 | `TestQ4KStep_verifyRowsMatchDecode`, `_promptMatchesSequential`, `_twoSequencesMatchAlone` and `_plantIsRed` | as they assert today (byte-equal to decode) |
| C1-G3 fidelity | 2, Mac | G-Q2 (`TestQ4KLane_realPhi3NonInferiority`) with the new kernels | PASS by its rule |
| C1-G4 speed | timed, Mac, timing lock | a whole-token in-process A/B on Phi-3 mini at `--quant q4k`: one resident, its q4k-lane pipelines swapped between the new kernels and the old ones (a test-only library), 128 greedy decode tokens after a 32-token prompt, A B B A over 10 pairs after 2 warm runs each | SHIP iff the median new/old per-token time is ≤ 0.98 with at least 9 of 10 pairs faster; otherwise PARK |

Phi-3's load is fit-checked first; refused by day, G3 and G4 go to the night queue as case (a). Planted defect for (a): the min term
dropped from the high half (`m1·Σa_hi` omitted); C1-G1 goes red. Stopping rule: one pass each. Cost: G0 about 2 min; G1, G2 seconds;
G3 about 20 min; G4 about 4 min.

**C1-G0 read 2026-10-23 (13:56, the lock held exclusively, load 3.50, 3.4 GB available), and an amendment to (a) before any grading
gate.** Ceiling 178.6 GB/s. The first Q4_K layout (one `uint` of quants a lane, 8 weights) read old → R = 1: o 0.139 → 0.103 ms (38 →
52 GB/s), gate|up 0.708 → 0.517 (40 → 55), down 0.365 → 0.266 (39 → 53), R = 2 and 4 no better; the int8 kernels qkv 0.170 → 0.160 at
R = 2 (166 → 177 GB/s), the Q6_K down layers 0.175 → 0.158 at R = 4 (144 → 160). Log: `~/goinfer-logs/ksc1-g0-2026-10-23.log`. The
Q4_K kernel stays under a third of the ceiling because every lane decodes two sub-blocks' scales, d and dmin for 8 weights.
**Amended (a), before C1-G1 to G4 ran:** each lane takes 32 weights of one super-block, 8 lanes a super-block and 4 super-blocks in
flight (lane l: super-blocks l/8, l/8 + 4, …; within one, chunk (l mod 8)/2 and half l mod 2, one `uint4` of quants, 16 low and 16
high nibbles against 8 `float4` activations), so the scale decode, d·dmin and the min-term sums are paid once per 32 weights. Its
own C1-G0 reading picks R; the gates and bars are unchanged.

**C1-G0 re-read with the amended layout (14:09, load 3.99, 3.1 GB available), and R = 8 added before any grading gate.** Ceiling
179.3 GB/s. Q4_K old → R = 1 / 2 / 4: o 0.140 → 0.101 / 0.090 / **0.082 ms (1.70×, 64 GB/s)**; gate|up 0.708 → 0.514 / 0.423 / **0.364
(1.94×, 78 GB/s)**; down 0.365 → 0.248 / 0.225 / **0.201 (1.81×, 70 GB/s)**. Int8 as before: qkv best at R = 2 (1.06×), the Q6_K down
layers at R = 4 (1.09×). Still improving at R = 4, so the Q4_K R set gains 8 (one more C1-G0 pass); the rule stays the fastest R.
Log: `~/goinfer-logs/ksc1-g0b-2026-10-23.log`.

**C1-G0 with R = 8 (14:11, load 3.25, 3.4 GB available):** Q4_K R = 4 / 8: o 0.083 / **0.080 ms (1.75×)**, gate|up 0.365 / **0.351
(2.02×, 81 GB/s)**, down 0.202 / **0.198 (1.84×)**; summed, R = 8 is 3.3% under R = 4, outside the 2% tie: **Q4_K R = 8**. Int8 summed
over its two shapes: R = 2 0.336 ms against R = 1 0.346 and R = 4 0.338: **int8 R = 2**. Log: `~/goinfer-logs/ksc1-g0c-2026-10-23.log`.

**KS-C1 read 2026-10-23, by day.** C1-G1 **PASS**: G-Q1 on the rows kernels at every built R, worst |err|/Σ|w·a| 1.65e-7 (Q4_K) and
1.26e-7 (int8) against 1e-5; all five plants red, the registered one (the high half's min term) at 2.98e-2. The int8 rows kernels
equal `gemv_w8_f32` and its variants byte for byte in 72 cells (both libraries), once the old kernels compile where they stood:
appended elsewhere, the precise library contracted their residual and bias epilogues differently (`ksc1OldLib`). C1-G2 **PASS**: the
four `TestQ4KStep_*` tests (the plant retargeted at the new step text, red at 151,936 logits) and a kernel-level step-vs-decode check.
Getting there took one fix: under fast math the compiler reassociated the Q4_K sum differently in decode's and the step's code
shapes (500 of 1,000 outputs differed), so both loops carry `#pragma clang fp reassociate(off)`, which C1-G0 re-read at no cost
(R = 8: 0.080 / 0.348 / 0.198 ms). C1-G3 and C1-G4: Phi-3 at q4k is fit-refused beside the owner's apps (G-Q2's own load, 14:20), so
both are on the night queue as case (a) (`docs/measurements/ks-night-2026-10/run-ksc1-g3.sh`, `run-ksc1-g4.sh`). G-Q2's reference
path named the unversioned `….int4.metal.giw`, which no longer exists; it now names the v17 sidecar. Logs: `~/goinfer-logs/ksc1-*-2026-10-23.log`.

**C1-G3 and C1-G4 run by day 2026-10-24, on the owner's word.** Neither started on the night queue: the start check read about 6.3 GB
free all night. Francis then changed macOS's available-memory reading to `memory_pressure`'s free percentage × RAM − 500 MB (#203).
Each test binary was rebuilt at its registered commit with that reading applied (`internal/hostmem`, `decoder/hostram_darwin.go`;
nothing the gates measure), from `docs/measurements/ks-night-2026-10/run-ksc1-g4-day.sh` and `run-ksc1-g3-day.sh`; the fit guard
passed on its own, with no bypass. C1-G4 (08:37–08:39, the timing lock held exclusively; the 1-minute load 3.88 at the start, left
over from the two builds and under the 4.0 line, each pair's 1.99–3.65; 12.6 GB available by the new reading; swap 464 MB before and
after): old 44.89–51.65 ms a token, new 29.58–34.22, **median new/old 0.6741, 10 of 10 pairs below 1: SHIP** (22.3 → 33.0 tok/s
after pair 0; the survey's estimate was 37–48). Log: `~/goinfer-logs/ksc1-g4-day-2026-10-24-0837.log`.
C1-G3 (08:39–08:44, the lock held exclusively; not timed): G-Q2 on the rows kernels, Phi-3 mini q4k teacher-forced agreement
**98.66% (367 of 372), 8 of 8 prompts** passing the free-run rule, against the reference (Qwen2.5-Coder-1.5B) 93.99%, 7 of 8: **delta
+4.66 points against the −2.0 margin: PASS**. Log: `~/goinfer-logs/ksc1-g3-day-2026-10-24-0839.log`. Both gates pass, so (a)+(b) ship;
(c), native Q5_K/Q6_K, is KS-O1's decision.
- **First measurement:** time the shipped kernels at the real Phi-3 shapes with `TestMetalGEMVS0`'s method. Kill (a) if
  Q4_K already reads ≥100 GB/s.

**KS-O1 on Metal: gates, registered 2026-10-24 before any code (O1-G), on the owner's word of 10:22 am (build native Q5_K/Q6_K
for the GPU q4k lanes, Metal first).** The build:
- **Kernels:** `gemv_q5k_f32_rows<R>` and `gemv_q6k_f32_rows<R>` (plain, residual, bias), their batched-step twins in
  `metal/batch_rows.go`, and a native Q6_K head (`gemv_q6kf32_head` and its block-argmax twin) beside `gemv_w8f32_head`. Q5_K
  takes `q4k_rows_acc`'s lane map (8 lanes a super-block, 4 in flight, one `uint4` of quants a lane) plus the matching 16 bytes of
  `qh`. Q6_K's 210-byte block is not word-aligned, so it is uploaded split into planes of the same total size (`ql`, `qh`, scales,
  `d`). Lane k of a block takes half k/4 and elements 8·(k mod 4)…+7 of each of its four 32-element groups.
- **Data:** the raw blocks come from the GGUF mapping when the resident is built. The CPU copy stays int8, as it is now, so the CPU
  path is unchanged. This needs a raw Q5_K/Q6_K accessor in aikit's `embed` (only `Q4KRaw` exists), which means an aikit release.
- **Why not bit-identical:** today these tensors run as an int8 re-quantization of the dequantized file. The native kernels compute the
  file's own weights, so their outputs move toward the file and cannot match the old bits. The small-gains rule does not apply, and
  the fidelity gates below decide instead.

| gate | tier | instrument | PASS iff |
|---|---|---|---|
| O1-G0 measure | timed, Mac, timing lock | C1-G0's method (`TestKSC1_gemvTiming`'s harness, SLC-defeating rotation, median of 7) on prototype kernels at Phi-3 mini's shapes. Q5_K qkv is 9216×3072 ×32 layers; Q6_K down is 3072×8192 ×16 layers; the Q6_K head is 32064×3072. Each runs beside `gemv_w8_f32_rows2` at the same shape and the streaming ceiling, with R = 1, 2, 4, 8 | read, then a kill rule per kind. A kind is built only if its best R's time, summed over Phi-3's tensors of that kind, is ≤ 0.95× the int8 rows kernel's sum. If neither kind clears, KS-O1 on Metal is killed and the record says so. R is the fastest, or the smaller one on a tie within 2% |
| O1-G1 kernels | 0 | G-Q1's method: every new kernel and variant against a float64 reference that dequantizes the same blocks on the host. The reference is cross-checked against aikit's GGUF dequantizer on the same bytes. Shapes: one super-block, a partial threadgroup, Phi-3's qkv / down / head, random scales at the 6-bit range ends. Planted defects: (1) Q5_K's high bit dropped; (2) Q6_K's −32 offset dropped; (3) a Q6_K group read with its neighbour's scale; (4) a super-block offset of one | every row within G-Q1's bar (\|err\| ≤ 1e-5 of Σ\|w·a\|); the reference within 1e-6 of aikit's dequantization; each plant red |
| O1-G2 step | 0 | the four `TestQ4KStep_*` tests, plus a kernel-level step-vs-decode check on the new kinds | byte-equal to decode, as today |
| O1-G3 memory | 0 | bytes uploaded for the q4k lane's weights (the sum of the `q4kSeg` buffers) and bytes allocated on the host during the resident build (`q4k_upload_alloc_test.go`'s counter) | device bytes fall to the GGUF's raw bytes for those tensors (Phi-3: about −375 MB); host bytes not higher |
| O1-G4 CPU unchanged | 0 | the CPU q4k forward on the 0.5B q4_k_m (its Q6_K tensors), logits bytes before and after the change | byte-equal |
| O1-G5 fidelity | 2, Mac | G-Q2 (`TestQ4KLane_realPhi3NonInferiority`) on the native kinds | PASS by its rule (≥ the reference's agreement − 2.0 points, free-run passes ≥ the reference's − 1) |
| O1-G6 toward the file | 1, Mac | Qwen2.5-Coder-0.5B q4_k_m (Q6_K attn_v / ffn_down in half its layers, and the head): Metal q4k last-row logits, old lane and native lane, each against the CPU at f32 on the same GGUF (the file exactly), over G3's 8 prompts, teacher-forced. Phi-3 at f32 does not fit 16 GB | mean KL(new) ≤ mean KL(old); ambiguous (parked) up to 1.10×; FAIL above |
| O1-G7 speed | timed, Mac, timing lock | C1-G4's A/B: one Phi-3 mini q4k resident that holds both the int8 and the native buffers for the Q5_K/Q6_K tensors (a test-only build hook), swapped per run. 128 greedy decode tokens after a 32-token prompt, A B B A over 10 pairs after 2 warm runs each. Then the same on Qwen2.5-Coder-1.5B q4_k_m (Q6_K only) | SHIP iff Phi-3's median new/old per-token time is ≤ 0.98 with at least 9 of 10 pairs faster **and** the 1.5B's median ≤ 1.00. PARK if Phi-3 is between 0.98 and 1.00 or has fewer than 9 pairs faster. KILL if Phi-3 is above 1.00 or the 1.5B is above 1.00 |

Each model's load is fit-checked before G5 to G7. Stopping rule: one pass each, and O1-G0's kill rule ends the build early.
Expected cost: G0 about 3 min; G1 to G4 seconds; G5 about 6 min; G6 about 4 min; G7 about 3 min for Phi-3 and 2 for the 1.5B.
These are C1-G3's and C1-G4's walls, plus a second load. Planned order: G0 first; the aikit accessor only if a kind survives G0.

**O1-G0 read 2026-10-24 (10:33, the lock held exclusively, load 3.48, 13.9 GB available), and one amendment to Q5_K before any
grading gate.** The ceiling was 133.9 GB/s, lower than C1-G0's 179, so the box was busier. Ratios are read within this run. The
prototypes were first checked against the float64 reference, worst |err|/Σ\|w·a\| 1.0e-8 (Q5_K) and 7.0e-9 (Q6_K).
- **int8 → native, best R:** qkv 0.242 → 0.298 ms at R = 8 (Q5_K, 65 GB/s, 0.81×); down 0.174 → **0.128** at R = 4 (Q6_K, 161 GB/s,
  1.36×); head 0.528 (`gemv_w8f32_head` 0.573) → **0.476** at R = 4 (1.11×).
- **Summed over Phi-3:** Q5_K 1.230 of int8 at R = 8, so it **fails** the kill rule (≤ 0.95). Q6_K 0.761 at R = 4 **clears**: Q6_K is
  built at R = 4.
- **Log:** `~/goinfer-logs/kso1-g0-2026-10-24.log`.
- **Why Q5_K is slow:** Q6_K runs the same lane map at 161 GB/s. Q5_K shares `q4k_scale_min` with Q4_K, which reads its scale bytes
  one at a time on a branch that splits the simdgroup (lanes with j < 2 against j ≥ 2), twice a block.
- **Amended (Q5_K only), before O1-G1 to G7 run:** the 12 scale bytes are read as three words and every lane's scale and min are
  decoded with masks and shifts, with no branch. Variant (b) keeps (a)'s two dot chains. Variant (c) splits each into two halves.
  The kill rule and R set are unchanged, and this is the last Q5_K reading. If neither variant clears, Q5_K is killed and only Q6_K is
  built.

**O1-G0 re-read with the amended Q5_K (10:35, the lock held exclusively, load 3.31, 14.1 GB available).**
- **Conditions:** ceiling 181.1 GB/s; the int8 qkv kernel ran 0.194 ms (146 GB/s).
- **Q5_K variants, R = 1 / 2 / 4 / 8:**
  - (a) 0.365 / 0.314 / 0.273 / 0.285 ms;
  - **(b) 0.258 / 0.215 / 0.146 / 0.168** (133 GB/s at R = 4, 1.32× the int8 kernel);
  - (c) 0.259 / 0.226 / 0.167 / 0.149.
- **Summed over Phi-3:** (b) at R = 4 is **0.755 of int8 and clears**. (c)'s best (R = 8, 0.767) is 1.6% slower. **Q5_K is built as (b)
  at R = 4.**
- **Q6_K, read again:** 0.620 of int8 at R = 4 (down 1.69×, head 1.29× against `gemv_w8_f32_rows2`, and 1.22× against the shipped
  `gemv_w8f32_head`). Still R = 4.
- **Log:** `~/goinfer-logs/kso1-g0b-2026-10-24.log`.
- **Both kinds pass O1-G0.** The build continues once aikit's `embed` has the raw Q5_K/Q6_K accessor.

**O1 amendments during the build, 2026-10-24, before any grading gate (O1-G4 to G7) ran:**
- **The LM head stays int8.** A native Q6_K head saves Phi-3 17.7 MB a token, about 0.1 ms by O1-G0. It needs a new launch shape at
  six head dispatch sites and a new argmax-partial layout, in code another session is editing. It is left for a follow-up, and the
  native head kernels are not built.
- **Q6_K's planes span the whole segment, not each row.** The planes are ql, qh, scales and d, so every load is aligned for any
  block count. The size stays exactly the file's.
  - **Why:** per-row planes are NB × 210 bytes a row. The 0.5B's ffn_down (K = 4864, NB = 19) then starts odd rows 2-byte aligned,
    and its `uint2` loads misread. This read as cosine −0.75 in `TestQ4KLaneResidentQwen`, and as 0.998 once fixed.
  - **Test change:** O1-G1 gains a K = 4864 shape for both kinds.
- **O1-G3 is measured through `decoder.Load` on the Metal backend,** which builds the resident inside the load. It counts the heap of
  the whole load, a superset of the build's. The int8 arm sets `q4kNativeOffForTest` before the load, so it never opens the reader,
  as before KS-O1.
  - The load keeps its GGUF mapping open for its own resident build, then releases it. Re-opening the file would parse its header
    again: 51 MB of transient heap on the 0.5B's 152k-token vocabulary.
- **Provenance:** a weight records its source as a tensor name and a first row, plus a row table only for a permuted load, so
  retained host memory does not grow per row.

**KS-O1 read 2026-10-24, by day.** Built at 88dcfc1d against aikit 0df4c98 (townsendmerino/aikit#18, `embed.Q5KRaw`/`Q6KRaw`)
through `go.work`.
- **O1-G1: PASS.**
  - Every native kernel and variant at R = 4 is within the bar. The worst |err|/Σ\|w·a\| was 1.35e-7 (Q5_K, Phi-3 qkv) and 1.34e-7
    (Q6_K, the head shape), against 1e-5. The shapes covered one super-block, a partial threadgroup, K = 4864 and Phi-3's.
  - All four plants are red: (1) 4.66e-1, (2) 7.94e-1, (3) 1.25, (4) 9.40e-1.
  - The float64 reference is within 1e-6 of aikit's `RowDequantizer` on 32 Q5_K and 29 Q6_K tensors of the two real files.
  - Tests: `TestKSO1_nativeKernels`, `TestKSO1_referenceMatchesAikit`.
- **O1-G2: PASS.**
  - The step's `mc3_gemv_q5k/q6k_f32_rows<B>` is byte-equal to decode at B = 1, 2, 4, 8 in all three modes (`TestKSO1_stepMatchesDecode`).
  - The four `TestQ4KStep_*` tests pass on the 0.5B, whose Q6_K layers now run native.
  - `TestQ4KLaneResidentQwen/0.5B` reads worst cosine 0.998474 and argmax 31/32 against the CPU, with its bar at 0.99.
- **O1-G3: PASS** on the 0.5B through `decoder.Load` on Metal (`TestKSO1_laneBytes`).
  - Lane device bytes: 335,239,680 → 325,831,680. The Q6_K segments hold exactly the file's 42,900,480 bytes. These undercount the
    int8 scales; the corrected figures are in "Review fixes for #215".
  - Load heap: 775.5 → 770.3 MB.
- **O1-G4: PASS.** The CPU q4k fingerprint (`TestKSO1_cpuQ4KFingerprint`: 16 greedy tokens plus the last hidden state's bytes) is
  `acab5556…50f5` both at origin/main 8cfd5fca and at KS-O1. This hashed tokens and the hidden state; the registered
  logits-bytes digest is in "Review fixes for #215".
- **O1-G6: PASS** (11:26–11:27, the lock held shared). On the 0.5B, against the CPU at f32 on the file, over 566 teacher-forced
  positions: mean KL 0.00155 (int8 lane) → 0.00119 (native), **0.7638**. Native is lower on all 8 prompts and at 371 of 566
  positions. Log: `~/goinfer-logs/kso1-g6-2026-10-24.log`.
- **O1-G7 run:** 11:28–11:31, the lock held exclusively. At the start the 1-minute load was 1.83 (each pair 0.84–1.61), with 13.9 GB
  available and swap 1,074 → 1,066 MB.
- **O1-G7, Phi-3 mini q4k:**
  - 48 projections native; lane device bytes 2,255 MB, of which Q5_K 623 MB and Q6_K 330 MB.
  - int8 23.62–23.77 ms a token, native 23.12–23.50: **median 0.9809, 10 of 10 pairs faster** (42.3 → 43.1 tok/s). It misses the
    ≤ 0.98 bar by 0.0009, so alone it would PARK.
- **O1-G7, Qwen2.5-Coder-1.5B q4_k_m:**
  - 28 projections native.
  - int8 14.25–14.30 ms, native 14.52–14.58: **median 1.0183, 0 of 10 pairs faster**.
- **Verdict by the registered rule: KILL** (the 1.5B is above 1.00).
- **Log:** `~/goinfer-logs/kso1-g7-2026-10-24-1128.log`.
- **O1-G5 (G-Q2 on Phi-3) did not run.** The stopping rule ends the build at G7's verdict.
- **Ungraded reading after G7** (11:33, the lock held exclusively, load 0.80; `TestKSO1_smallShapeReading`). `gemv_w8_f32_rows2`
  → `gemv_q6k_f32_rows<R>` at R = 1 / 2 / 4:
  - the 1.5B's ffn_down (1536×8960): 0.0818 → 0.1125 / 0.0813 / 0.0917 ms;
  - its attn_v (256×1536): 0.0124 → 0.0076 / 0.0092 / 0.0140;
  - Phi-3's ffn_down (3072×8192): 0.1731 → 0.1873 / 0.1407 / 0.1456.
  - O1-G0 fixed R = 4 from Phi-3's shapes alone. On the 1.5B's small-N shapes R = 4 is the slowest native choice, and even the best R
    only ties int8 on its down projection.
- **What the in-model reading says about O1-G0:** a token saves 0.45 ms on Phi-3, where O1-G0's summed kernel times projected about
  3 ms.
- **What the native lane does buy:**
  - Phi-3's lane holds 357 MB less on the device, the file's own bytes. The head stays int8, so its 17.7 MB of the survey's
    375 MB is not saved.
  - It moves the logits toward the file (O1-G6, 0.7638 of the int8 lane's KL).
- **Where the code is:** parked, unmerged, on branch `ks-o1-native-2026-10-24` (88dcfc1d and after). The aikit accessor it needs is
  townsendmerino/aikit#18, also unmerged.

**Owner override of O1-G7's KILL, Francis 2026-10-24 11:47 am (relayed by the coordinator): option C, ship.**
- **The decision:** native Q5_K/Q6_K ships on Metal for its memory and fidelity gains. On Phi-3 the device holds 357 MB less, the
  file's own bytes. On the 0.5B, the logits' KL to the file is 0.7638 of the int8 lane's. Francis accepts the measured 1.8% slowdown
  on Qwen2.5-Coder-1.5B q4k (median 1.0183).
- **What it does not change:** this is his decision, not a re-grading. O1-G7 stays KILL as registered, and `TestKSO1_speed` still
  fails on it.
- **The condition:** O1-G5 (G-Q2 on Phi-3) runs first. If it fails, the change stops for his decision.
- **Amendments with the override, before O1-G5 ran:**
  - O1-G1 adds super-block counts NB = 1, 2, 6, 19 and 35 (K = 256, 512, 1536, 4864, 8960) for both kinds, beside Phi-3's 12 and 32.
  - The step-vs-decode check runs at NB = 1, 6, 12, 19 and 35.
  - The O1-G0 prototype's Q6_K timing now runs the production kernel over the segment-wide planes, which are aligned for every NB.
    The row-strided prototype layout #212's review flagged is gone.
  - Its weights rotate over ≥ 256 MiB of native bytes.
  - Every test dispatch checks its error.
- **Re-run at the merge of #212's fixes:**
  - O1-G1 and the kernel half of O1-G2 PASS at every new shape: worst 1.31e-7 against 1e-5, the plants red at 9.3e-2 to 2.39,
    and 120 of 120 step cells byte-equal.
  - O1-G3 and `TestQ4KLaneResidentQwen` (worst cosine 0.998474) pass.
  - Log: `~/goinfer-logs/kso1-native-tests2-2026-10-24.log`.

- **O1-G5: PASS** (11:53–11:58, the lock held shared, not timed; load 2.19, 13.6 GB available).
  - G-Q2 on the native lane, Phi-3 mini q4k: lane weights 2,255 MB on the device (Q5_K 623, Q6_K 330, Q4_K 1,302).
  - Teacher-forced agreement against the CPU at q4k **98.39%, 8 of 8 prompts** passing the free-run rule, against the reference
    (Qwen2.5-Coder-1.5B) 93.99%, 7 of 8. **Delta +4.39 points** against the −2.0 margin.
  - On the int8 copies C1-G3 read 98.66%. The CPU reference keeps the int8 copies, so a lane closer to the file sits slightly further
    from it.
  - Log: `~/goinfer-logs/kso1-g5-2026-10-24.log`.
- **With O1-G5 passed, KS-O1 ships on Metal under the override.**

**Review fixes for #215, 2026-10-24 (the second-opinion review; no output-changing code):**
- **Metal's fit pricing now prices the native lane.**
  - `ResidentNeedBytes("metal")` and `ResidentDenseWeightBytesFor("metal")` take off `metalKQuantSavings`: the int8 bytes less the
    file's block bytes of every weight the lane uploads natively. The host copy stays int8 and is still priced so.
  - No guard threshold or margin changes; only the device term now matches what the build allocates. It follows from Francis's
    option C, relayed by the coordinator as a guard change. Francis can veto it separately.
  - O1-G3 asserts the pricing's saving equals the device bytes saved, exactly, on both models.
- **The mapping's lifetime.** Only `Load` reaches a q4k GGUF load, and every exit after `loadWeights` goes through `withResidency`
  (which releases it) or `Close`. No path holds it past the load.
  - A CPU-backend load releases it at `withResidency` by design: keeping it would retain the parsed header (51 MB on the 0.5B) for a
    model that never builds a Metal resident.
  - A resident built later in a test maps the file again, and pays that parse.
- **O1-G3 corrected and extended.**
  - The first counter read each scale buffer's `Len()`, an element count, as bytes; it now counts 4 bytes a scale. One discarded
    warm-up load runs before the two arms.
  - **The 0.5B:** 336,119,808 → 326,679,552 bytes; Q6_K exactly the file's 42,900,480; the pricing saves 9,440,256, the device drop.
    Load heap 770.3 → 770.3 MB.
  - **Phi-3 mini, new (`TestKSO1_laneBytesPhi3`), covering Q5_K:** 2,612,330,496 → 2,255,486,976 bytes (**−356.8 MB**). Q5_K is
    exactly the file's 622,854,144 and Q6_K its 330,301,440. The pricing saves 356,843,520, the device drop. Load heap 2,848.8 →
    2,848.5 MB.
  - **PASS on both.** Log: `~/goinfer-logs/kso1-g3b-2026-10-24.log`.
  - The saving is 357 MB, not 375: the head stays int8.
- **O1-G4 on its registered quantity.** `TestKSO1_cpuQ4KFingerprint` now hashes every position's logits bytes, over the prompt and
  16 greedy tokens. It checks errors and the token count, and asserts the digest taken at origin/main 8cfd5fca, `e662bdd6…0bfbe`.
  The tests are tagged `darwin && arm64 && goinfer_testhooks`, because the CPU reference is bit-identical only within an arch.
  **PASS.**
- **O1-G6 does not cover Q5_K.** No model on the Mac stores Q5_K and also fits at f32 for the reference: Phi-3 at f32 is about
  15 GB on a 16 GB machine. Q5_K's fidelity rests on O1-G1 (kernels), O1-G5 (G-Q2 on Phi-3) and O1-G3 (its bytes are the file's).
- **O1-G0 was not re-timed** after the prototype fixes (aligned Q6_K planes, ≥ 256 MiB rotation, dispatch checks). Its readings
  stand as recorded, with the notes above. The kernels themselves were re-graded by O1-G1, G2 and G3.
- **`TestKSO1_speed` changes:**
  - It loads on the CPU backend, so only one resident is built.
  - It checks that the int8 arm's copies fit before allocating them: available memory, the 2 GB margin, and the GPU working set.
  - It swaps every projection group holding a native segment (qkv, o, gate|up, down).
  - It was re-run on the fixed harness: see the second pass below.
- **Also applied:**
  - O1-G1 adds NB = 32 for Q5_K.
  - `TestKSO1_referenceMatchesAikit` fails on an accessor error.
  - `TestKSO1_smallShapeReading` checks its pipelines and dispatches, and rotates ≥ 256 MiB.
  - #212's follow-ups: both run scripts fail when their result line is missing, O1a-G2 compares logits' bits, and the untouched
    guard fails on any element still holding its initial bits.

**Review fixes for #215, second pass, 2026-10-24:**
- **O1-G7 re-run on the fixed harness** (CPU-backend load, every native projection group swapped, the int8 arm fit-checked first;
  binary built at 6eb384d1 on aikit v1.75.0). 13:38–13:40, the lock held exclusively. At the start the 1-minute load was 3.16, under
  the 4.0 line (pairs 1.63–3.04); 13.4 GB available; swap 1,377 → 1,345 MB.
  - **Phi-3 mini:** 48 projection groups native. int8 23.81–24.83 ms a token, native 23.14–23.75: **median 0.9743, 10 of 10 pairs
    faster**.
  - **Qwen2.5-Coder-1.5B:** int8 14.27–14.84, native 14.42–14.57: **median 1.0127, 1 of 10**.
  - **Verdict by the registered rule: still KILL**, because the 1.5B is above 1.00. Phi-3 alone now clears its ≤ 0.98 bar.
  - The pre-fix reading (0.9809 / 1.0183) stays in the record above as what it was. Francis's override stands either way.
  - Log: `~/goinfer-logs/kso1-g7-2026-10-24-1335.log`.
- **The fit saving counts only what the lane uploads natively.** `metalKQuantSavings` sums, on a q4k load for Metal only, the
  recorded weights among each layer's q, k, v, o, gate, up and down projections: exactly the groups `buildResident` hands
  `q4kLaneProj`. A record anywhere else, another backend, or a load that is not q4k (the lane off) prices nothing.
  `TestKSO1_metalSavingsCountsOnlyTheLane` pins all three cases.
- **The permuted-row mapping is tested.** `kquantRows` keeps a run as its start and a permutation row by row. `TestKSO1_kquantRows`
  checks a plain load, a fused split's offset and a permutation; neither real file exercises the permuted branch.
- **O1-G3 checks contents, not only counts.**
  - Layer 0's native segments are compared byte for byte with the file's blocks read through `KQuantReader` (Q5_K verbatim, Q6_K
    through the segment planes): 1 segment on the 0.5B, 2 on Phi-3. **PASS**.
  - The heap comparison allows `kso1HeapSlack`, 1 MiB, for the runtime's own allocations, registered here before this run.
    Measured: the 0.5B 770,301,944 → 770,270,960 bytes, Phi-3 2,848,838,368 → 2,848,522,472.
  - Both arms record provenance (the decoder does not know the Metal test hook). Its cost is a map entry a weight with no row table
    on these files, and the comparison against main is O1-G4's.
  - Log: `~/goinfer-logs/kso1-g3c-2026-10-24.log`.
- **The reference check runs per file:** the 0.5B (12 Q6_K tensors) always, and Phi-3 through the asset registry (32 Q5_K,
  17 Q6_K). **PASS**.

**Notes on O1-G0's instrument, from #212's review (2026-10-24):**
- **Where it lives:** the prototype kernels, `TestKSO1_gemvTiming`, `TestKSO1_protoMatchesRef` and `TestKSO1_q4kSideReading` are on
  branch `ks-o1-native-2026-10-24` only, beside the parked native lane.
- **The prototype's Q6_K layout is valid only when NB is a multiple of 4.** It splits planes per row with a row stride of NB × 210
  bytes, so its `uint2` loads are 8-byte aligned only then. O1-G0 read Q6_K at NB 12 and 32, where the layout holds. The production
  kernels use planes over the whole segment instead (the amendment above).
- **The native weights rotated over less than registered.** The copy count was sized from the int8 bytes, so the native kernels
  rotated over 186 MiB (qkv), 217 MiB (down) and 231 MiB (head), against the ≥ 256 MiB registered. That is still far beyond the
  system cache this rotation is meant to defeat.

**KS-O1a · The q4k lane's Q4_K scale decode without a branch (bit-identical). Registered 2026-10-24 before the code, from an
ungraded side reading beside O1-G0.**
- **Side reading (10:42–10:47, the lock held exclusively, load 1.68 at the start):** the shipped `q4k_rows_acc`, changed only in
  its scale decode to the amended Q5_K's form (`TestKSO1_q4kSideReading`). Shipped R = 8 → new R = 8 / R = 4:
  - o: 0.080 → 0.055 / 0.053 ms;
  - gate|up: 0.348 → 0.236 / 0.240;
  - down: 0.197 → 0.136 / 0.128.
  - Summed over Phi-3's Q4_K GEMVs: 16.85 → 11.51 / 11.43 ms a token (0.683 / 0.678).
  - In this library, 0 of N outputs differed in bits at any shape. Log: `~/goinfer-logs/kso1-q4kside-2026-10-24.log`.
- **The change:**
  - `q4k_scale_min2` reads the 12 scale bytes as three words and decodes a lane's two scales and mins with masks and shifts. It
    replaces the two `q4k_scale_min` calls in `q4k_rows_acc` and in the batched step's `mc3_gemv_q4k_f32_rows`.
  - R stays 8: R = 4 is within 2% (0.7%), and the rule takes the smaller R only on a tie, so KS-C1's R stands unless this changes it.
  - The scale values are the same integers, so each output's arithmetic is unchanged.
- **Class:** bit-identical, under the small-gains rule.

| gate | tier | instrument | PASS iff |
|---|---|---|---|
| O1a-G1 bits | 0 | the full decode library before and after (the old one rebuilt from the new source by swapping the decode text back), `gemv_q4k_f32_rows<R>` plain / residual / bias at R = 1, 2, 4, 8. Shapes: a partial threadgroup and Phi-3's o, gate\|up and down. Random blocks cover all 64 values of every 6-bit scale and min. Both the fast-math and the precise library. The four `TestQ4KStep_*` tests and C1-G2's kernel-level step check | byte-equal everywhere; the step tests as they assert today |
| O1a-G2 model bits | 1, Mac | inside O1a-G3's run: each arm's 128 greedy tokens and the final logits' bytes | identical in every run |
| O1a-G3 speed | timed, Mac, timing lock | C1-G4's A/B (`TestKSC1_phi3Speed`'s shape): one Phi-3 mini q4k resident, its three Q4_K pipelines swapped between the old library and the new one, 128 greedy decode tokens after a 32-token prompt, A B B A over 10 pairs after 2 warm runs each | SHIP iff O1a-G1 and G2 pass, memory is unchanged (no buffer is added or resized) and at least 9 of 10 pairs are faster (the small-gains rule). The registered C1-G4 bar (median ≤ 0.98) is reported alongside. Fewer than 9 pairs faster: PARK |

Expected cost: G1 seconds. G3 about 3 min, from C1-G4's wall, after Phi-3's fit check. Stopping rule: one pass.

**KS-O1a read 2026-10-24, by day: SHIP.**
- **O1a-G1: PASS.**
  - The decode rows kernels at R = 1, 2, 4, 8 (every variant) and the step's `mc3_gemv_q4k_f32_rows<B>` at B = 1, 2, 4, 8 (every
    mode) are byte-equal before and after in 192 cells, in both the fast-math and the precise library (`TestKSO1a_q4kBitsUnchanged`).
  - The four `TestQ4KStep_*` tests and C1-G2's kernel-level step check pass.
  - G-Q1 passes. Its plant (2) is retargeted at `q4k_scale_min2`, since the old text no longer reaches the rows kernel; it is red at
    2.09.
  - Log: `~/goinfer-logs/kso1a-g1-2026-10-24.log`.
- **O1a-G3 run:** 10:53–10:55, the lock held exclusively. At the start the 1-minute load was 3.28, under the 4.0 line; each pair
  started at 1.16–2.33; 13.9 GB available; swap 1,138 MB before and after.
- **O1a-G3 result:** old 29.55–31.06 ms a token, new 23.75–27.17; **median new/old 0.8044, 10 of 10 pairs below 1** (33.4 → 41.7 tok/s
  at the median pair). No buffer is added or resized. The run passes the small-gains rule and C1-G4's registered bar (≤ 0.98): **SHIP**.
- **O1a-G2: PASS.** All 24 runs, both arms, gave the same 128 tokens and the same final logits' bytes.
- **Log:** `~/goinfer-logs/kso1a-g3-2026-10-24-1053.log`. Binary built at f9f41aa5 from
  `docs/measurements/ks-night-2026-10/run-kso1a-g3.sh`.

**KS-O1 on CUDA: gates, registered 2026-10-24 before any product code (O1C-G), on the owner's word of 4:37 pm ("#2 then #1":
KS-A1's CUDA half, then KS-O1 on CUDA, the twin of what shipped on Metal).** Each kind is built only if it passed A1-G0's rule
("KS-A1 on CUDA"); a kind A1-G0 kills stops for the owner's decision before any code for it. The build:
- **Kernels** (`cuda/actgroup.cu`): per-row `rowQ5K` and `rowQ6K` bodies in A1-G0's chosen lane map, the per-32 int8 activations
  and the explicit `__fmaf_rn` / `__fmul_rn` form of `rowQ4K`. `projRow` and `finishRow` gain two kinds (`WK_Q5K`, `WK_Q6K`), so the
  fused per-32 kernels (`fused_rms_qkv_g32`, `fused_rms_gu_g32`, used at hidden sizes up to `fusedG32MaxHidden`) take them, and
  standalone `gemv_q5k_g32` / `gemv_q6k_g32` at A1-G0's R serve `doG`. Q5_K is uploaded in the file's own 176-byte blocks (16-byte
  aligned); Q6_K in planes over the whole weight (ql, qh, int8 scales, d in one buffer, offsets from the weight's block count), the
  file's exact bytes, aligned for every block count (Metal's segment-wide layout, which fixed the odd-NB misread there).
- **Data:** at `BuildResident`, from the GGUF's own blocks through `decoder.Model.OpenKQuant` / `KQuantReader`, streamed through the
  existing staging pool (`streamW`), so the host holds one staging block, not a native copy. The CPU keeps the int8 copies; the CPU
  path is unchanged.
- **Scope:** every per-layer projection the CUDA resident plans (q, k, v, o, gate, up, down) and the **untied LM head**
  (`output.weight`; on CUDA the head is an ordinary `doG` launch, so the launch-shape cost that kept Metal's head int8 does not arise).
  A tied head (the embedding table) stays int8. `noteKQuant` also records the untied head; Metal's lane does not read that record
  (its head is uploaded outside `q4kLaneProj` and `metalKQuantSavings` counts layer projections only), so Metal is unchanged.
- **One precondition on scope:** the native kinds are taken only under per-32 activations (`ActQuantGroup` 32), whose group sums
  the kernels read. Every `--quant q4k` load gets it (`modelFromOptions`), and under it CUDA already declines the batched prefill,
  the drafter and LoRA (`prefillStaticDecline`, `NewDrafter`, `SetAdapter`), so the native kinds take no batched path away.
- **Fit pricing:** `ResidentDenseWeightBytesFor("cuda")` (which the CUDA build's KV plan and `Plan` both read) takes off, per weight
  the resident uploads natively and only those, its int8 bytes less the file's block bytes. The host term is unchanged. No threshold
  or margin moves: the device term is made to equal what the build allocates (#215's Gemini finding: price native blocks only where
  they are actually uploaded).
- **Why not bit-identical:** as on Metal, the native kernels compute the file's own weights instead of an int8 re-quantization, so
  the small-gains rule's bit-identity clause cannot hold; the fidelity gates below decide, and the speed gate takes the small-gains
  rule's pair count.

| gate | tier | instrument | PASS iff |
|---|---|---|---|
| O1C-G1 kernels | 0 | O1-G1's method on CUDA: every new kernel (standalone at every built R, and each kind through both fused kernels) against a float64 host reference that dequantizes the same blocks by ggml's formulas, with the same per-32 int8 activations; the reference cross-checked against aikit's GGUF dequantization on the 0.5B q4_k_m's Q6_K tensors and Phi-3's Q5_K and Q6_K tensors. Super-block counts NB = 1, 2, 6, 12, 19, 32, 35 and a partial block of rows. Plants: (1) Q5_K's high bit dropped; (2) Q6_K's −32 offset dropped; (3) a Q6_K group read with its neighbour's scale; (4) a super-block offset of one | every row within G-Q1's bar (\|err\| ≤ 1e-5 of Σ\|w·a\|); the reference within 1e-6 of aikit's dequantization; each plant red |
| O1C-G2 untouched bits | 0 | `actgroup.ptx` at origin/main against the new one: every existing kernel (`gemv_w8a8_g32`, `gemv_w4a8_g32`, `gemv_q4k_g32`, both fused kernels with int8 / int4 / Q4_K projections at rows-per-warp 1, 2, 4) on random inputs; `TestActGroupKernels_fusedG32BitIdentical` extended with the new kinds mixed into a group; `TestQ4K_fusedG32ResidentBitIdentical` | the old kinds' outputs byte-equal old against new; fused equal to unfused, byte for byte, with the new kinds |
| O1C-G3 memory | 1 | through `decoder.Load` on CUDA, the 0.5B q4_k_m and Phi-3 mini q4: the resident's weight bytes by kind, native arm against an int8 arm (a test hook keeping the copies); layer 0's native buffers and the head read back and compared byte for byte with the file's blocks (Q6_K through the planes); the pricing's saving; the host heap over the load | each native buffer holds exactly the file's bytes for its weight; device bytes fall by exactly the pricing's saving (Phi-3: about −375 MB, the head included); the host heap not higher than the int8 arm's plus 1 MiB |
| O1C-G4 CPU unchanged | 1 | the CPU q4k forward on the 0.5B q4_k_m (Q6_K in half its layers): every position's logits bytes over a prompt and 16 greedy tokens, hashed, against the digest taken at origin/main on this box (amd64) | byte-equal |
| O1C-G5 agreement | 2 (by day) | the CUDA G-Q2: `g3Run` (CUDA's G3c procedure) on Phi-3 mini at `--quant q4k`, the CUDA resident on the native lane against the CPU at q4k on the same file, with Qwen2.5-Coder-1.5B as G3c runs it as the reference in the same process. Beside it, `TestQ4K_phi3ResidentMatchesCPU` by its own bars | PASS iff agreement ≥ the reference's − 2.0 points and free-run passes ≥ the reference's − 1; 2.0–4.0 points below is ambiguous (parked); worse fails. `TestQ4K_phi3ResidentMatchesCPU` must pass both its bars; if only its CPU-agreement bar fails (the CPU keeps the int8 copies, so a lane nearer the file sits further from it), that is the owner's call, not a pass |
| O1C-G6 toward the file | 1 | Phi-3 mini q4 (Q5_K and Q6_K) and the 0.5B q4_k_m (Q6_K): the CUDA q4k resident's last-row logits on the int8 arm and the native arm, each against the CPU at f32 on the same GGUF (the file exactly), over G3's 8 prompts teacher-forced. Phi-3 at f32 is about 15 GB on the CPU, inside this box's 62 GB, so Q5_K is covered here, unlike on Metal | for each model, mean KL(native) ≤ mean KL(int8); up to 1.10× is ambiguous (parked); above fails |
| O1C-G7 speed | timed, nobara, timing lock | C1-G4's A/B on CUDA: one Phi-3 mini q4k resident holding both the int8 and the native buffers of every natively uploaded weight (a test hook), swapped per run; 128 greedy decode tokens after a 32-token prompt, A B B A over 10 pairs after 2 warm runs each; then the same on Qwen2.5-Coder-1.5B q4_k_m (Q6_K only). Load and free memory logged per cell, CLAUDE.md's void rule | **SHIP** iff Phi-3 is faster on at least 9 of 10 pairs (the small-gains rule's count; device memory falls, so its memory clause holds) **and** the 1.5B's median native/int8 is ≤ 1.00. **PARK** if Phi-3 is faster on 6–8 pairs. **KILL** if Phi-3 is faster on 5 or fewer, or the 1.5B's median is above 1.00. A KILL or PARK goes to the owner (Metal shipped under his option C with the 1.5B at 1.0127) |

Fit checks: every load's plan is read from the build's own `fit:` / `KV plan:` banner first; a refused load is not started. Order:
G1, G2, G4 (seconds), G3, then G7 (timed, under the lock), then G6 and G5. Stopping rule: one pass each; a FAIL in G1 to G4 stops
the build for a fix, and the fixed build re-runs every gate from G1. Expected cost: G1–G4 a few minutes; G3 about 2 min; G7 about
6 min (two models); G6 about 10 min (Phi-3 at f32 on the CPU); G5 about 15 min (two models, G3c's procedure).

**O1C amended 2026-10-24, 16:58, before any O1C gate ran:** the scope guardrail first registered here ("only when the resident also
holds a Q4_K projection", so as not to take a batched prefill away) protected nothing. Every q4k load runs per-32 activations, under
which CUDA has no batched prefill, drafter or LoRA, with or without a Q4_K tensor. The precondition is now per-32 activations, which
the native kernels need for their group sums. The gates are unchanged.

**KS-O1 on CUDA read, 2026-10-24, by day.** Run by `night.py serve` (the timing lock held exclusively) from
`docs/measurements/ks-a1-2026-10/run-o1c-day.sh`, base origin/main 4814c869a, NVIDIA driver 595.91.07.
- **First pass (437124772, 17:14): three failures, so the build was fixed and every gate re-ran from G1.** Log:
  `docs/measurements/ks-a1-2026-10/` holds only the second pass; the first is `~/goinfer-logs/ks-a1-2026-10/o1c-t0-2026-10-24-1714.log`.
  - **A crash in main's decoder, from #215.** `Weights.noteKQuant` wrote the `Weights.kquant` map from `parallelLayers`' goroutines
    with no lock. Phi-3's q4k load died with `fatal error: concurrent map writes` (in the phi3 fused-split loader). The code is
    backend-neutral, so a Metal or CPU load of the same file is exposed too; it is a race, so not every load. Fixed: the record is
    taken under a package-level mutex (`kquantMu`).
  - **The f16 reference dropped a subnormal's sign** (`ksA1HalfVal`, test code): a real Q6_K block of the 0.5B has a negative
    subnormal `d`, and the reference read it positive, so `TestKSO1C_referenceMatchesAikit` failed against aikit. The kernels use
    `__half2float` and were not affected; the random blocks of A1-G0 and O1C-G1 use positive normal halves only.
  - **The FMA lint read pointer offsets as bare MACs** in `rowsQ6K` and `gemv_q5k_g32`. They are written as named offsets now, the
    lint's idiom; the arithmetic is unchanged.
- **Second pass (e3bd04a0d, 17:18; load 0.6–0.9).** Log: `docs/measurements/ks-a1-2026-10/o1c-t0-2026-10-24-1718.log`.
  - **O1C-G1: PASS.** `TestKSO1C_nativeKernels`: worst |err|/Σ|w·a| 1.51e-7 (Q5_K) and 1.44e-7 (Q6_K) against 1e-5, over NB = 1, 2, 6,
    12, 19, 32, 35 and 29 rows (a partial block of rows), with bias and accumulate. All four plants are red, the smallest at 0.16. CUDA's
    PTX is built offline, so the plants are carried by the reference: a kernel that made the planted mistake would match the planted
    reference. `TestKSO1C_referenceMatchesAikit`: within 1e-6 of aikit's dequantizer on 12 Q6_K tensors of the 0.5B and 32 Q5_K and 17
    Q6_K tensors of Phi-3.
  - **O1C-G2: PASS.** 22 cells byte-equal between origin/main's `actgroup.ptx` and the new one (the quantizer, the three standalone
    GEMVs, both fused kernels with int8, int4 and Q4_K projections at rows-per-warp 1, 2, 4). The fused kernels with Q5_K and Q6_K in a
    group equal the standalone GEMVs byte for byte (`TestKSO1C_fusedEqualsUnfused`). `TestQ4K_fusedG32ResidentBitIdentical` (the 1.5B,
    whose Q6_K `attn_v` now runs native inside `fused_rms_qkv_g32`): 40 positions bit-identical fused against unfused. The FMA lint,
    the local-memory census and the PTX consistency checks pass.
  - **O1C-G4: PASS.** The CPU q4k digest on the 0.5B is `8bd2324c…5a89` at the base and at e3bd04a0d (amd64).
  - **O1C-G3: PASS.** Through `decoder.Load` on CUDA at `--quant q4k`, after a discarded warm-up load:

    | model | int8 arm, device bytes | native arm | saving = pricing's | heap, int8 → native |
    |---|---:|---:|---:|---:|
    | Qwen2.5-Coder-0.5B q4_k_m | 472,862,208 | 463,421,952 (Q6_K 42,900,480) | 9,440,256 | 803.68 → 803.72 MB |
    | Phi-3 mini q4 | 2,710,959,360 | 2,336,288,256 (Q5_K 622,854,144; Q6_K 411,102,720) | **374,671,104** | 2,878.91 → 2,879.37 MB |

    Phi-3's weights on the device are now the file's own 2,336.29 MB, the survey's figure, with the head (80.8 MB of the Q6_K bytes)
    included. Each native buffer holds exactly the file's bytes for its weight; layer 0's native buffers and the head read back equal
    to the file's blocks (1 on the 0.5B, whose head is tied and stays int8; 5 on Phi-3). The heap rose by 0.04 and 0.46 MB, inside
    the 1 MiB slack.
- **O1C-G7: SHIP** (e3bd04a0d, 17:29–17:30, the lock held exclusively; load 0.34 at the start and 0.56–1.11 at each pair, 54.6–56.4 GB
  available, no void pair; the desktop compositor's context on the card). Log: `docs/measurements/ks-a1-2026-10/o1c-g7-2026-10-24-1729.log`.

  | model | weights native | int8 arm | native arm | median native/int8 | pairs faster |
  |---|---:|---:|---:|---:|---:|
  | Phi-3 mini q4 | 113 (q, k, v ×32, 16 `ffn_down`, the head) | 8.696–8.716 ms (114.9 tok/s) | 7.847–7.861 ms (127.3 tok/s) | **0.9025** | **10 of 10** |
  | Qwen2.5-Coder-1.5B q4_k_m | 29 | 4.952–5.004 ms (200.2 tok/s) | 4.870–4.896 ms (204.6 tok/s) | **0.9786** | 10 of 10 |

  - Phi-3 saves 0.85 ms a token, against A1-G0's kernel-sum bound of about 0.9 ms. On Metal the same change saved 0.45 ms against a
    kernel-sum projection of about 3 ms, and the 1.5B was slower (1.0127); on CUDA the 1.5B is faster.
  - Each arm's 128 greedy tokens were the same in every run. The two arms agree for the first 40 tokens on Phi-3 and all 128 on the
    1.5B: different weights (the file's own against an int8 copy), so a divergence is expected; G5 and G6 grade it.
- **O1C-G6: PASS** (e3bd04a0d, 17:31–17:37, the lock held exclusively; not timed). Against the CPU at f32 on the same file, every
  teacher-forced position over G3's 8 prompts and the CPU's own 32-token continuations:

  | model | positions | mean KL, int8 arm | native arm | native / int8 | native lower at |
  |---|---:|---:|---:|---:|---:|
  | Qwen2.5-Coder-0.5B q4_k_m (12 weights native) | 566 | 0.00223 | 0.00203 | **0.9114** | 333 |
  | Phi-3 mini q4 (113 weights native, Q5_K covered) | 372 | 0.01570 | 0.00614 | **0.3912** | 273 |

  Native is lower on 7 of 8 prompts on the 0.5B and on all 8 on Phi-3. Metal's O1-G6 read 0.7638 on the 0.5B and could not cover
  Q5_K.
- **O1C-G5: the G-Q2 half PASS; `TestQ4K_phi3ResidentMatchesCPU` FAIL on both its bars, so O1C-G5 FAILS as registered.**
  - G-Q2 (17:37–17:40): the reference Qwen2.5-Coder-1.5B 96.64% (547 of 566), 8 of 8 prompts; Phi-3 on the native lane **96.77%
    (360 of 372), 8 of 8: delta +0.13 points** against the −2.0 margin. The CUDA reference reads 96.64%, not Metal's 93.99%, under
    ChatML's default system message (the G3c record).
  - `TestQ4K_phi3ResidentMatchesCPU` (one 141-token prompt: 110 × "the ", then an instruction; context 4096): CUDA q4k against CPU q4k
    median cosine **0.993030** (bar ≥ 0.999); CUDA q4k against f32 p10 **0.882586** against the CPU q4k path's 0.926208 (bar: no more than
    0.01 below). Log: `docs/measurements/ks-a1-2026-10/o1c-g56-2026-10-24-1731.log`.
  - **On main's int8 lane the same test passes** (median 0.999804; p10 0.940085). Main itself could not run it: the Phi-3 load hit the
    `noteKQuant` race, so the baseline is main with only the mutex applied, uncommitted, in a base worktree. Log:
    `docs/measurements/ks-a1-2026-10/o1c-g5-base-int8lane-1742.log`.
- **Diagnosis (not graded, run after the verdict above; `TestKSO1C_fillerDiagnostic`, both arms of one resident at context 512, the CPU
  at f32 on the file, the shared lock).** It reproduces the two arms' p10 to six figures at context 512, so the KV plan is not the
  cause.
  - **Where:** native is closer to f32 at 76 of the 110 filler positions (median cosine 0.9830 → 0.9868) and further at 27 of the 29
    instruction positions after them (0.9712 → 0.9231). Mean KL to f32 over the prompt: int8 0.427, native 0.510.
  - **Which weights:** only the Q5_K qkv native, or only the Q6_K `ffn_down` native, each gives the same instruction-region median
    (0.9236, 0.9237); only the head native changes nothing (0.9712). Two independent perturbations landing on one value is a bifurcation
    of a degenerate repeated context, not a per-weight error.
  - **Other filler lengths** (`GOINFER_KSO1C_DIAG_REPS`; instruction-region median and mean KL to f32, int8 → native): 60 repeats
    0.9987 → 0.9984, KL 0.275 → 0.296; 90 repeats 0.9966 → 0.9959, KL 0.321 → 0.310; 130 repeats 0.9923 → 0.9927, KL 0.330 → 0.314.
    The 110-repeat prompt the test pins is the outlier.
  - **The p10:** native's p10 is lower at every length (0.963 / 0.959 / 0.967 against 0.971 / 0.970 / 0.976). It comes from a few
    early filler positions (position 7, the sixth "the", reads cosine 0.28 native against 0.80 int8 at every length), where the
    next-token distribution after a run of "the" is flat and cosine is a poor reader.
  - **The CPU-agreement bar** cannot hold for a lane nearer the file: the CPU q4k reference keeps the int8 copies, which is the case
    this registration named as the owner's call.
  - Logs: `docs/measurements/ks-a1-2026-10/o1c-filler-diag-*.log`.

**Owner decision on O1C-G5, Francis 2026-10-24 (option A): accept, and ship KS-O1 on CUDA.**
- **The decision:** native Q5_K/Q6_K ships on the CUDA q4k lane on G7's SHIP, G6's KL and G-Q2's +0.13 points. This is his decision,
  not a re-grading: O1C-G5 stays FAIL as registered.
- **`TestQ4K_phi3ResidentMatchesCPU` now runs the int8 lane** (`decoder.SetKQuantNativeOffForTest` around its load, and an assertion
  that no weight is native), so its two bars keep guarding that lane. On this branch it reads main's numbers exactly: median 0.999804,
  p10 0.940085, **PASS**. Log: `docs/measurements/ks-a1-2026-10/o1c-phi3-int8lane-branch.log`. The native lane's fidelity gates are
  `TestKSO1C_realPhi3NonInferiority` and `TestKSO1C_towardTheFile`.

**#230's review, item 1 (Gemini, relayed by the coordinator, 2026-10-24 22:40): a long-context f32 quality bar for the native lane.
Registered 2026-10-24 before any of it ran.** The review is right that the record overstated the diagnosis. The committed logs show
the native lane's p10 cosine to f32 below the int8 lane's at all four filler lengths (60, 90, 110, 130). They also show its mean KL
to f32 worse at 60 and 110 repeats, and "only Q5_K qkv native" worse on KL at all four. The PR text's "60/90/130 do not show it" held
only for the instruction region's median.
- **LQ1, the old bar on the native lane (tier 1, Phi-3, the shared lock):** `TestQ4K_phi3ResidentMatchesCPU` runs both lanes. The
  int8 lane keeps both its bars. The native lane takes the f32 quality bar unchanged: p10 cosine to f32 no more than 0.01 below the CPU
  q4k path's own on the same prompt. The CPU-agreement bar does not apply to it (the CPU keeps the int8 copies).
- **LQ2, the filler table (tier 1, `TestKSO1C_fillerDiagnostic` at 60, 90, 110, 130 repeats):** for the int8 arm, the native arm and
  the CPU q4k path, each against the CPU at f32: p10 and min cosine, and mean KL. The old bar is applied at every length.
- **LQ3, natural long context (tier 1, the same diagnostic on a document):** two in-repo English documents, `README.md` and
  `docs/benchmarks.md`, each cut to its first 900 tokens inside "Summarize this document:" under Phi-3's chat template (about 920
  positions each, context 1024). Per document: mean KL to f32 over all positions and over positions 256 and later; p10 and min
  cosine over the same.
- **The reading, fixed now:** the native lane is **worse than int8 at long context** if, on either document, its mean KL to f32 over
  positions 256 and later is above the int8 arm's by more than 5%. Otherwise the gap is the filler's alone. Either way the
  work stops for the owner, with these numbers and the options. Shipping is not re-decided by this reading.
- **Cost:** LQ1 about 4 min (two Phi-3 loads, three CPU passes over 141 tokens). LQ2 about 1.5 min a length. LQ3 about 8 min a
  document: the CPU at f32 runs at about 0.5 s a position.

**LQ1–LQ3 read 2026-10-24, 22:44–23:19, by day** (d0b69c56a; the shared lock, not timed; load under 1.2). Log:
`docs/measurements/ks-a1-2026-10/lq-2026-10-24.log`.
- **LQ1:** `TestQ4K_phi3ResidentMatchesCPU`, int8 lane **PASS** (median 0.999804, p10 to f32 0.940085). Native lane **FAIL** on the f32
  bar: p10 **0.882586** against the CPU q4k path's 0.926208 (floor 0.9162), min 0.029366.
- **LQ2, the filler at four lengths** (p10 / min cosine to f32, mean KL to f32; the old bar is native p10 ≥ the CPU q4k p10 − 0.01):

  | repeats | CPU q4k | int8 arm | native arm | only Q5_K native | old bar on native |
  |---:|---|---|---|---|---|
  | 60 | 0.9691 / 0.788, 0.284 | 0.9707 / 0.798, 0.275 | 0.9627 / 0.281, **0.296** | 0.9516 / 0.302, 0.338 | pass (floor 0.9591) |
  | 90 | 0.9663 / 0.788, 0.327 | 0.9704 / 0.798, 0.321 | 0.9593 / 0.281, 0.310 | 0.9516 / 0.302, 0.369 | pass (0.9563) |
  | 110 | 0.9262 / 0.435, 0.449 | 0.9401 / 0.454, 0.427 | 0.8826 / 0.029, **0.510** | 0.9079 / 0.058, 0.545 | **FAIL** (0.9162) |
  | 130 | 0.9754 / 0.788, 0.337 | 0.9765 / 0.798, 0.330 | 0.9666 / 0.281, 0.314 | 0.9545 / 0.302, 0.395 | pass (0.9654) |

  On this filler the native arm's p10 and min are below the int8 arm's at every length, and its mean KL is worse at 60 and 110 and
  better at 90 and 130. "Only Q5_K native" is worse on every measure at every length. The review's reading of the logs is right;
  the PR text's "60/90/130 do not show it" was wrong for p10, min and the KL at 60, and is corrected.
- **LQ3, natural long context** (912 positions each; mean KL to f32 over all positions / over positions 256 and later; p10 and min
  over 256 and later):

  | document | CPU q4k | int8 arm | native arm | only Q5_K native | native / int8 past 256 |
  |---|---|---|---|---|---:|
  | `README.md` | 0.00304 / 0.00187 | 0.00296 / 0.00173; p10 0.99941, min 0.9821 | **0.00148 / 0.00068**; p10 0.99973, min 0.9597 | 0.00190 / 0.00089 | **0.39** |
  | `docs/benchmarks.md` | 0.00243 / 0.00168 | 0.00247 / 0.00175; p10 0.99936, min 0.9931 | **0.00105 / 0.00081**; p10 0.99968, min 0.9917 | 0.00153 / 0.00096 | **0.46** |

  **By the reading fixed before the run, the native lane is not worse than int8 at long context.** Its KL past position 256 is 0.39×
  and 0.46× the int8 arm's, against the +5% line. "Only Q5_K native" also beats int8 on both documents (0.51×, 0.55×). The one
  statistic that favours int8 is the minimum past 256: on `README.md` native reads 0.9597 against 0.9821, and on
  `docs/benchmarks.md` 0.9917 against 0.9931. One position sets each minimum.

**Owner decision on #230's long-context bar, Francis 2026-10-25 6:11 am (relayed by the coordinator): option A.**
- **The gate:** the native lane's long-context quality gate is `TestKSO1C_longContextQuality`. On `README.md` and `docs/benchmarks.md`,
  each cut to its first 900 tokens, the native lane's mean KL to f32 over positions 256 and later must be at most 1.05× the int8
  lane's. `TestQ4K_phi3ResidentMatchesCPU` and its filler bar stay on the int8 lane.
- **Stated plainly, as the decision requires:**
  - **The native lane does worse than int8 on highly repetitive input.** On the 110-repeat filler its p10 cosine to f32 is 0.8826
    against the old bar's floor of 0.9162, and its minimum is 0.029. Its p10 and minimum are below int8's at every filler length
    measured.
  - **Its single-position minimum on `README.md` is lower:** 0.9597 against int8's 0.9821.
- **Open item:** KS-O1-F in the ledger.
- **The gate's first run: PASS** (e3a66d6d9, 06:13–06:33, the shared lock, not timed). On `README.md`, mean KL to f32 past position
  256 is 0.00173 int8 against 0.00068 native, **0.3899**. On `docs/benchmarks.md` it is 0.00175 against 0.00081, **0.4623**. These are
  LQ3's numbers to the digit. Log: `docs/measurements/ks-a1-2026-10/longctx-gate-2026-10-25.log`.

**KS-O1-F diagnostic, registered 2026-10-25 before any data (owner, 1:52 pm).** The question: why the CUDA native lane's logits are
further from f32 than the int8 lane's on highly repetitive input (Phi-3 mini q4, 110 × "the ": p10 0.8826, min 0.029, against
int8's 0.9401 and 0.454), while about 2× closer on natural text (LQ3). Phi-3 mini q4 `Phi-3-mini-4k-instruct-q4.gguf`, the CUDA
resident at `--quant q4k` holding both arms (`kso1cLoadBothArms`), against the CPU at f32 on the same GGUF. Instrument: a scratch
harness beside `cuda/kso1c_fidelity_test.go:TestKSO1C_fillerDiagnostic`, kept as text in `docs/measurements/ks-o1-f-2026-10-25/`.
Tier 1, the shared timing lock, not timed; about 5 minutes a run.
- **D1, where they diverge.** The same prompt (TestQ4K_phi3ResidentMatchesCPU's filler at 110 repeats under Phi-3's chat template),
  token by token through `cudaResident.Forward` with every layer's output captured (`cudaResident.SetHiddenCapture`), for four
  arms: int8, native, only Q5_K qkv native, only Q6_K ffn_down native. The CPU at f32 over the same ids with every layer captured
  (`decoder.KVCache.SetCaptureLayersForTest` on its batched prefill, bit-identical to its per-token walk). Per layer and arm: the
  minimum and p10 over positions of the layer output's cosine to f32, its relative L2, and the position of the minimum. Reported:
  the first layer at which the native arm's minimum falls below the int8 arm's by more than 0.05, and at which positions.
- **D3, the KV path.** D1's native and int8 arms rerun with `--kv f32` (explicit). The KV path is implicated if the native arm's
  final-layer minimum moves by more than half its gap to the int8 arm's.
- **D3 amended (14:34, D1 running, none of its output read).** The figures this diagnostic explains were read on the f32 cache: the
  CUDA test suite's default is f32 KV (`cuda/kv16_default_test.go`), and every LQ log reads `layer 0 KV half false`. KV precision is
  therefore excluded by construction, and D3 as registered is D1's own configuration: it runs as an A/A repeat of D1 and is reported
  as that (each layer's minimum must repeat to within 0.001, or D1's readings are not deterministic enough to locate anything). The
  rest of the KV path, cached K and V that differ because each arm's own projections wrote them, is inside D1's per-layer reading.
- **D2, which numerics, registered after D1 and before it runs.** At D1's first divergent layer, the suspect projection's output on
  the same input computed four ways (f32 file weights × f32 activations; file weights × per-32 int8 activations; the int8 lane's
  weights × per-32 int8 activations; the native kernel itself), to split activation quantization, weight representation and
  accumulation.
- **Stopping rule:** stop when one factor reproduces the native arm's gap at the first divergent layer to within 20%; the rest are
  reported as not run. Any fix is a numerics change and gets its own registered fidelity gate and a Gemini review.

**KS-O1-F D1 and D3 read (14:32–14:35, the shared lock, not timed).** Log: `docs/measurements/ks-o1-f-2026-10-25/d1.log`, `d3.log`.
- **The figures reproduce:** at the logits, native p10 0.8826, min 0.0294; int8 0.9401, 0.4543 (LQ2's to the digit).
- **D3, the A/A repeat:** every printed figure equal to D1's (largest difference 0), so the per-layer readings are deterministic.
- **The registered metric names layer 31:** the first layer where the native minimum is below int8's by more than 0.05 (0.1528 at
  position 122 against 0.5559; positions 7 and 114–122). Below layer 31 the minimum is a different thing: from layer 4 on, both
  arms' minimum is position 8 (a massive-activation position, relative L2 above 1 in both arms) and the native arm's is the higher.
- **The p10 tells where the gap opens:** native against int8, p10 cosine 0.9962 against 0.9979 at layer 2, 0.9824 against 0.9952 at
  layer 4, 0.8939 against 0.9772 at layer 8, then flat (0.87–0.90 against 0.95–0.97) to layer 30. Each one-group arm (only the Q5_K
  qkv native, only the Q6_K ffn_down native) carries about half of it at every layer from 4 on (layer 8: 0.9192 and 0.9191).
- **What this rules out:** one tensor kind. Both kinds open about half the gap, starting at the same layers; Q6_K has no minimum term,
  so the Q5_K minimum term is not the cause.

**D2, registered 2026-10-25 after D1's read and before it runs.** D1 places the gap's opening at layers 2–8 in both native kinds, so
D2 runs there as well as at the registered layer 31. D2a is the Q5_K qkv projection; D2b (ffn_down, Q6_K) is registered only if D2a
does not stop.
- **Input:** the native arm's own residual after layer L−1 at every one of the 141 positions (`SetHiddenCapture`), RMS-normed in
  float64 with layer L's `attn_norm` weight and the GGUF's epsilon, then rounded to float32: the qkv input x. Layers L = 2, 4, 8, 31.
- **Five outputs per row and position:** (1) the file's weights (GGUF `attn_qkv`, dequantized in float64) × x, the truth; (2) the
  file's weights × the per-32 int8 activation (`quantG32Go`'s codes × scales), in float64; (3) the int8 lane's weights (the resident's
  own int8 copies, downloaded) × the same activation, in float64; (4) `gemv_q5k_g32` on the file's blocks and that activation, on
  the GPU; (5) `gemv_w8a8_g32` on the int8 copies and that activation, on the GPU.
- **Read per position, over each of q, k and v separately:** e(i) = ‖(i) − (1)‖ / ‖(1)‖. Reported: the median and the p90 over
  positions of each e, of ‖(4) − (2)‖/‖(2)‖ and ‖(5) − (3)‖/‖(3)‖ (kernel arithmetic), and the same at positions 7 and 114–122.
  Sanity, before any reading counts: the int8 copies' rows match the file's within 10% relative L2 (a wrong row mapping reads
  about 1.4). Amended 16:28, before any D2 figure was read: the first run stopped at this check at 1% with 1.73% on layer 2's q,
  which is per-row int8 rounding of these rows, not a mapping error; 1% was set without that arithmetic.
- **Attribution, at the projection:** the native gap is e(4) − e(5). Kernel arithmetic explains it if ‖(4) − (2)‖/‖(2)‖ is at least
  0.8× the gap; activation quantization with the file's weights explains it if e(2) − e(3) is within 20% of it. If e(4) ≤ e(5) (the
  native projection is no worse than int8's on the same input), no per-projection factor explains the gap: that is the finding,
  and the next step is a question for Francis, not a fourth run.
- **Cost:** one resident load with both arms, one GPU pass of 141 tokens, about 30 s of float64 a layer: about 5 minutes. The shared
  lock, not timed.

**KS-O1-F D2a read (18:31–18:39, nobara's night queue, the exclusive lock, only the compositor on the card).** Log:
`docs/measurements/ks-o1-f-2026-10-25/d2a.log`; harness `proj_test.go.txt`. Two attempts before it are kept beside it:
- `d2a-1-oom.log`: out of device memory beside another session's serve, under the shared lock.
- `d2a-2-sanity.log`: stopped at the 1% sanity bar, amended above before any D2 figure was read.

Sanity: the int8 copies are the file's rows at 0.86–1.78% relative L2 at every layer and segment, which is per-row int8 rounding.

| layer | segment | e2 file × aq | e3 int8 × aq | e4 native kernel | e5 int8 kernel | e4 − e5 |
|---|---|---|---|---|---|---|
| 2 | q / k / v | 0.00243 / 0.00298 / 0.0290 | 0.00409 / 0.00459 / 0.0354 | = e2 | = e3 | −0.0017 / −0.0016 / −0.0064 |
| 4 | q / k / v | 0.00194 / 0.00374 / 0.0522 | 0.00272 / 0.00514 / 0.0681 | = e2 | = e3 | −0.0008 / −0.0014 / −0.0159 |
| 8 | q / k / v | 0.00266 / 0.00481 / 0.0270 | 0.00377 / 0.00659 / 0.0366 | = e2 | = e3 | −0.0011 / −0.0018 / −0.0096 |
| 31 | q / k / v | 0.00307 / 0.00844 / 0.0052 | 0.00393 / 0.01082 / 0.0064 | = e2 | = e3 | −0.0009 / −0.0024 / −0.0013 |

(Medians over the 141 positions. The p90s and the positions 7 and 114–122 order the same way, every one of them.)
- **Kernel arithmetic:** both kernels equal their float64 twins: ‖(4) − (2)‖/‖(2)‖ and ‖(5) − (3)‖/‖(3)‖ print 0.00000 at every layer,
  segment and position.
- **The registered branch:** e4 ≤ e5 everywhere. On the same input, the native Q5_K qkv projection is closer to the truth than the
  int8 lane's, by about a quarter. Both carry the same per-32 activation-quantization error (e2, largest on v: 3–5% at layers 2–8),
  and the int8 lane adds its weight rounding on top. No per-projection factor explains the gap. By the registration, that is the
  finding, and the next step is Francis's question, not a fourth run.
- **Not run:** D2b (ffn_down, Q6_K). It was registered to run only if D2a did not stop.

**KS-O1-F (c), registered 2026-10-25 before any run (Francis, 18:57: option c).** The question: does "native worse" come from
repetitive input in general, or only from 110 × "the "?
- **Instrument:** D1's logits-level read (TestKSO1C_fillerDiagnostic's statistics) for the int8 and native arms against the CPU at f32,
  over all positions of each prompt. Phi-3 mini q4, one CUDA q4k resident holding both arms, the f32 cache. Harness:
  `docs/measurements/ks-o1-f-2026-10-25/general_test.go.txt`.
- **Five prompts, each under Phi-3's chat template:**
  - **The control:** 110 × "the " followed by D1's instruction. It must reproduce D1 (native p10 0.8826, int8 0.9401, each within
    0.001), or the run is void.
  - **Three other fillers,** 110 repeats each with the same instruction: "and " (another function word), "apple " (a content word)
    and "hello ".
  - **Natural text:** `README.md`'s first 110 tokens inside "Summarize this document:", about the fillers' length, so length is
    not the difference (LQ3's 900-token documents were six times longer).
- **Reading per prompt:**
  - **native worse:** native p10 below int8's by more than 0.01 and native mean KL above 1.10× int8's.
  - **native not worse:** native p10 at least int8's − 0.01 and native KL at most 1.10× int8's.
  - **mixed:** anything else.
- **Reading over the prompts:**
  - **Repetitive input in general:** at least 2 of the 3 other fillers read native worse, and the natural text reads native not
    worse.
  - **Specific to "the ":** at least 2 of the 3 other fillers read native not worse.
  - **Ambiguous, parked:** anything else, and the natural text reading native worse.
- **Cost:** one process, about 1.5 minutes a prompt: about 10 minutes, as one job on nobara's night queue at priority 55, ahead of
  ks-o4-gfid. Not timed.

**KS-O1-F (c) read (19:50–20:01, nobara's night queue, the exclusive lock).** Log: `docs/measurements/ks-o1-f-2026-10-25/c.log`.

| prompt | positions | int8 p10 / min / KL | native p10 / min / KL | native ÷ int8 KL | reading |
|---|---|---|---|---|---|
| 110 × "the " (control) | 141 | 0.9401 / 0.4543 / 0.42710 | 0.8826 / 0.0294 / 0.50969 | 1.193 | native worse |
| 110 × "and " | 141 | 0.9887 / 0.8084 / 0.08847 | 0.9882 / 0.8861 / 0.06437 | 0.728 | native not worse |
| 110 × "apple " | 141 | 0.9826 / 0.6715 / 0.08729 | 0.9925 / 0.6699 / 0.02393 | 0.274 | native not worse |
| 110 × "hello " | 141 | 0.9817 / 0.8059 / 0.10998 | 0.9916 / 0.8021 / 0.05983 | 0.544 | native not worse |
| `README.md`, first 110 tokens | 122 | 0.9958 / 0.9567 / 0.01024 | 0.9979 / 0.9921 / 0.00634 | 0.619 | native not worse |

- **The control reproduces D1** to the digit.
- **By the registered rule, the result is specific to "the ":** all three other fillers and the natural text read native not worse,
  with the native arm's KL to f32 at 0.27–0.73× the int8 arm's.
- **"the " is unusual as a filler:** both arms are far further from f32 on it than on any other prompt (int8 KL 0.43 against 0.09–0.11
  on the other fillers).
- **Together with D2a:** on the same input, the native projections are the more accurate ones. On every other prompt measured, the
  native lane is the closer of the two end to end.
- **Closed (Francis, 2026-10-25, 20:31):** KS-O1-F closes on this finding. The native lane's gap is specific to 110 × "the ", and
  no fix is indicated.

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

**KS-C2 first measurement, registered 2026-10-23 before the code, on the owner's order of 11:50.** The prototype,
`gemv_w4a8_resid_mk<R>` in `metal/gemv_r18_test.go`'s `r18Kernels`: `gemv_w4a8_resid_staged<R>` (production's down) with each
word's integer sum taken as R18b does, through two masked half-word dots against the word's eight activations pre-scaled by
16^-(k mod 4), less 8·Σa. The activations stay staged as int8 (K bytes, so the 7B's 18,944 fits) and are scaled in registers.
Every partial sum is an integer below 2^24, so `float(gi)` is the shipped integer exactly and the per-row chain
`acc += float(gi) · scale` is unchanged. Instruments, in order:
1. `TestR18Step0Standalone`'s down step at the 1.5B's (1,536 × 8,960) and the 7B's (3,584 × 18,944) shapes: `resid_mk2` and
   `resid_mk4` against `gemv_w4a8_resid`, byte-equal, timed beside `resid_st4`.
2. `TestR18InSequence` on the 1.5B (the Metal `.gguf` at int4), 7 paired reps, a new arm `h4244m` (production with the masked down)
   against production `h4244`: its bit-identity precondition and the in-sequence down-category work.

**Kill** on any differing byte in either instrument, or if the in-sequence down work of `h4244m` is under 1.03× production's
(median over the 7 reps); otherwise the wiring follows, behind a whole-token in-process A/B. Timed, Mac, the timing lock held
exclusively. Stopping rule: one pass each. Cost: about 2 and 10 minutes.

**KS-C2 first measurement read 2026-10-23 (12:53–12:54, the timing lock held exclusively, load 2.23 then 2.04, 5.2 GB available).**
Standalone, byte-equal at both shapes: the 1.5B's down `resid_mk4` 0.062 ms against production's `resid_st4` 0.089 (1.44×), the 7B's
0.237 against 0.344 (1.45×). In sequence on the 1.5B (7 reps): `h4244m` 0 of 151,936 logits differ over 16 positions; its down work
1.741 ms against production's 2.571 (**1.48×**), all four GEMVs 1.133× production's (bar ≥ 1.03), the full token −0.77 ms (10.55 →
9.78). Not killed: the wiring follows. Log: `~/goinfer-logs/ksc2-2026-10-23.log`.

**KS-C2 wiring gates, registered 2026-10-23 before the code (C2b-G).** The build replaces the bodies of production's
`gemv_w4a8_resid_staged<R>` and `gemv_w4a8_coal_staged<R>` with the masked form; `KSC2_OLD_DOWN` (prepended to the decode library
by `ksc2OldDownForTest`, read at build) compiles the per-word integer form back for the A/B, as `KSD7_OLD_DEQUANT` does.

| gate | tier | instrument | PASS iff |
|---|---|---|---|
| C2b-G1 identity | 0 | `gemv_w4a8_resid_staged<2,4>` against `gemv_w4a8_resid` and `gemv_w4a8_coal_staged<2,4>` against `gemv_w4a8_coal` at the 1.5B's and the 7B's down shapes, random int4 weights, f16 scales and int8 activations, fast-math and precise libraries; the 1.5B's Metal `.giw` and qwen35-tiny (a DeltaNet layer, `coal_staged`), 32 greedy decode steps, every step's logits, the new library against `KSC2_OLD_DOWN` | byte-equal everywhere |
| C2b-G2 speed | timed, Mac, timing lock | a whole-token in-process A/B on the 1.5B's Metal `.giw`: two residents in one process (old and new library), 128 greedy decode tokens after a 32-token prompt, A B B A over 10 pairs after 2 warm runs each | SHIP iff the median new/old per-token time is ≤ 0.98 and at least 9 of 10 pairs are faster; otherwise PARK with the reading |

Planted defect: the high half-word dotted with the low half's activations (`x0` for `x1`); C2b-G1 goes red. Stopping rule: one pass
each. Cost: G1 seconds; G2 about 4 minutes.

**KS-C2 wiring read 2026-10-23.** C2b-G1 **PASS**: 16 kernel cells (both staged kernels, R = 2 and 4, both shapes, both libraries)
0 differ; qwen35-tiny 0 of 8,192 and the 1.5B 0 of 4,861,952 logits differ over 32 decode steps. The registered plant (`x0` for `x1`)
turned every masked cell red; a second plant inside the `KSC2_OLD_DOWN` branch only turned the model identity red, so the define
reaches the old arm (KS-D7's lesson). C2b-G2 (12:57–12:58, the lock held exclusively, load 1.37 at the start, 5.4 GB available;
each pair's load 1.14–1.50): old 10.70–10.86 ms a token, masked 9.97–10.13, **median 0.9332, 10 of 10: SHIP**. Logs:
`~/goinfer-logs/ksc2b-g1-2026-10-23.log`, `~/goinfer-logs/ksc2b-g2-2026-10-23.log`. Also: `touched.py` green after merging main; the
decode families by name (`R18|MC3|DeltaNet|Qwen35|Decode|Step|Gemv|GEMV|Coal|Staged|KSC2`) 53 pass, 31 skip, 0 fail; staticcheck
clean.

**KS-C3 · Native int8 decode GEMVs get R18's rows and staging (bit-identical).**
- **Waste:** `W8A8_SA_BODY` / `W8A8_COAL_BODY` run one row per simdgroup with 4-byte loads, and the down projection re-reads
  activations from device memory in every lane. The 1.5B at int8int8 decodes at 70.6 tok/s, with its int8 body near
  132 GB/s (inferred).
- **Change:** R=2/4 rows per simdgroup, `uint4` loads, and activations staged per threadgroup. int32 sums are exact.
- **Gain (est.):** −1.7 to −2.0 ms (12–14%) if the body reaches 160–165 GB/s. This is `-backend auto`'s dense int8int8 path,
  including the goinfer-chat bundles.
- **Size:** S–M.
- **Prior art:** `docs/tasks/task-metal-int8-2026-10.md` slice 1, "can follow", never built.

**KS-C3 first measurement, registered 2026-10-23 before the run, on the owner's order of 11:50.** Instrument: `TestMetalGEMVS0`
gains the native int8 body at the 1.5B's shapes, each the kernel decode dispatches there: qkv (`gemv_w8a8_sa`, 2,048 × 1,536), o
(`gemv_w8a8_sa`, 1,536 × 1,536), gate|up (`gemv_w8a8_sa`, 17,920 × 1,536) and down (`gemv_w8a8_body_resid`, 1,536 × 8,960), timed
as its other rows are (SLC-defeating rotation, GPU timestamps, median of 7), beside the run's streaming-read ceiling. Timed, Mac,
the timing lock held exclusively. The reading is the body's bandwidth: the four categories' bytes over their summed time.
**Build iff** that is under 145 GB/s **and** under 80% of the ceiling read in the same run; otherwise the headroom to the
estimate's 160–165 GB/s is under 10% and KS-C3 closes as measured. Stopping rule: one pass. Cost: about 2 minutes.

**Read 2026-10-23 (12:37), the timing lock held exclusively (1-minute load 1.43, 5.3 GB available):** the streaming ceiling
179.7 GB/s; the int8 body at the 1.5B's shapes qkv 0.028 ms (113 GB/s), o 0.025 ms (93), gate|up 0.187 ms (148), down 0.108 ms
(127): **47.0 MB in 0.348 ms, 135 GB/s, 75.1% of the ceiling**. Under both lines: **build**. At 160–165 GB/s the four would take
0.285–0.294 ms, −0.054 to −0.063 ms a layer, −1.5 to −1.8 ms a token over 28 layers. Log: `~/goinfer-logs/ksc3-gemv-2026-10-23.log`.

**KS-C3 gates, registered 2026-10-23 before the code (C3-G).** The build: int8 twins of R18's rows kernels, each taking its int4
twin's arguments in the same order at the same launch shape, so a native int8 resident keeps `gemvRows` (R18's widths: 4 for qkv,
gate|up and down, 2 for o) and swaps the handles, as the one-row int8 twins already do. `gemv_w8a8_sa_rows<R>` / `_bias_rows<R>` /
`_resid_rows<R>` stage the activation as `uint4` words per threadgroup, and each lane reads `uint4` words of R rows;
`gemv_w8a8_body_resid_staged<R>` does the same for down. The int32 sums are exact and the epilogue is the shipped
`simd_broadcast_first(float(acc) · aScale · wScale)`, so the outputs are the shipped kernels' bits. Taken only where every K is a
multiple of 16 (the `uint4` rows) and the row counts tile (`gemvRowsFor`). The batched step's int4 adjacency kernels
(`mc3_gemv_w4a8_*_rows_adj`) never run for a native int8 resident. `w8RowsOffForTest` keeps the one-row kernels for the A/B.

| gate | tier | instrument | PASS iff |
|---|---|---|---|
| C3-G1 identity | 0 | each twin against its shipped one-row kernel at the 1.5B's and the 0.5B's four shapes, R = 2 and 4, random int8 weights and activations, f32 scales, in the fast-math and the precise library; llama-tiny and the 0.5B at int8int8, 32 greedy decode steps, every step's logits, rows on against off | byte-equal everywhere |
| C3-G2 kernel | timed, Mac | `TestMetalGEMVS0`'s int8 rows at the 1.5B's shapes, the twins beside the shipped kernels | read, not gated: the body's GB/s, for the record and the direction |
| C3-G3 speed | timed, Mac, timing lock | a whole-token in-process A/B: the 0.5B at int8int8 (the 1.5B at int8 is fit-refused beside the owner's apps), 128 greedy decode tokens after a 32-token prompt, rows on against off on one resident, A B B A over 10 pairs after 2 warm runs each | SHIP iff the median rows/one-row per-token time is ≤ 0.98 and at least 9 of 10 pairs are faster; otherwise PARK with the reading |

Planted defect: the twin's epilogue drops `simd_broadcast_first` (the shipped kernel's guard against fast-math fusing the residual
add); if C3-G1 stays green the guard is not needed here and the plant becomes a scale off by one row (`sct[row0 + r + 1]`), which
must go red. Stopping rule: one pass each. Cost: G1 seconds; G2 about 1 minute; G3 about 3 minutes.

**KS-C3 read 2026-10-23 (12:41–12:49).** C3-G1 **PASS**: 32 kernel cells (both models' four shapes, R = 2 and 4, fast-math and
precise) 0 differ; 32 decode steps at int8int8, llama-tiny 0 of 8,192 and the 0.5B 0 of 4,861,952 logits differ. The first plant
(no `simd_broadcast_first`) turned 24 of 32 cells red, so the guard stays. C3-G2 (12:48, lock exclusive, load 1.39): the twins at the
1.5B's shapes qkv 0.024 ms (129 GB/s, from 116), o 0.022 (106, from 109), gate|up 0.156 (176, from 148), down 0.097 (142, from 127):
the body **157 GB/s, from 135**, the ceiling 179.0. C3-G3 (12:48–12:49, lock exclusive, load 1.36 falling to 1.03): the 0.5B at
int8int8, **median 1.0021, 3 of 10 pairs faster: PARK** by the bar. Pairs 0–3 ran 5.8–8.4 ms a token as the GPU's clock settled;
pairs 4–9 read 0.998–1.004 at 5.8 ms. Logs: `~/goinfer-logs/ksc3-g1-2026-10-23.log`, `~/goinfer-logs/ksc3-g2g3-2026-10-23.log`.

**Amended 2026-10-23 (12:50), before any 1.5B decode reading: C3-G3b.** The 0.5B was the registered cell only because the 1.5B at
int8 is fit-refused by day. Its int8 body is about 358 MB a token, 2.6 ms of a 5.8 ms token, so a 15% faster body is at most about
0.4 ms there, and the reading did not resolve it. The survey's estimate is for the 1.5B (the body about 1.3 GB a token). C3-G3b is
C3-G3's instrument on the 1.5B's `.gguf` at int8int8, at night as case (a), through
`docs/measurements/ks-night-2026-10/run-ksc3-g3b.sh` (a test binary built at the registering commit; exit 75 on a fit refusal).
Same bar: SHIP iff the median rows/one-row is ≤ 0.98 with at least 9 of 10 pairs faster; otherwise KS-C3 closes as PARK with both
readings. Stopping rule: one pass. Cost: about 5 minutes.

**C3-G3b read 2026-10-23 (22:19, the night queue, case (a)).** The first attempt (22:01) was refused by the fit guard (2.9 GB
available against 2.7 GB needed at the 70% budget) and exited 75; the retry at 22:18 loaded. The night runner held the timing lock
exclusively; the 1-minute load was 1.81 at the start, each pair's 1.44–2.10. The 1.5B's `.gguf` at int8int8 on Metal, rows
`{qkv:4 o:2 gu:4 down:4}`: one-row 13.821–13.836 ms a token, rows 12.525–12.610, **median rows/one-row 0.9109, 10 of 10 pairs
below 1: SHIP** (8.9% faster, about 1.23 ms a token; the survey's estimate was 1.7–2.0 ms). The two "Gated-DeltaNet model runs
resident at int4 only" lines in the log come before the timed load and appear in the day run's C3-G3 log too; they are not the
1.5B's load. Log: `~/goinfer-logs/ksc3-g3b-2026-10-23-2218.log`.

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

**KS-C4 and KS-C5 first measurements, registered 2026-10-24 before the run (on the owner's order of the Mac work, 08:31).** One
instrument answers both: `TestMetalDecodeDecomp` (tier 1, timed, the timing lock held exclusively), Qwen2.5-Coder-1.5B's Metal int4
resident, at depths 128 and 512 (`GOINFER_METAL_DDECOMP_DEPTHS=128,512`), with its "rope, kv store, act-quant, ctx-quant" category
split into one category per pipeline (`rope2`, `kv store`, `swiglu_quant`, `quant_vec`); a test-only change. The reading is each
category's in-sequence work cost (full − the category no-op'd), medians as the test takes them. A cell started under the void rule's
load or memory line is re-run.
- **KS-C4:** kill if `swiglu_quant` + `quant_vec` total under 0.25 ms a token at depth 128; otherwise its build's gates are
  registered next.
- **KS-C5:** kill if the attention category at depth 128 is under 0.30 ms a token (the estimated 0.1–0.2 ms could not be half of
  it); otherwise its build's gates are registered next. The 512-key reading is for the record.

Stopping rule: one pass. Cost: about 6 minutes.

**Read 2026-10-24 (09:24, the timing lock held exclusively; the 1-minute load 2.58 at the start, 11.5 GB available).** The 1.5B's
`.gguf` on its Metal int4 resident (`attention_fa` engages from 1,024 keys, so both depths ran the legacy attention); medians of 5 reps
× 20 tokens, ms a token:

| category | depth 128 | depth 512 |
|---|---|---|
| attention | **0.643** | 1.677 |
| GEMV qkv / o / gate\|up / down | 0.570 / 0.407 / 3.223 / 1.864 | 0.630 / 0.659 / 3.229 / 1.823 |
| LM head (int8) | 1.418 | 1.308 |
| norm+quant (rms) | 0.599 | 0.459 |
| rope (rope2) / kv store | 0.153 / 0.187 | 0.038 / 0.150 |
| act-quant (`swiglu_quant`) | **0.644** | 0.455 |
| ctx-quant (`quant_vec`) | **0.231** | 0.129 |
| dispatch floor | 0.650 | 0.611 |
| full token | 9.708 (103.0 tok/s) | 10.461 (95.6 tok/s) |

- **KS-C4:** `swiglu_quant` + `quant_vec` **0.875 ms at depth 128**, over the 0.25 ms line: **build**; its gates are registered next.
- **KS-C5:** attention **0.643 ms at depth 128**, over the 0.30 ms line: **build**, after KS-C4.

Neither quantizer's work depends on depth, yet `swiglu_quant` read 0.644 and 0.455: the method's spread on a small category is about
±0.2 ms (the paired columns agree), so the 0.875 ms sum is good to roughly a quarter. Both clear their lines at either depth. The run
took 29 s, not the 6 minutes registered. Log: `~/goinfer-logs/ksc4c5-decomp-2026-10-24.log`.

**KS-C4 gates, registered 2026-10-24 before the code (C4-G).** The build folds `swiglu_quant` into its neighbours on the dense int4
decode path (the R18 gate|up rows kernel and the staged down kernel; not the f16 lane, LoRA, sandwich, post-only or parallel-block
layers, the native int8 lane, MoE, the batched step or prefill, which keep the two-step form):
- `gemv_w4a8_sa_glu_rows<R>`: `sa_rows_acc`'s body with simdgroup s taking gate rows p0 … p0 + R/2 − 1 and the matching up rows
  I + p0 …, p0 = s·R/2. Lane 0 forms each g and u as `gemv_w4a8_sa_rows` writes them (`acc·asc`), then h = `glu_act_pinned(g)·u`,
  `swiglu_quant`'s statement. It writes h (into the gate|up buffer's first I floats) and one max |h| partial per threadgroup.
- `gemv_w4a8_resid_staged_hq<R>`: each threadgroup takes the max of the partials (max is exact in any order), forms sc and 1/sc as
  `swiglu_quant` does (sc 0 → 1), stages each h quantized with `swiglu_quant`'s rounding, then runs `gemv_w4a8_resid_staged`'s body
  with sc as the activation scale.
- The partials live in `r.dq`'s memory, which the fold path no longer writes. `ksc4FoldOffForTest` keeps the two-step form for the
  A/B. The ctx quantizer (`quant_vec`) is a second step (C4b), registered only if this one ships.

| gate | tier | instrument | PASS iff |
|---|---|---|---|
| C4-G1 identity | 0 | (a) the folded pair against `gemv_w4a8_sa_rows<R>` → `swiglu_quant` → `gemv_w4a8_resid_staged<R>` at the 0.5B's, 1.5B's and 7B's shapes, R = 2 and 4, SiLU and GELU-tanh, random int4 weights, f16 scales and int8 activations, the fast-math and the precise library; (b) llama-tiny and the 0.5B and 1.5B Metal `.giw`s at int4, 32 greedy decode steps, every step's logits, fold on against off, and the fold taken (a dispatch counter); (c) `TestMetalSnapshotGolden` | (a) the down outputs byte-equal in every cell; (b) byte-equal and taken; (c) PASS |
| C4-G2 speed | timed, Mac, timing lock | a whole-token in-process A/B on the 1.5B's Metal `.giw`: fold on against off on one resident, 128 greedy decode tokens after a 32-token prompt, A B B A over 10 pairs after 2 warm runs each | SHIP iff the median on/off per-token time is ≤ 0.98 with at least 9 of 10 pairs faster; under it, the owner's rule for a bit-identical change (CLAUDE.md, "Measurement discipline") decides; otherwise PARK |
| C4-G3 memory | 0 | Metal's allocated size after the 1.5B's resident is built, fold on against off | not higher |

Planted defect, run once on the committed tree and reverted: the up row taken one row off (I + p0 + j + 1); C4-G1 goes red.
Stopping rule: one pass each. Cost: G1 about a minute; G2 about 4 minutes; G3 seconds. Gain (est.): 0.3–0.5 ms a token on the 1.5B
(`swiglu_quant` reads 0.455–0.644 ms; the fold adds a redundant quantize of h in each down threadgroup).

**Amended 2026-10-24, before any C4-G run.** (1) There is no dispatch counter, so C4-G1 (b)'s "taken" is shown by a third arm: the
folded down kernel swapped for a no-op, whose logits must differ from the fold's. (2) The down kernel takes the gate|up R as a
template parameter (`gemv_w4a8_resid_staged_hq<Rd, Rg>`) rather than a constant buffer, so the fold allocates nothing (C4-G3).
The instruments are `TestKSC4_foldMatchesThreeStep`, `TestKSC4_decodeIdentity`, `TestKSC4_allocatedNotHigher` and
`TestKSC4_decodeSpeed`; the bars are unchanged.

**KS-C4 read 2026-10-24 (09:30–09:38): PARK.** The fold's code is kept on branch `ksc4-fold-parked-2026-10-24`, not merged.
- **C4-G1 (a) PASS**: 24 kernel cells (three shapes, SiLU and GELU-tanh, R = 2 and 4, both libraries) 0 differ, and the same cells
  through the resident's own compiled pipelines 0 differ. **C4-G1 (b) red**: llama-tiny 0 of 8,192; the 0.5B 3,494,525 and the 1.5B
  2,582,910 of 4,861,952 logits differ over 32 steps, fold on against off; the no-op arm shows the fold taken on every model.
  Localized on the 0.5B: steps 0–8 byte-equal, then layer 17 at position 9 differs in one of 4,864 quantized activations. There
  h·inv is exactly −10.5: the fold rounds it to −11 (Metal's `round`, half away from zero, measured scalar and `float4`, fast and
  precise), and `swiglu_quant` gives −10. Its second pass recomputes glu(g)·u and multiplies by inv in one expression, which fast
  math re-associates, so a value on a rounding tie lands on the other side. The partial max and every other value matched. Passing
  g and u through `simd_broadcast_first` did not change it. A fold cannot reproduce that without matching the compiler's choice in
  another kernel. **C4-G3 PASS**: allocated 1,001,684,992 bytes either way.
- **C4-G2, read for the record after G1 went red** (09:37, the lock held exclusively, load 2.08, 12.8 GB available): off 10.01–10.30 ms
  a token, on 10.21–11.46, **median on/off 1.0489, 1 of 10 pairs faster: 4.9% slower**. Every down threadgroup re-quantizing all of h
  (35 KB read per threadgroup, 48 threadgroups a layer) and the gate|up kernel reading rows I apart cost more than the dispatch saved.
- So the fold fails both as a bit-identical change and on speed, and re-registering it as fidelity-gated would not help. What the
  0.875 ms buys back, if anything, is a multi-threadgroup quantizer (partials, then a parallel quantize), the shape of KS-E2's
  widening on WebGPU; not built. Logs: `~/goinfer-logs/ksc4-g1a-2026-10-24.log`, `~/goinfer-logs/ksc4-g1b-g3-2026-10-24.log`,
  `~/goinfer-logs/ksc4-g2-2026-10-24.log`.

**KS-C5 · Legacy decode attention below 1024 keys (bit-identical).**
- **Waste:** at 128 keys it costs 0.67 / 0.78 ms per token, with serial 8-wide V-load batches (16 dependent round trips) and
  about 16 barriers.
- **Change:** 16/32-wide V batches in the same order, and `simd_shuffle_down` for tree levels ≤16, keeping
  `tgReduceAttn` at 128.
- **Gain (est.):** −0.1 to −0.2 ms (1–2%); low confidence.
- **Size:** S.

**KS-C5 gates, registered 2026-10-24 before the code (C5-G).** The build changes `attention`'s single-tile path (nWin ≤ 4096; the
tiled path is untouched): the V read issues 16 loads ahead instead of 8, the adds retired in the same sequential order (a scalar
tail for the rest); the max and sum reductions run their levels at stride 32 and above through threadgroup memory as today, and
the levels at stride 16 to 1 in simdgroup 0 with `simd_shuffle_down`, each level combining the same two operands in the same order,
so 5 barriers fewer per reduction. `attention_f32`, `mc3_attention_rows` and the int8/f32 rows forms derive from this text and take
the same change. `KSC5_OLD_ATTN` (a test-only define, `ksc5OldAttnForTest`, read at build) compiles the shipped form back.

| gate | tier | instrument | PASS iff |
|---|---|---|---|
| C5-G1 identity | 0 | (a) the new `attention` against the `KSC5_OLD_ATTN` library: head dims 64 and 128, nKeys 1, 7, 8, 15, 16, 17, 31, 100, 128, 129, 512, 1000 and 4096, window 0 and 64, a sink on and off, the fast-math and the precise library; (b) llama-tiny and the 0.5B and 1.5B Metal `.giw`s at int4, 32 greedy decode steps after a 300-token prompt (keys to 332), every step's logits, a resident built with each library | byte-equal everywhere |
| C5-G2 speed | timed, Mac, timing lock | a whole-token in-process A/B on the 1.5B's Metal `.giw`, two residents in one process (old and new library), 128 greedy decode tokens after a 384-token prompt (keys 385–512, where the decomposition put attention at 16% of the token), A B B A over 10 pairs after 2 warm runs each | SHIP iff the median new/old per-token time is ≤ 0.98 with at least 9 of 10 pairs faster; under it, the owner's rule for a bit-identical change decides; otherwise PARK |

C5-G3: none; the change allocates nothing (threadgroup memory only). Planted defect, run once and reverted: the sum's shuffle level
adds the lane's own value twice (`v + v`) at stride 1; C5-G1 goes red. A second plant inside the `KSC5_OLD_ATTN` branch must turn
(b) red, so the define reaches the old arm. Stopping rule: one pass each. Cost: G1 about a minute; G2 about 5 minutes.

**KS-C5 read 2026-10-24 (09:44–09:53): PARK.** The code is kept on branch `ksc5-attn-parked-2026-10-24`, not merged.
- **C5-G1 PASS.** (a) 208 cells (both libraries, head dims 64 and 128, 13 key counts, windows 0 and 64, a sink on and off), 0 outputs
  differ; the registered plant turned every cell red. (b) The 0.5B and the 1.5B after a 300-token prompt, 0 of 4,861,952 logits
  differ over 32 steps; llama-tiny 0 of 8,192 after a 95-token prompt (its context is 128). A plant in the `KSC5_OLD_ATTN` branch
  turned llama-tiny's 8,192 logits red, so the define reaches the old arm.
- **C5-G2** (09:51–09:53, the lock held exclusively; the 1-minute load 2.46 at the start, each pair's 2.53–3.23; 12.9 GB available):
  old 10.88–13.05 ms a token, new 10.81–13.42; **median new/old 1.0108, 4 of 10 pairs faster: PARK.** Both arms moved between about
  10.8 and 12.9 ms within the run, a spread far larger than the 1–2% the change could give (attention is 16% of a 512-key token, so
  0.2 ms is its ceiling). The reading does not resolve a direction; it is recorded as unresolved, not as a loss.
Logs: `~/goinfer-logs/ksc5-g1a-2026-10-24.log`, `~/goinfer-logs/ksc5-plant-2026-10-24.log`, `~/goinfer-logs/ksc5-g1b-2026-10-24.log`,
`~/goinfer-logs/ksc5-oldplant-2026-10-24.log`, `~/goinfer-logs/ksc5-g2-2026-10-24.log`.

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

**KS-D1 first measurement, registered 2026-10-23 before the run, on the owner's order of 11:50.** Instrument:
`TestKSD1_attnShare`, Gemma 4 E2B (its default Metal sidecar, `decoder.SidecarPath`), `PrefillLast` on 4,096 resident rows through
the batched E-model pass (`emodelBatched`, the fast-prefill floor set to 0), the full pass against the same pass with the prefill
attention dispatch skipped (`prefillAttnSkipForTest`, a new test seam; the output is then meaningless and is not read), A B B A
over 3 pairs after one warm run each. The reading is attention's share, 1 − skip/full, the median over the pairs. Timed, Mac, the
timing lock held exclusively, a cell started under the void rule's load or memory line re-run. **Kill under 50%**; otherwise the
build's gates are registered next. Stopping rule: one pass. Cost: about 4 minutes.

**Read 2026-10-23 (13:28–13:30, the timing lock held exclusively, the 1-minute load 1.33 at the start, 5.8 GB available; each
pair's 1.03–2.41).** E2B's default sidecar (`…int4.ple4.metal.v17.giw`), 4,096 rows in one pass (the exact kernel's widest head dim
512, so `PrefillBlockRows` 0): full 28.17–28.35 s, without attention 5.29 s in every pair; **attention 81.2% (median of 3), over
the 50% line**. The build follows. Log: `~/goinfer-logs/ksd1-share-2026-10-23.log`.

**KS-D1 gates, registered 2026-10-23 (13:32) before the code (D1-G).** The build: `attention_prefill_steel_split<HD>` for head dims
256 and 512, R19's kernel with the head dim split over the threadgroup's 8 simdgroups: NS = HD/128 slices of 128, 8/NS row groups
of 8 rows (32 query rows a threadgroup at 256, 16 at 512), K/V blocks of 16 keys at 256 and 8 at 512 staged once in padded
threadgroup memory (17 KB). Each simdgroup forms its slice's partial Q·Kᵀ, writes it to threadgroup memory, and sums its row
group's NS partials in slice order, so every slice of a row group holds the same scores and the same online-softmax state; it then
accumulates its own 128 output columns. A layer whose head dim is 256 or 512 takes it where the pass would take the exact kernel
(not an image block, not a softcap); `prefillSteelAttnOff` keeps the exact kernel for the A/B. Not bit-identical (half P, exp2, the
order): fidelity-gated.

| gate | tier | instrument | PASS iff |
|---|---|---|---|
| D1-G1 kernel | 0 | the split kernel against an f64 attention at head dims 256 and 512: random q, k, v; M = 100 and 300; start 0 and 37; window 0 and 64 (at 256); 8 query heads over 1 KV head | every output finite; max-abs error ≤ 4e-3 and mean-abs ≤ 2e-4 in every cell (about 4× D2-G2's steel readings at head dim 64) |
| D1-G2 fidelity | 2, Mac, by day | (a) G-S9d (`TestS9EModelBatched_realE2B`) with the split kernel in the batched pass, its own rule against the layer-major reference (PASS: agreement ≥ the reference's − 2.0 points and free-run passes ≥ the reference's − 1); (b) a long cell: E2B, a 2,048-token prompt (this repository's text), the CPU int4 reference over the last prompt row and 32 teacher-forced positions, argmax agreement of the split kernel's batched pass against the exact kernel's. Tier 2 because only a reference can say which arm is closer; budget about 30 min | (a) PASS by its rule, and (b) agreement ≥ the exact kernel's − 2.0 points |
| D1-G3 speed | timed, Mac, timing lock | E2B, `PrefillLast` on 4,096 rows through the batched pass, split kernel against exact (`prefillSteelAttnOff`), A B B A over 3 pairs after one warm run each | SHIP iff the median new/old is ≤ 0.80 (≥ 1.25×); otherwise PARK |

Planted defect: the row group sums only its own slice's partial (no exchange); D1-G1 goes red. Stopping rule: one pass each; a red G1
stops the build until understood. Cost: G1 seconds; G2 about 30 min; G3 about 4 min.

**KS-D1 read 2026-10-23.** D1-G1 **PASS** (13:37): 12 cells, max-abs 0.38–1.21e-3, mean-abs 2.6–4.8e-5, all finite; not a gate, but
recorded: every cell's 300 rows run as two passes of 150 equal the single pass byte for byte. The plant (each simdgroup's own slice
only) read max-abs 2.0, red. D1-G3 (13:42–13:44, the lock held exclusively, the 1-minute load 3.59 at the start, 8.3 GB available):
exact 28.24–28.30 s, split 6.13 s in every pair, **median 0.2170 (4.61×): SHIP**. D1-G2 (13:44–13:47, shared): (a) G-S9d, the batched
pass with the split kernel 95.31% (244/256) and 8/8 free-run passes against layer-major's 96.48% (247/256) and 7/8, delta −1.17
points: **PASS**; (b) the long cell, `CLAUDE.md` (sha256 `374fae4f…`) cut to 2,048 tokens, exact 31/32 and split 31/32: **PASS**. Logs:
`~/goinfer-logs/ksd1-g1-2026-10-23.log`, `~/goinfer-logs/ksd1-gates-2026-10-23.log`. Also: `touched.py` green; the prefill and E-model
families by name (`Prefill|MC5|DB01|R19|Steel|Chunk|Batched|Attention|S9|EModel|KH1|Gemma4|Gemma3|KSD1`) 60 pass, 71 skip, 0 fail.

Two consequences beyond the speed. With no exact-attention layer left, E2B and E4B no longer report `exactAttnHeadDim` 512, so
`PrefillBlockRows` now blocks their prompts like the plain pass (KS-H1's open head-dim-512 item): the two-pass byte check above is
what makes that safe. And every model with head dim 256 or 512 and no softcap takes the kernel, Gemma 3 included; only E2B was graded
at model level.

**KS-D2 · Steel prefill attention for head dims 64, 80 and 96.**
- **Where:** `metal/prefill.go:resident.prefillAttnKernels` takes steel only for head dim 128. Others run
  `attention_prefill_fused`, which R19 measured 7.2× slower.
- **Affected:** Qwen2.5-0.5B, Llama-3.2-1B and SmolLM2 (64), and Phi-3 at int4 (96).
- **Gain (est.):** attention is ~65–70% of the 0.5B's pass at 3900, so 3.4–3.8 s → ~1.6 s (~2.2×), and ~1.2× at 512.
- **Class:** fidelity-gated.
- **Size:** M.
- **First measurement:** `TestMetalPrefillDecomp` on the 0.5B. Kill it if attention is under 30% at 3900.

**KS-D2 first measurement, 2026-10-23 (11:59).** `TestMetalPrefillDecomp` on Qwen2.5-Coder-0.5B (`.gguf`, int4), 5 reps, the timing
lock held exclusively (1-minute load 3.46 at the start, 2.84 GB available): at K=3,900 attention is **2,504.8 of 3,603.3 ms GPU,
69.5%** (GEMM 26.9%); at K=512, 57.4 of 195.2 ms, 29.4%. The replica's logits equal `PrefillLast`'s bit for bit at both. Over the
30% kill line: the build follows. Log: `~/goinfer-logs/ksd2-decomp-2026-10-23.log`.

**KS-D2 gates, registered 2026-10-23 (12:02) before the code (D2-G), on the owner's order of 11:50.** The build:
`attention_prefill_steel` becomes a template on the head dim, instances 64, 80, 96 and 128, its K/V staging a loop over the block's
`16 × HD / 4` half4 words (at 128 the same four words a thread as today). `prefillAttnKernels` admits steel at head dim 64 as well as
128. The 80 and 96 instances are built and checked at tier 0 but not admitted: no model at those head dims has references on this
Mac, so each waits for its own model-level gate (Phi-3 at int4 for 96). `prefillSteelAttnOff` keeps the A/B's old arm.

| gate | tier | instrument | PASS iff |
|---|---|---|---|
| D2-G1 128 unchanged | 0 | the 1.5B (Metal `.giw`), `PrefillLast` at K=512 and 3,900, last logits, a test binary at this branch against one at its base | byte-equal at both |
| D2-G2 kernel | 0 | steel against an f64 CPU attention at head dims 64, 80 and 96: random q, k, v; M = 100 and 300; start 0 and 37; window 0 and 64; 14 query heads over 2 KV heads. The fused kernel against the same reference, the same cells | every output finite, and steel's max-abs error at most 2× the fused kernel's and its mean-abs error at most 1.25× the fused kernel's in every cell |
| D2-G3 fidelity | 2, Mac, by day | set A's §3.2 pooled gate (`TestPrefillGateVsReference`), a new cell Q05 (the 0.5B, f32 CPU references at K = 256, 512, 1,024 decision and 3,900 confirm, built first by `TestPrefillGateReference`), steel as production's prefill attention; the fused kernel's run (`GOINFER_METAL_R19_BASELINE=1`) beside it, reported. Tier 2 because tiers 0 and 1 cannot say which arm is closer to the reference; budget: references about 15 min, two gate runs about 25 min | Q05 **SHIPS** (critA, critB and critC all true) |
| D2-G4 speed | timed, Mac, timing lock | the 0.5B, `PrefillLast` at K=3,900 and at K=512, steel against fused on one resident, A B B A over 10 pairs after 2 warm runs each | SHIP iff the median steel/fused ratio is ≤ 0.80 at 3,900 (R19's ≥ 1.25× band) and ≤ 1.02 at 512; under 1.25× at 3,900, PARK with the reading |

Planted defect: the hd-64 staging takes the 128 instance's word index (`(idx & 31) × 4`); D2-G2 goes red. Stopping rule: one pass
each; a red G1 or G2 stops the build until it is understood. Cost: G1 and G2 seconds; G3 about 40 min; G4 about 3 min.

**KS-D2b gates, registered 2026-10-24 before the code (D2b-G), on the owner's order of the Mac work.** The build: `steelHeadDim`
admits head dim 96, and `steelFor` returns the hd-96 instance (`attention_prefill_steel_hd96`, built and checked at tier 0 in
D2-G2). The model at 96 is Phi-3 mini at int4 (`~/models/phi3-mini-4k-gguf/Phi-3-mini-4k-instruct-q4.gguf`). Its set-A references
are a new cell, PH3, in `TestPrefillGateReference` (f32 CPU, K = 256, 512, 1,024 decision and 3,900 confirm), built on nobara: an
f32 3.8B does not fit this Mac's CPU path. The cell's references are copied to the Mac's `~/goinfer-logs/prefill-ref`.

| gate | tier | instrument | PASS iff |
|---|---|---|---|
| D2b-G1 kernel | 0 | D2-G2's instrument at head dim 96 on this branch | as D2-G2 |
| D2b-G2 fidelity | 2, Mac | set A's §3.2 pooled gate (`TestPrefillGateVsReference`), cell PH3, steel as production's prefill attention; the fused kernel's run (`GOINFER_METAL_R19_BASELINE=1`) beside it, reported | PASS by the gate's own rule |
| D2b-G3 speed | timed, Mac, timing lock | Phi-3 at int4, `PrefillLast` at K = 3,900 and 512, steel against fused on one resident, A B B A over 10 pairs after 2 warm runs each | SHIP iff the median steel/fused is ≤ 0.80 at 3,900 and ≤ 1.02 at 512; under 1.25× at 3,900: PARK |

Admitted only on D2b-G2 PASS and D2b-G3 SHIP. Stopping rule: one pass each. Cost: G1 seconds; the references about an hour on
nobara; G2 about 40 minutes; G3 about 4 minutes.

**KS-D2b read 2026-10-24 (12:21): blocked, nothing admitted.** D2b-G1 PASS (the hd-96 cells: steel/fused max-error 0.92–1.01, mean
0.99). D2b-G3 could not run: Phi-3 mini at int4 is not Metal-resident. The resident declines it, because every resident projection
quantizes activations to int8 and Phi-3's activation outliers round to zero (queue-engineering H2); on Metal Phi-3 runs only the
q4k lane, which has no batched prefill pass. No other model at head dim 96 is on this Mac, so steel at 96 has no model to be graded
on or to serve, and the admission is reverted (the branch keeps this record only; the PH3 reference cell was not built). The KS-D2
entry's "Phi-3 at int4 (96)" as an affected model was wrong for the same reason. Reopens with a Metal-resident model at head dim 96.
Log: `~/goinfer-logs/ksd2b-g1-2026-10-24.log`, `~/goinfer-logs/ksd2b-g3-2026-10-24.log`.

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

**KS-D4 gates, registered 2026-10-23 before the code (D4-G), on the owner's order of 11:50.** Scope: the w32 lever only.
`eg2Ops.gemm` (the GEMMs of the EmbeddingGemma 2 text encoder and of the Gemma 4 vision and audio towers, their unfused attention's
score and value products included) moves from aikit's `GEMMF32Plan` (`gemm_f32_sg_big` where M and N are multiples of 32, else
`gemm_f32_sg`) to S17's `tower_gemm_w32`. w16 and the fused attention change outputs; each takes the tower gate as its own item.

| gate | tier | instrument | PASS iff |
|---|---|---|---|
| D4-G1 class | 0 | `tower_gemm_w32` against `GEMMF32Plan`'s kernel at (M, N, K) = (256, 768, 768), (1,024, 3,072, 768), (1,024, 768, 3,072), (100, 768, 768), (37, 1,536, 1,536) and (512, 512, 64), random f32, both in the precise library the towers compile | byte-equal in every cell. **A differing byte stops the w32 swap as a bit-identical change:** the reading is recorded and the lever goes to the tower gate, registered separately |
| D4-G2 identity | 0 | after the swap: each tower's committed tiny fixture (EmbeddingGemma 2, the Gemma 4 vision tower, the Gemma 4 audio tower), its output against `main`'s, through a test binary at each | byte-equal |
| D4-G3 speed | timed, Mac, timing lock | the Gemma 4 E4B vision tower (`~/models/gemma-4-E4B-qat-unq-towers`), one forward per image over the F2a images, the new GEMM against aikit's on one accelerator (a test seam), A B B A over 10 pairs after 2 warm runs each | SHIP iff the median new/old forward time is ≤ 0.98 with at least 9 of 10 pairs faster; otherwise PARK |

Stopping rule: one pass each; G1 red stops the item. Cost: G1 seconds; G3 about 5 minutes.

**KS-D4 read 2026-10-23.** D4-G1 **PASS**: `tower_gemm_w32` equals aikit's kernel byte for byte in all six cells, the `sg_big`
and the `sg` paths both. D4-G2 **PASS**: `mem2Towers`' seven tiny-fixture towers × three inputs, 21 output hashes equal at the base
(9bafcc12's merge-base) and the change; a planted `dbg = 3` in `eg2Ops.gemm` changed all 21, so the new path runs in every tower.
D4-G3 (13:18–13:20, the lock held exclusively, the 1-minute load 2.90 at the start falling to 1.10, 4.9 GB available): the E4B
vision tower over S3's four images at serve's cap, aikit 7,438–7,499 ms, `tower_gemm_w32` 4,581–4,673 ms, both arms' outputs equal
in every pair; **median 0.6192 (1.61×), 10 of 10: SHIP**. Logs: `~/goinfer-logs/ksd4-g1-2026-10-23.log`,
`~/goinfer-logs/ksd4-g3-2026-10-23.log`. Also: `touched.py` green after merging main; the tower families by name 12 pass, 17 skip,
0 fail; staticcheck clean. Left as their own items: w16 weights and the fused attention for these towers (both change outputs).

**KS-D5 · Fuse SwiGLU into the gate|up GEMM and drop `guF` (bit-identical).**
- **Waste:** `guF` exists only to carry gate and up to `swiglu_f16`. It is 52–55% of the 1.5B's and 7B's per-row prefill
  scratch.
- **Gain:**
  - A 3,900-token pass goes 255 → 116 MB on the 1.5B and 563 → 268 MB on the 7B.
  - 2.1× the rows per MEM7 block.
  - Speed neutral to +1%.
- **Size:** M.
- **Risk:** fast-math may compile the inlined activation differently; a byte compare decides.

**KS-D5 gates, registered 2026-10-23 (05:55) before the code (D5-G), on the owner's plan of 5:16 am.** The first measurement: one
3,900-row pass on the 1.5B counts 261.9 MB of scratch, `guF` 139.9 MB (53.4%). The build: `gemm_w4f16_swiglu`, the main 64 × 64
int4 prefill GEMM with its feature tile split into 32 gate rows and the matching 32 up rows. It stages the same weights with the
same dequant and MMA chain, rounds the tile to half in threadgroup memory as `guF` held it, and writes
`glu_act_f16(g) × u` for its 32 pairs. So the output is `swiglu_f16`'s, bit for bit, and `guF` is not written. A dense layer whose
pass takes the 64 × 64 tile uses it; the small-M tiles, the int8 and q4k lanes and the MoE expert GEMMs keep the two-step path,
and `guF` is allocated only for a pass that takes one of them. `swigluFusedOffForTest` keeps the two-step path for the A/B.

| gate | tier | instrument | PASS iff |
|---|---|---|---|
| D5-G1 identity | 0 | the fused GEMM against GEMM + `swiglu_f16` at random int4 weights, M = 64, 100 and 200, I = 96 and 8,960 (both activations); the Metal prefill of llama-tiny at int4 and of the 1.5B at 3,900 rows, last logits | byte-equal everywhere |
| D5-G2 scratch | 0 | `TestKSD5_guFShare`'s counter, one 3,900-row pass on the 1.5B | at most 123 MB (261.9 counted on main; `guF`'s 139.9 gone) |
| D5-G3 speed | timed, Mac, timing lock | the 1.5B (Metal `.giw`), a 3,900-token `PrefillLast`, fused against two-step, two residents in one process, A B B A over 10 pairs after 2 warm runs each | SHIP iff the median fused/two-step ratio is ≤ 1.01 (the survey's "speed neutral to +1%"; the gain is the scratch); otherwise PARK and decide |

Planted defect: the fused epilogue computes the activation on the f32 accumulators, before the half rounding `guF` imposed.
D5-G1 goes red. Stopping rule: one pass each. Cost: seconds; D5-G3 about 2 minutes.

**KS-D5b gates, registered 2026-10-23 (9:05 am) before the build, on the owner's "try it" (D5b-G).** D5-G3 parked the fused
kernel at 10.4% slower. The suspected cause: its added 8 KB threadgroup tile, which takes the kernel from about 9.6 KB to 17.6 KB of
threadgroup memory. The build: `gemm_w4f16_swiglu` places the half tile in the staging arrays' memory (one array holding `sa`, then
`sb`, 8.7 KB; the tile needs 8 KB), after a threadgroup barrier that ends the K loop's reads of them. Nothing else changes.

| gate | tier | instrument | PASS iff |
|---|---|---|---|
| D5b-G1 identity | 0 | D5-G1's instruments: the 12 kernel cells, llama-tiny, the 1.5B at 3,900 rows | byte-equal everywhere |
| D5b-G2 scratch | 0 | D5-G2's counter | at most 123 MB |
| D5b-G3 speed | timed, Mac, timing lock | D5-G3's instrument, the new build | SHIP iff the median fused/two-step ratio is ≤ 1.01; otherwise KS-D5 closes as parked, with both readings |

Stopping rule: one pass each, no further build on a miss. Cost: seconds; D5b-G3 about 2 minutes.

**KS-D5c gates, registered 2026-10-23 (10:12) before the code (D5c-G), from KS-D5's handoff ("a fused form of those is the next
idea").** Of the three lanes that keep `guF`, the small-M tiles hold at most 128 rows (under 5 MB on the 1.5B) and the MoE expert
GEMMs pick a tile per expert from its row count, so neither is a straight extension. The native int8 lane is: on by default for an
int8 dense model (`w8Eligible`), it runs the same 64 × 64 MMA chain (`gemm_w8f16_m64n64`), and its `guF` is the same 53% of a long
pass, since the scratch is half-precision whatever the weights. The build: `gemm_w4f16_swiglu` becomes a template on `W8`, and the
`W8` instance (`gemm_w8f16_swiglu`) stages its weights as `gemm_w4f16_tile<64, 64, true>` does (int8 codes, one f32 scale a row);
the epilogue is the shipped one. `swigluFused` admits a native int8 resident. `swigluFusedOffForTest` keeps the A/B's old arm.

| gate | tier | instrument | PASS iff |
|---|---|---|---|
| D5c-G1 identity | 0 | the W8 fused GEMM against `gemm_w8f16_m64n64` + `swiglu_f16` at random int8 weights and f32 row scales, D5-G1's 12 cells; the Metal prefill of llama-tiny at int8int8 (native int8) and of the 1.5B at int8int8 (the `.gguf`, a direct load) at 3,900 rows, last logits | byte-equal everywhere, and the int4 cells of D5-G1 still byte-equal |
| D5c-G2 scratch | 0 | D5-G2's counter, the 1.5B at int8int8, one 3,900-row pass | at most 123 MB |
| D5c-G3 speed | timed, Mac, timing lock | D5-G3's instrument on the 1.5B at int8int8 | SHIP iff the median fused/two-step ratio is ≤ 1.01; otherwise PARK with the reading (the owner's bit-identical rule of 2026-10-23 also ships it at 9 of 10 pairs faster) |

Planted defect: the up rows read the gate rows' scale (`WS[i0 + (fr & 31)]`). D5c-G1's kernel cells go red. Stopping rule: one pass
each, no further build on a miss. Cost: seconds; D5c-G3 about 3 minutes with the load.

**Amended 2026-10-23 (10:15), before any model reading: the 0.5B replaces the 1.5B in D5c-G1's model cell, D5c-G2 and D5c-G3.** The
fit guard refused the 1.5B at int8int8 (2.8 GB needed against a 2.1 GB budget, 3.0 GB available with the owner's apps open). The
0.5B (`qwen2.5-coder-0.5b-instruct-q4_k_m.gguf` at int8int8, a direct load) takes the same 64 × 64 tile at 3,900 rows with
I = 4,864, a multiple of 32, so it exercises the same mechanism; the tier rule asks for the smallest such model. D5c-G2's bar becomes
the 0.5B's own: the two-step pass's counted scratch less `guF` (Mpad × 2 × 4,864 halves), within 1%. The bars of D5c-G1 and D5c-G3
are unchanged. The kernel cells (K = 1,536, I = 96 and 8,960) already ran and passed at 10:13; they stand.

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

**KS-D7 shipped by owner decision, 2026-10-23 (8:36 am), over the registered 2% bar.** Francis: "it seems silly to give up gains,
any gains, if it's bit-identical and doesn't use more memory." The re-run read a median of 0.9888 with 10 of 10 pairs below 1
(`docs/measurements/ks-night-2026-10/README.md`). His condition is a gate, registered here before it ran: **D7-G3 memory**. The
1.5B (the Metal `.giw`), a 3,900-token prefill per arm, the half2 build against the per-weight build: the prefill scratch counter
(`prefillScratchSeen`) and Metal's allocated size after the prefill must be equal or lower, and the logits byte-equal. If memory
goes up, stop and tell him.

**KS-D8 · Batched Mamba-2 prefill on Metal (Nemotron-H).**
- **Waste:** `metal/model.go:resident.recurrentPrefillOK` declines Mamba, so every prompt token is a decode step.
- **Gain (est.):** the DeltaNet precedent was 6.7–8.4×, so 5–8×.
- **Class:** fidelity-gated, like D-B01.
- **Size:** M–L. Pairs with KS-B12.

**KS-D8 baseline, registered 2026-10-23 before the run, on the owner's order of 11:50 (the night queue, case (a)).** Instrument:
`TestKSD8_sequentialPrefillBaseline`, NVIDIA-Nemotron-Nano-9B-v2's int4 Metal `.giw` resident on Metal (`ResidentContext` 1024),
one warm 512-token greedy generation of one token, then three timed ones on prompts that differ in their first token; the reading
is 512 / time to the first token, each run's start load recorded. The test also prints whether the resident admits the batched
prefill, which must read false (`recurrentPrefillOK` declines Mamba). Read, not gated: it is the denominator for KS-D8's estimated
5–8× and KS-B12's CUDA baseline. Through `docs/measurements/ks-night-2026-10/run-ksd8-baseline.sh` (a test binary built at the
registering commit; exits 75 on a fit refusal). Stopping rule: one pass. Cost: about 5 minutes.

**KS-D8 baseline read 2026-10-24 (08:53–08:56, by day on the owner's word; the timing lock held exclusively).** It never started on
the night queue (its start check read about 6.3 GB free). It ran from `run-ksd8-baseline-day.sh`: the test binary rebuilt at
0140adf1 with #203's memory reading applied (`memory_pressure`'s free % × RAM − 500 MiB), so the fit guard passed with no bypass; 11.95 GB
available, swap 464 MB before and after. Each run's start load 1.92–2.68. Nemotron-Nano-9B-v2's int4 Metal `.giw` loaded in 14.9 s,
**batched prefill admitted: false** (as required); a 512-token prompt to the first token in 35.30 / 35.27 / 35.16 s: **14.5 tok/s**.
That is the denominator for KS-D8's estimated 5–8× and for KS-B12. Also seen: the resident keeps no recurrent checkpoint here
(138.8 MiB beside the load's 10.49 GB is over its 6.55 GB budget), so a reply rendered back differently re-prefills the whole
prompt. Log: `~/goinfer-logs/ksd8-baseline-day-2026-10-24-0853.log`.

**KS-D8 design and gates, registered 2026-10-24 before the code (D8-G), on the owner's word of 10:22 (start KS-D8 now; design
and gates first).** What the pass lacks for Nemotron-H, read off `MissingResidentFeatures(prefillFeatures)` on `nemotron-tiny`
(layers Mamba, attention, MLP, Mamba, attention) and on the 9B's three-layer slice (Mamba, MLP, attention; dInner 10,240, dState
128, in_proj 22,656 × 4,480): `ssm`, `non-gated-mlp` and `non-gated-relu2`. Its attention layers are already in the pass. The build:
- **The layer loop takes decode's shape:** the mixer by kind (DeltaNet, Mamba-2, attention, or none for an MLP-only layer), then
  the FFN only where the layer has one (`L.ffn`), as `encodeLayerResidualWith` does.
- **The Mamba-2 mixer over M rows** (`encodePrefillMambaMixer`, a new `metal/prefill_mamba.go`, after D-B01's DeltaNet form):
  the pre-norm into the pass's f16 norm scratch; in_proj as an int8 prefill GEMM (`gemm_w8f16_*`, which takes the int8 codes and
  per-row f32 scales the layer already holds; the W8 tiles compile for a Mamba resident too); then three row kernels over the M
  rows, `mamba_conv_seq` (a thread per channel, the window in registers), `mamba_ssm_seq` (a thread per head and channel, the state
  in the layer's buffer as decode keeps it) and `mamba_gnorm_rows` (a threadgroup per row and group). Each is its decode kernel's
  body with a token loop, so given the same inputs it writes decode's bits, window and state included. Then the gated rows to f16
  and out_proj as an int8 GEMM accumulating into the f16 residual (mode 2). The window and state continue from what the resident
  holds, as M decode steps would.
- **The non-gated relu² MLP:** norm, the up projection (int4 GEMM, its bias by mode 1), `nongated_act_rows_f16` (decode's
  `nongated_act` per element), the down projection accumulating into the residual (mode 2). Admitted only where the down projection
  has no bias (Nemotron-H); a biased one (GPT-2) stays sequential, so the shared GEMM kernel is not touched.
- **Admission:** `recurrentPrefillOK` admits a Mamba resident when `mambaPrefillOn` (a test switch, on) and dConv ≤ 8; the three
  features leave the missing list as DeltaNet's does. The new kernels live in `metal/kernels/mamba_kernels_msl.go`.
- **Class: fidelity-gated.** What leaves decode's arithmetic is what every batched layer changes: f16 activations into the
  projections and f16 projection outputs, where decode runs W8A8 with f32 outputs.

| gate | tier | instrument | PASS iff |
|---|---|---|---|
| D8-G1 kernels | 0 | each row kernel against M steps of its decode kernel at `nemotron-tiny`'s and the 9B's mixer shapes, M = 1, 7, 64: random inputs representable in f16 (the row kernels read the f16 projection rows, decode the f32), a non-zero starting window and state | outputs, window and state byte-equal |
| D8-G2 fidelity, tiny | 0 | `nemotron-tiny`: a CPU f32 reference, the Metal int4 sequential path and the Metal int4 batched pass, prompts of 16, 48 and 100 tokens (8 prompts each, seeded ids), last-position logits | the batched pass is non-inferior to the sequential one against the reference: its mean KL to the reference at most the sequential path's + 0.005 nats, and its top-1 agreement with the reference at least the sequential path's − 1 prompt |
| D8-G3 fidelity, 9B | 1, Mac | Nemotron-Nano-9B-v2's int4 Metal `.giw`, 8 prompts of 512 seeded token ids, last-position logits, batched against sequential (no f32 reference fits this box) | top-1 agrees on at least 7 of 8 and the mean symmetric KL ≤ 0.05 nats: PASS; 6 of 8 or KL in (0.05, 0.10]: PARK for a CPU f32 reference on nobara; worse: FAIL |
| D8-G4 speed | timed, Mac, timing lock | the 9B, a 512-token prompt's time to the first token, batched against sequential, interleaved A B B A over 2 pairs after one warm run each (the sequential arm is 35 s; its baseline read 14.5 tok/s) | SHIP iff the median speedup is ≥ 2× |
| D8-G5 memory | 0 and 1 | the pass's scratch counter (`prefillScratchSeen`) against `PrefillScratchBytes` on `nemotron-tiny` and the 9B at 512 rows; the 9B's 512-token prompt admitted by the fit guard with the pass priced | counted ≤ planned; admitted |
| D8-G6 unchanged | 0 | `touched.py`; the prefill, DeltaNet, MEM6/MEM7 and attention-prefill tests by name; `llama-tiny`, `qwen35-tiny` and the 1.5B batched-prefill logits against a resident built at `origin/main` (their paths call no new code) | all PASS; byte-equal |

Planted defects, run once on the committed tree and reverted: `mamba_ssm_seq` restarting the state at each row (D8-G1 red), and the
batched pass skipping the conv window's write-back (D8-G1 red on the window). Stopping rule: one pass each; a FAIL is diagnosed,
not re-run. Cost: G1, G2, G5 and G6 minutes; G3 about 10 minutes (16 prompt passes, 8 of them sequential at 35 s); G4 about 3
minutes. Size: L (the estimate was M–L; the relu² MLP and the mixer-only layers were not in it).

**KS-D8 read 2026-10-24 (Mac, 10:30–11:15): D8-G3 FAIL by its registered bar; every other gate passes; not merged.**
- **D8-G1 PASS:** 12 cells (fast-math and precise, nemotron-tiny's and the 9B's shapes, M = 1, 7, 64), 0 values differ in the
  outputs, the window and the state. Both plants are red: the scan restarting its state (8,090,871 values differ) and the window
  left unwritten (every window value).
- **D8-G2 PASS** (nemotron-tiny against the CPU f32 reference, mean KL; top-1 of 8): M = 16 sequential 5.54e-5, pass 5.28e-5,
  8 and 8; M = 48 4.85e-5 and 4.63e-5, 8 and 8; M = 100 3.07e-5 and 3.15e-5, 6 and 6. Graded per prompt length, the stricter of
  the registration's two readings.
- **D8-G3 FAIL:** the 9B, batched against sequential, top-1 agrees on **3 of 8**, mean symmetric KL **0.0504** (per prompt
  0.025–0.124), cosine 0.994–0.997 on every prompt. The top-1 flips are between neighbours (43085 and 7612 swap between prompts
  4 and 5).
- **Diagnostic, after the FAIL (not a gate):** the same instrument on two families whose batched pass ships. Qwen3.5-9B (D-B01)
  agrees on **6 of 8**, mean KL **0.0384**, cosine 0.987–0.998; Qwen2.5-Coder-1.5B on 7 of 8, KL 0.0192. So the bar was
  miscalibrated for random-token prompts (a shipped 9B hybrid reads PARK on it), and the 9B Nemotron still reads worse than that
  yardstick: KL about 30% higher and top-1 3 of 8 against 6 of 8, at the same cosines. Which path is nearer the true output needs
  a CPU f32 reference this box cannot hold (the registered PARK route, on nobara).
- **D8-G4 SHIP** (11:05–11:08, the lock held exclusively, load 0.70–0.97, 12.9 GB available): a 512-token prompt sequential
  39.10 and 40.74 s (13.1 and 12.6 tok/s), batched 7.93 and 7.61 s (64.6 and 67.3 tok/s): **median speedup 5.14×** (the survey's
  estimate 5–8×).
- **D8-G5 PASS:** scratch counted equals the plan, nemotron-tiny 231,456 bytes at 100 rows, the 9B 138,610,688 at 512; the 9B
  loads under the fit guard.
- **D8-G6 PASS:** llama-tiny, qwen35-tiny and the 1.5B, batched-prefill logits 0 differ against a build at the base commit;
  `touched.py`'s one red was `TestRecurrentGuards_declineMamba`, which asserted the decline this item removes (updated); the
  other reds were fixtures absent from a new worktree, green once cloned in, and `TestI8_exactKernelModelIsSplit`, red on main too.
Logs: `~/goinfer-logs/ksd8-g1-2026-10-24.log`, `~/goinfer-logs/ksd8-g2-2026-10-24.log`, `~/goinfer-logs/ksd8-9b-2026-10-24.log`,
`~/goinfer-logs/ksd8-yardstick-2026-10-24.log`, `~/goinfer-logs/ksd8-g6-2026-10-24.log`.

**D8-G3's reference route, registered 2026-10-24 (11:55) before the reference runs, on the owner's word of 11:46 ("KS-D8 is a
go"; the registered PARK route).** Instrument: the same 8 prompts of 512 seeded token ids. The reference is the Hugging Face
checkpoint on nobara's CPU (`/srv/models/nemotron-hf`; not a timed read) with its bf16 weights kept bf16 and f32 compute
(`Quant: "bf16"`), which is the checkpoint's own arithmetic; its last-position logits are written on nobara and copied to the
Mac, where the sequential path and the batched pass are read as in D8-G3. **D8-G3 PASS iff, against that reference, the pass is
non-inferior to the sequential path in D8-G2's form: its mean KL(ref ‖ ·) at most the sequential path's + 0.005 nats, and its
top-1 agreement with the reference at least the sequential path's less one prompt; otherwise FAIL.** The 9B ships on PASS, with
D8-G4's speed; on FAIL it stays unmerged and the gap is localized per layer kind before anything else.

**D8-G3 reference-route read 2026-10-24: PASS.** The reference ran on nobara's CPU 12:54–14:42 (`ksd8-g3-cpu-ref`, all 8 prompts,
`ref.bin` sha256 `d46c44ce…`, 8 × 131,072 float32); the Mac read the sequential path and the batched pass against it 14:46–14:54
(not timed; shared lock; 13.7 GB available at the start). Against the reference, **the pass is closer than the sequential path on
every prompt**: mean KL(ref ‖ ·) **0.0463 for the pass against 0.1024 for the sequential path** (bar: ≤ 0.1074), top-1 agreement
**7 of 8 against 4 of 8** (bar: ≥ 3). Per prompt, KL(ref ‖ seq) → KL(ref ‖ pass): 0.081 → 0.040, 0.028 → 0.022, 0.225 → 0.056,
0.281 → 0.111, 0.027 → 0.017, 0.023 → 0.016, 0.055 → 0.028, 0.099 → 0.081; the pass misses top-1 only on prompt 7, the sequential
path on prompts 2, 4, 5 and 6. So D8-G3's first FAIL measured the sequential path's own distance from the checkpoint, not a
defect in the pass. KS-D8 ships with D8-G4's 5.14×. Log: `~/goinfer-logs/ksd8-g3-ref-2026-10-24.log`.
After merging main at 922054f3 (KS-O1's q4k-lane GEMVs touch the same files): D8-G1, D8-G2 and D8-G5 PASS again, and D8-G6 against a
fresh base dump of 922054f3 reads 0 logits differ on llama-tiny, qwen35-tiny and the 1.5B's Metal `.giw`.

**After the review of #222 (15:15), registered before the re-runs.**
- **The down-bias decline was a no-op, fixed.** `buildResident` read `residLayer.downBiasSet` before `r.layers` was built, so
  the check scanned an empty slice and the non-gated MLP features were always admitted. A non-gated MLP with a down-projection bias
  would have taken the pass, which has no down-bias epilogue. The check now reads the weights (`LayerWeights.DownBias`).
  `TestKSD8_downBiasDeclines` builds nemotron-tiny as shipped (admitted) and again with one layer's `DownBias` set (it must decline);
  it is run once against the unfixed check, where it must be red.
- **Not a defect: the f16 conversion past `M·dInner`.** aikit/gpu's Metal `Encoder.Dispatch` encodes `dispatchThreads` with the
  exact element count, so its last threadgroup is partial and `f32_to_f16` (indexed by `thread_position_in_grid`) never reads past
  `gated` or writes `gatedH`'s pad rows. D8-G2 still gains odd lengths (17, 33, 101), where `M·dInner` is off a multiple of 256.
- **D8-G2 gains a continuation:** after each prompt, both paths decode 4 more tokens from position M, and the last is graded
  against the CPU reference over all M+4 tokens on the same non-inferiority bar. A wrong window, state or K/V left by the pass
  would show there.
- `p.dConv < 2` now declines the pass too (the conv kernel indexes `w[K-2]`).
- `TestKSD8_9BFidelity` (the first D8-G3 instrument, batched against sequential) is retired; D8-G3's instrument is
  `TestKSD8_9BAgainstReference`. `TestKSD8_9BSpeed` refuses to run outside the timing lock and fails under D8-G4's bar.
- Re-runs: D8-G1, D8-G2 (extended), D8-G5, the new decline test (and its red run against the old check), and D8-G6 against
  the 922054f3 base dump, since the admission change could touch another non-gated family.

**Read 2026-10-24 (15:17–15:25, Mac, not timed): all PASS.**
- `TestKSD8_downBiasDeclines` is red against the old check ("a non-gated MLP with a down-projection bias was admitted") and green
  with the fix.
- D8-G1 0 values differ in 12 cells; D8-G5 counted equals planned (231,456 bytes).
- D8-G2 PASS at all six lengths, at the prompt's end and after the continuation. Pass against sequential KL, at the end then after
  4 steps: M=17 7.39e-5 vs 7.33e-5, then 5.76e-5 vs 5.82e-5; M=101 3.68e-5 vs 3.71e-5, then 3.36e-5 vs 3.41e-5. Top-1 is equal to
  or better than the sequential path everywhere.
- D8-G6: 0 logits differ on llama-tiny, qwen35-tiny and the 1.5B.
- **Plant, run once and reverted:** `mamba_conv_seq` not writing its window back. D8-G2 goes red. Every length's after-continuation
  KL rises 10–20× (5e-5 → 4.6e-4 to 1.0e-3) while the prompt-end readings are unchanged. Only M = 33, 48 and 100 fail outright: the
  registered +0.005-nat margin is about 100 times this tiny model's KL, so it lets a 20× rise through at the other three. The bar
  is left as registered; a relative margin for a tiny-model continuation would be its own registration.
Logs: `~/goinfer-logs/ksd8-review-g125-2026-10-24.log`, `~/goinfer-logs/ksd8-review-g6-2026-10-24.log`,
`~/goinfer-logs/ksd8-review-plant-2026-10-24.log`.

**Correction to the plant reading above, 2026-10-24 (after the review of #222).** The three lengths that failed (M = 33, 48 and
100) failed on the **top-1 clause** (pass 4, 4 and 5 of 8 against the sequential path's 6, 8 and 7), not on KL. The KL clause failed
**none** of the six: the 10–20× rise (to 4.6e-4–1.0e-3) stayed far inside the flat +0.005-nat margin. So the KL clause was vacuous at
this model's scale, and the plant was caught by the top-1 clause alone.

**D8-G2's margin scaled, registered 2026-10-24 (23:20) after that review.** The KL clause's margin δ becomes half the sequential
path's KL, at least 1e-5 and at most the registered 0.005 nats: never looser than registered, and on nemotron-tiny (both KLs about
5e-5) a 1.5× rise fails it. The basis is the readings already recorded: clean, the pass sits at 0.91–1.06 of the sequential path's KL
(both reads on the merged tree); under the window plant, at 13× or more. KS-D8 is merged, so this decides nothing about shipping; it
makes D8-G2 able to catch what it exists for. The top-1 clause is unchanged. Also added: `TestKSD8_mambaShapeDeclines`, the
batched Mamba-2 pass's shape guards in `recurrentPrefillOK` (pass off, a DeltaNet layer beside the Mamba ones, postOnly, conv width
outside 2..8, hidden or inner width off a multiple of 32), each set alone on a built nemotron-tiny resident, plus one through
`buildResident` (pass off → `prefillOK` false).

**Read 2026-10-24 (23:20–23:30, Mac, not timed).**
- D8-G2 PASS at all six lengths on the scaled margin: pass/sequential 0.91–1.01 at the prompt's end and after the continuation.
- The window plant, run once and reverted, is now red on the **KL clause at all six lengths**: after the continuation the pass reads
  13.0–28.0× the sequential path's KL. The top-1 clause still catches three of them.
- `TestKSD8_mambaShapeDeclines` PASS; the plant (the `dConv < 2` decline dropped) is red: "conv width 1: recurrentPrefillOK admits it".
Logs: `~/goinfer-logs/ksd8-g2-scaled-2026-10-24.log`, `~/goinfer-logs/ksd8-g2-scaled-plant-2026-10-24.log`.

**KS-D9 · Batched phase 2 for the generic paged MoE prefill.**
- **Waste:** `metal/prefill_g4paged.go:resident.moeLayerMajorRows` runs experts per row and re-reads the shared expert for
  every prompt row (19.5 MB per layer on Qwen1.5-MoE).
- **Change:** reuse `moe_batch_gemv`.
- **Gain (est.):** ~1.2×.
- **Class:** bit-identical. **Size:** M.

**KS-D9 gates, registered 2026-10-23 before the code (D9-G), on the owner's word of 2:30 pm.** The build, D-P01's shape on the
generic pager: `moeLayerMajorRows`' phase 2 runs a slot group's routed experts expert by expert through `moe_batch_gemv` (every
pair routed to an expert in one pass over its weights), `mc3_swiglu_quant_rows` per pair, the down projection's raw sums, and a new
`moe_batch_combine_acc` that adds each row's k experts in route order with `gemv_w4a8_moe_wacc`'s epilogue `fma(wgt·acc, asc, out)`
starting from the row's own residual, as the per-row path's k accumulating dispatches leave it. The rows' residuals are packed into
the batch before and copied back after (`copy_f32`). The shared expert stays per row, after the combine, as today. It applies where
both projections fill whole threadgroups (`moeBatchR`) and the experts are int4 and unbiased (not the native int8 lane, not
gpt-oss); `moeExpertBatchOn` keeps the per-row phase 2 for the A/B.

| gate | tier | instrument | PASS iff |
|---|---|---|---|
| D9-G1 identity | 0 | `TestMoELayerMajor_matchesSequential` (Mixtral, two Qwen3-MoE shapes, the Qwen3.5 DeltaNet MoE hybrid; 2 and 3 slots; 4 prompt shapes) with the batched phase 2 taken, and the same cells batched against per-row (`moeExpertBatchOn`) | byte-equal everywhere (logits, K/V, 3 decode steps), and the batched path taken on every fixture whose shapes admit it |
| D9-G2 speed | timed, Mac, timing lock | a real generic MoE paged on the Mac, its layer-major prefill of 512 rows, batched against per-row phase 2 on one resident, A B B A over 3 pairs after one warm run each; the model named in an amendment before the run (none is on this Mac's `~/models` today; Qwen1.5-MoE-A2.7B is the survey's shape) | SHIP iff the median batched/per-row time is ≤ 0.98 with every pair faster; otherwise PARK |

Planted defect: the combine starting from zero instead of the residual; D9-G1 goes red. Stopping rule: one pass each. Cost: G1
seconds; G2 a few minutes plus the model's pull.

**KS-D9 read 2026-10-23, by day.** D9-G1 **PASS** (14:36): `TestMoELayerMajor_matchesSequential`'s 7 paged cells (Mixtral and
two Qwen3-MoE shapes at 2 and 3 slots, the k=3 shape at 3, the Qwen3.5 DeltaNet MoE hybrid at 2 and 3) bit-identical to sequential
decode in logits, K/V and 3 decode steps, the batched phase 2 taken in every cell (42–82 slot groups); `TestMoELayerMajor_generateThroughDecoder`
PASS. The plant (the combine from zero) first tripped the startup self-test, which declined the backend so the cells skipped; with
the self-test skipped (`TestKSD9_plantSeenByTheGate`) every cell went red. Also: `touched.py` green; the MoE and paged families
by name 55 pass, 17 skip, 0 fail. D9-G2 waits for a real generic MoE: none is in `~/models`, and nobara's archive was under another
session's timing lock at 14:40, so no pull. Logs: `~/goinfer-logs/ksd9-*-2026-10-23.log`.

**D9-G2's model, amended 2026-10-24 before the run (the registration asked for it; Francis's word of 12:45).** The model is
Qwen1.5-MoE-A2.7B, its int4 Metal `.giw` transcoded on nobara from the archive copy (`cmd/prequant -quant int4 -target metal`,
built at origin/main) and copied to the Mac, then deleted once graded. It is paged with `GOINFER_METAL_MOE_SLOTS=16` (of its 60
routed experts, top-4), so the generic pager's layer-major prefill is the path timed. Instrument `TestKSD9_layerMajorSpeed`:
512 seeded token ids, `PrefillLast`, the bar unchanged (SHIP iff the median batched/per-row is ≤ 0.98 with every pair faster;
otherwise PARK). Each pair also counts the last logits that differ, which must be 0 (D9-G1's class).

**D9-G2's instrument, amended 2026-10-24 (committed 15:05:22, ffbe2da4) before the run (started 15:06:56)**, to the void rule as #214 and #216's second reviews applied it:
the test refuses to run outside the timing lock; each arm reads the 1-minute load and reclaimable memory just before and just after
its timed run (after the load and the warm runs), and a pair with any read over the line (load above 4.0, under 2 GiB reclaimable)
is void and re-run, up to 6 times; an unreadable load or memory reading fails the test; any differing last logit fails it. The bar
is unchanged. The `.giw` sits in `~/goinfer-bench/qmoe/` on the Mac's internal SSD (the session's brief keeps it out of
`~/models`; the same disk, not an archive path).

**D9-G2 read 2026-10-24 (15:06–15:15, the lock held exclusively; 13.9 GB available and 8.1 GB reclaimable at the start): SHIP.**
Qwen1.5-MoE-A2.7B's int4 Metal `.giw` (sha256 `da920a0d…`, equal on both boxes) paged at 16 slots, a 512-row layer-major
`PrefillLast`. Per-row phase 2 16.08, 15.98 and 16.08 s; batched 14.55, 14.47 and 15.60 s; **median batched/per-row 0.9050, 3 of 3
pairs faster**, 0 of 151,936 last logits differ in every pair. Six pairs were void and re-run: the load rose to 10.9 during
one batched arm, then stayed between 3.7 and 10.1 for five attempts. Every valid arm read a load of 2.09–3.72 and 6.60–6.88 GiB
reclaimable. Log: `~/goinfer-logs/ksd9-g2-2026-10-24.log`. The `.giw` was deleted from the Mac after grading (Francis's word of 12:45);
nobara's copy is in `~/goinfer-bench/qmoe/`.

**After the review of #224 (15:45), registered before the re-runs.**
- **The batch scratch was outside the plan and could pile up, fixed.** `moeBatchEnsure` grew the batched phase 2's buffers per slot
  group, inside `moeLayerMajorRows`, and a later, larger group allocated new buffers while the superseded ones stayed alive until the
  call returned; the fit plan priced none of it. Now `newMoEBatch` allocates it once per call, sized for all the call's rows (at
  most `g4LayerMajorChunk`; a group's entries are bounded by its pairs), and `prefillScratchBytes` adds `moeBatchBytes` of
  min(M, `g4LayerMajorChunk`) for a paged MoE the batched phase 2 admits. A group that would overflow it records an execution error
  instead of growing. `TestKSD9_scratchWithinPlan`: one 9-row call counts exactly `moeBatchBytes(9)` (`prefillScratchSeen`), and the
  plan's growth with the batched phase 2 on is `moeBatchBytes` at 9 and at 256 for a 300-row prompt.
- **D9-G1's test tightened:** a declined resident fails it (only a slot count under k skips); every fixture must admit the batched
  phase 2 and every prompt shape must take it; a third arm, layer-major with the per-row phase 2, must equal the batched one bit for
  bit (logits, K/V, 3 decode steps), the registered batched-against-per-row comparison. The plant check re-runs the gate in a child
  process with the self-test skipped (`GOINFER_KSD9_PLANT_CHILD=1`), so the skip no longer leaks into the rest of the test binary.
- **D9-G2's test:** each arm must take the phase 2 it names. The void rule's load term is read directly, as for #214: `voidWindow`
  brackets each timed prefill with `hostmem.ReadCPU` and voids the arm if the busy cores averaged over it exceed half the cores, or
  reclaimable memory is under 2 GiB at either end. The schedule stays the registered one, A B B A over 3 pairs (A B, B A, A B): an
  odd number of pairs cannot balance, which is why the bar asks every pair to be faster.
- **Re-runs:** D9-G1 and the new scratch test; the plant once (the combine from zero, which must be red in the child); then D9-G2
  again under the lock, since the scratch's allocation moved: the `.giw` is copied back from nobara for it and deleted after. The
  bar is unchanged; this reading decides, and 15:06's stays in the record.

**Read 2026-10-24 after the review: all PASS, D9-G2 SHIP.**
- D9-G1 (16:14–16:16, not timed): 7 paged cells PASS with the per-row arm equal to the batched one in logits, K/V and 3 decode
  steps, the batched phase 2 taken on every prompt shape (42–82 slot groups a cell); k=3 at 2 slots skips (the pager refuses
  under top-k). `TestMoELayerMajor_generateThroughDecoder` PASS. `TestKSD9_scratchWithinPlan` PASS (counted equals
  `moeBatchBytes(9)`; the plan's growth equals it at 9 and at 256 for 300 rows).
- **Plant** (the combine from zero, run once and reverted): the child-process gate red in all 7 cells; run plainly, the gate now fails
  (it skipped before) with "startup self-test declined this backend".
- **D9-G2** (16:22–16:24, the lock held exclusively; 13.4 GB available, 8.9 GB reclaimable at the start; the `.giw` copied back from
  nobara, sha256 equal, and deleted after): per-row 16.18, 18.95, 17.86 s; batched 14.53, 14.78, 14.72 s; **median batched/per-row
  0.8241, 3 of 3 pairs faster**, 0 of 151,936 last logits differ in every pair, no void pairs; every arm 2.19–3.00 busy cores and
  6.23–6.38 GiB reclaimable. The batched arm held within 0.25 s across both sessions (14.47–14.78 s), and the per-row arm spread
  wider this time, so the size is less settled than the direction: 0.9050 at 15:06 and 0.8241 now.
Logs: `~/goinfer-logs/ksd9-review-g1-2026-10-24.log`, `~/goinfer-logs/ksd9-review-plant-2026-10-24.log`,
`~/goinfer-logs/ksd9-g2-rerun-2026-10-24.log`.

**The guard's CPU reading fixed, 2026-10-24 (17:05), registered before a re-run.** `hostmem.ReadCPU` read `host_statistics64(HOST_CPU_LOAD_INFO)`,
which XNU rate-limits for non-Apple binaries: after a few calls in a second it serves the last reply again, so an arm's end sample
could be up to a second stale (three calls in 0.4 s returned the same sample). It now reads `host_processor_info(PROCESSOR_CPU_LOAD_INFO)`,
per core, which is not rate-limited. Each state's count is differenced on its own, modulo 2^32 on macOS and clamped at 0 on Linux
(NOHZ can step idle back a tick), so busy never exceeds total. A sample pair spanning no interval now voids the arm instead of
failing the run, and counts toward the void cap. The memory term still reads `host_statistics64`, so it can be up to a second
stale; that is immaterial for a point reading against a 2 GiB line. The guarded readings above were taken on the old reading, so
the gate is re-run on the new one; that run decides, and the bar and the cap are unchanged.

**D9-G2 read on the fixed guard, 2026-10-24 (17:40–17:42, the lock held exclusively; 10.0 GB reclaimable at the start; the `.giw`
copied again, sha256 `da920a0d…`, and deleted after): SHIP.** Every arm valid (2.10–2.51 busy cores, 6.41–6.57 GiB reclaimable); no
void pairs; each arm took its phase 2 and every pair's last logits were equal. **Median batched/per-row 0.9131, 3 of 3 pairs
faster.** The three readings: 0.9050, 0.8241, 0.9131. Log: `~/goinfer-logs/ksd9-g2-fixedguard-2026-10-24.log`.

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

**KS-E2 gates, registered 2026-10-24 before the code (E2-G), on the owner's order of the Mac work (the small items while KS-D8
waits on its reference).** The build: the six quantizers (`rmsnormQuantWGSL`, `swigluQuantWGSL`, `gegluQuantWGSL`,
`quantizeShaderWGSL`, `relu2QuantWGSL`, `gptossGluQuantWGSL`) and the prefill row forms derived from three of them compile
widened to 256 lanes: the shared array, the workgroup size, the max loop and the pack loop at 256, the max tree from stride 128.
`rmsnormQuantWGSL`'s sum of squares keeps lanes 0–63 in today's order (its loop guarded to them, its tree from stride 32), so its
scale is today's bits. The source stays the 64-lane text and is widened by exact-count edits where it is compiled; a Context
built with `quantNarrowForTest` (read at `New`) compiles the 64-lane form for the A/B.

| gate | tier | instrument | PASS iff |
|---|---|---|---|
| E2-G1 identity | 0 | (a) each widened shader against its 64-lane form on random inputs at n = 1, 3, 255, 896, 4864, 8960 (the max and the pack at partial lanes), and the row forms at M = 1, 3, 17; (b) llama-tiny and the 1.5B's `.gguf` on WebGPU at int8int8, two residents (narrow and wide Contexts), 32 greedy decode steps' logits | the packed int8 and the scales byte-equal; the logits byte-equal |
| E2-G2 speed | timed, Mac, timing lock | a whole-token in-process A/B on the 1.5B on WebGPU: the narrow and the wide resident in one process, 128 greedy decode tokens after a 32-token prompt, A B B A over 10 pairs after 2 warm runs each | SHIP iff the median wide/narrow per-token time is ≤ 0.98 with at least 9 of 10 pairs faster; under it, the owner's rule for a bit-identical change decides; otherwise PARK |

E2-G3: none; the change allocates nothing. Planted defect, run once and reverted: the widened rmsnorm sum taken over all 256 lanes
(today's order broken); E2-G1 goes red on the rmsnorm cells, or, if every cell stays equal, a pack loop stopping at 128 lanes
must. Stopping rule: one pass each. Cost: G1 a minute; G2 about 4 minutes.

**KS-E2 read 2026-10-24 (Mac, 11:50–12:01): SHIP.**
- **E2-G1 PASS.** (a) 63 cells: the six shaders at n = 1, 3, 255, 896, 4864, 8960 and the three row forms at n = 255, 896, 4864 ×
  M = 1, 3, 17, 0 words differ (packed int8 and scales). The plant (rmsnorm's sum over all 256 lanes) turned the rmsnorm cells red,
  so the guarded sum is what keeps the widened kernel exact. (b) llama-tiny 0 of 8,192 and the 1.5B 0 of 4,861,952 logits differ
  over 32 steps, a wide against a narrow resident (each Context checked to carry its width).
- **E2-G2 SHIP** (11:59–12:01, the lock held exclusively; the 1-minute load 3.49 at the start, each pair's 1.22–2.56; 12.6 GB
  available by the load figure, the reclaimable one not captured): narrow 24.66–25.56 ms a token, wide 21.81–21.94, **median
  wide/narrow 0.8821, 10 of 10 pairs faster**: the 1.5B's WebGPU decode 11.8% faster, about 2.9 ms a token (the survey's estimate
  0.78 ms).
Logs: `~/goinfer-logs/kse2-g1a-2026-10-24.log`, `~/goinfer-logs/kse2-g1b-2026-10-24.log`, `~/goinfer-logs/kse2-g2-2026-10-24.log`.

**E2-G2 re-run, registered 2026-10-24 (13:00) before it runs, after the second-opinion review of #214.** The first reading (0.8821)
was taken under the timing lock (its runner took it exclusively; the 1-minute load 3.49 at the start), but the test itself enforced
neither the lock nor the void rule, the reclaimable figure was not captured, and its 2.9 ms a token is 3.7× the survey's estimate.
The re-run, in one process under the lock, the test refusing to run without the lock's token:
- **A/A control first:** two narrow residents, the same A B B A over 10 pairs after 2 warm runs. PASS iff the median ratio is within
  1 ± 0.02; outside it, the two-resident setup biases the reading and E2-G2 is not graded.
- **Then the A/B** (narrow against wide), as registered: SHIP iff the median wide/narrow is ≤ 0.98 with at least 9 of 10 pairs
  faster.
- **Every pair records, at its start, the 1-minute load and the reclaimable memory** (`internal/hostmem`); a pair started above a
  load of 4.0 or under 6 GB reclaimable (the two residents and the host model, about 4 GB, plus 2 GB) is void, listed and re-run in
  order, up to 10 re-runs.
The first reading stays in the record; the re-run decides.

**Amended 2026-10-24 (13:06), after the first re-run attempt voided every pair and before any valid timed reading.** The 6 GB
line was applied at each pair, after the test's own loads (the host model and two residents, about 4 GB) had taken their memory, so
it counted that memory twice and every pair start read 1.7–2.3 GB reclaimable: void. The void rule's term is the memory the cell's
load needs plus 2 GB, read before the load. So: **reclaimable ≥ 6 GB before the loads** (the test refuses to start under it), and at
each pair start **load ≤ 4.0 and reclaimable ≥ 2 GB** (the margin still there after the loads); a pair under either is void and
re-run, as registered. Everything else is unchanged.

**Second attempt read 2026-10-24 (13:08–13:20), incomplete; amended before a third (13:22).** Before the loads 8.8 GB and 10.2 GB
reclaimable; after them, 1.4–2.8 GB at the pair starts, at the 2 GB margin, so most starts were void and each run stopped at its
10-void limit. Valid pairs, recorded and not graded: **A/A 2 pairs, 0.9910 and 1.0025; A/B 7 pairs, 0.8777–0.8911** (the first
reading's 0.8821 again). The two residents and the host model leave too little margin on this Mac, so the third attempt keeps **one
resident alive at a time**: each run builds a fresh narrow or wide resident, warms it twice, times it and closes it, the A B B A
order and the bars unchanged, the 6 GB before-load and 2 GB per-pair lines unchanged. Log of the second attempt:
`~/goinfer-logs/kse2-g2-rerun2-2026-10-24.log`; of the first: `~/goinfer-logs/kse2-g2-rerun-void-2026-10-24.log`.

**E2-G2 re-run read 2026-10-24 (13:21–13:34, one resident at a time, under the lock): SHIP stands.** Before the loads 9.2 and 9.0
GB reclaimable. **A/A control PASS: median 0.9998**, 5 of 10 pairs below 1, 4 void starts (the 1-minute load rose to 11.7 for a
minute mid-run; the starts above 4.0 were re-run); two valid pairs near that spike read 0.8426 and 1.1594. **A/B SHIP: narrow
24.36–28.11 ms a token, wide 21.53–24.66; median wide/narrow 0.8811, 10 of 10 pairs faster**, 3 void starts. Each valid pair
started at a load of 0.64–3.43 and 3.4–7.3 GB reclaimable. The gain is the first reading's (0.8821) again; on this Mac's WebGPU
the 1.5B's single-workgroup quantizers cost more than the survey's 0.78 ms, which was read on nobara's 2070. Log:
`~/goinfer-logs/kse2-g2-rerun3-2026-10-24.log`.

**After the second review of #214 (14:20), registered before a further re-run.** The guard read the void rule's terms once, before
each pair and before its residents were built, so a spike during a timed run was not caught: the A/A's two pairs at 0.8426 and
1.1594 sat next to a load spike to 11.7, and a pair could start at 3.4 GB and fall under 2 GB mid-run. The re-run reads the terms
after each arm's resident is built, just before and just after its timed run, and voids the pair if any of the four reads fails;
an unreadable load or memory reading fails the test (load from `sysctl` on macOS, `/proc/loadavg` on Linux). Otherwise as above:
the A/A control first (PASS iff its median is within 1 ± 0.02), then the A/B (SHIP iff median ≤ 0.98 with at least 9 of 10 faster).
The previous re-run's readings stay in the record; this one decides. Also: E2-G1's decode now starts after the 64-row prefill so
it attends over it, and the prefill comparison asserts non-empty logits on llama-tiny (the 1.5B's WebGPU prefill declines on
Metal). llama-tiny's widths reach only one stride of the widened loops end to end; the multi-stride loops are E2-G1 (a)'s cells.

**E2-G2 per-arm-guarded re-run read 2026-10-24 (14:15–14:26, the lock held exclusively): A/A PASS, A/B SHIP.** Every read of every
valid arm was under the line: the 1-minute load 2.65–3.84, reclaimable 3.46–4.55 GiB (Spotlight and `ANECompilerService` held the
load near 4 throughout). The A/A (narrow against narrow): median 1.0009, pairs 0.9972–1.0205, 2 void pairs (the load 4.02–5.39 at a
read) re-run. The A/B: narrow 24.30–24.67 ms a token, widened 21.41–21.85; **median 0.8846, 10 of 10 pairs faster**, no void pairs.
The gain holds: 0.8811 and 0.8846 on the two guarded readings. Log: `~/goinfer-logs/kse2-g2-rerun4-2026-10-24.log`.

**After the third review of #214 (15:45), registered before a further re-run.** The 1-minute load average is the wrong instrument
for a 3-second arm: the kernel updates it every 5 seconds as an exponential average, so the reads before and after an arm usually
see one sample, and a short spike reaches it damped about 12×. The guard now measures the void rule's load term directly:
`voidWindow` reads the machine's CPU ticks (`hostmem.ReadCPU`: Mach `HOST_CPU_LOAD_INFO` on macOS, measured fresh at four samples a
second, and `/proc/stat` on Linux) just before the timed decode starts and just after it stops. The arm is void if the busy cores
averaged over exactly that window exceed half the cores (4.0 here; the test's own process counts, as it does in a load average), or
reclaimable memory is under 2 GiB at either read. The 32-token prompt and the warm runs are outside the window. Also:
- The void cap is 10 pairs per phase, as first registered; the 15 in the last re-run was not amended, and is withdrawn.
- One `GOINFER_KSE2_TIMING=1` run does the A/A control (narrow against narrow; PASS iff its median is within 1 ± 0.02) and then the
  A/B in the same process. A failed control stops the run before the A/B. `GOINFER_KSE2_AA` is gone.
- Log lines name the arm that ran first in each pair.
- E2-G1 (a) runs 90 cells today (the row forms at all six n); the 11:50 read ran 63.
The bar is unchanged (SHIP iff the A/B median is ≤ 0.98 with at least 9 of 10 faster). This re-run decides; the earlier readings
stay in the record.

**E2-G2 read with the direct guard, 2026-10-24 (16:02–16:11, the lock held exclusively; 7.2 GB reclaimable at the start): A/A PASS,
A/B SHIP.** Every arm of both phases was valid: 0.23–1.96 busy cores averaged over its timed decode (the line is 4.0) and 2.02–4.32
GiB reclaimable at its ends; no void pairs. The A/A control: median 1.0002, pairs 0.9839–1.0112. The A/B: narrow 24.70–24.99 ms a
token, wide 21.70–22.02; **median wide/narrow 0.8824, 10 of 10 pairs faster** (11.8% less time a token). Over the three guarded
readings the A/B reads 0.8811, 0.8846 and 0.8824. Log: `~/goinfer-logs/kse2-g2-rerun5-2026-10-24.log`.

**The guard's CPU reading fixed, 2026-10-24 (17:05), registered before a re-run.** `hostmem.ReadCPU` read `host_statistics64(HOST_CPU_LOAD_INFO)`,
which XNU rate-limits for non-Apple binaries: after a few calls in a second it serves the last reply again, so an arm's end sample
could be up to a second stale (three calls in 0.4 s returned the same sample). It now reads `host_processor_info(PROCESSOR_CPU_LOAD_INFO)`,
per core, which is not rate-limited. Each state's count is differenced on its own, modulo 2^32 on macOS and clamped at 0 on Linux
(NOHZ can step idle back a tick), so busy never exceeds total. A sample pair spanning no interval now voids the arm instead of
failing the run, and counts toward the void cap. The memory term still reads `host_statistics64`, so it can be up to a second
stale; that is immaterial for a point reading against a 2 GiB line. The guarded readings above were taken on the old reading, so
the gate is re-run on the new one; that run decides, and the bar and the cap are unchanged.

**E2-G2 read on the fixed guard, 2026-10-24 (17:28–17:36, the lock held exclusively; 10.3 GB reclaimable at the start): A/A PASS,
A/B SHIP.** Every arm valid (0.15–2.41 busy cores, 2.75–4.77 GiB reclaimable); no void pairs. A/A median 1.0041; A/B **median
wide/narrow 0.8810, 10 of 10 pairs faster**. The four guarded A/B readings: 0.8811, 0.8846, 0.8824, 0.8810. Log:
`~/goinfer-logs/kse2-g2-fixedguard-2026-10-24.log`.

**KS-E3 · Encode token t+1 while the GPU runs token t (bit-identical).**
- **Where:** `gpu/decoderunner.go:DecodeRunner.Run`.
- **Waste:** recording starts only after the previous token's Poll, so about 0.62 ms of CPU encode is exposed per token.
- **Change:** record after the submit. Drop the pending buffer on `SetAdapter`, `UseKVSlot` or a mode switch.
- **Gain (est.):** ~0.6 ms (~8%). Metal's version measured +4.4%.
- **Size:** S–M.

**KS-E3 gates, registered 2026-10-24 before the code (E3-G), on the owner's order of the Mac work.** The build: after `Run`
submits token t, the runner records token t+1's command buffer (its steps, then the logits copy) while the GPU runs, and the next
`Run` writes its inputs and submits that buffer instead of recording one. `record` replays `r.steps`, whose pipelines, bind groups
and grids are fixed for the runner; the inputs move through buffer writes, which the queue orders before the submit. The pending
buffer is dropped wherever the steps or the recorded shape change: `SetAdapter` (which `UseKVSlot` also calls), `RunNoLogits`
and `Close`. `encodeAheadOffForTest`, read at each `Run`, records in line as today.

| gate | tier | instrument | PASS iff |
|---|---|---|---|
| E3-G1 identity | 0 | llama-tiny and the 1.5B's `.gguf` at int8int8 on WebGPU, encode-ahead on against off on one resident: 32 greedy `Forward` steps; then a sequence mixing `Forward`, `RunNoLogits` (through `PrefillLast`'s trunk-only rows where the backend takes it), `ForwardN` and a second KV slot (`UseKVSlot`) | every logit byte-equal |
| E3-G2 speed | timed, Mac, timing lock | a whole-token in-process A/B on the 1.5B on WebGPU: encode-ahead on against off on one resident, 128 greedy decode tokens after a 32-token prompt, A B B A over 10 pairs after 2 warm runs each | SHIP iff the median on/off per-token time is ≤ 0.98 with at least 9 of 10 pairs faster; under it, the owner's rule for a bit-identical change decides; otherwise PARK |

**E3-G2's bar amended 2026-10-25 by Francis (his word of 6:10 am, relayed by the coordinator).** E3-G2 now encodes the owner's rule
for a bit-identical change: **PASS iff at least 9 of 10 pairs are faster.** The median on/off is still printed on every run, as
information, so a shrinking gain stays visible; it no longer decides. The reason: the registered median bar (≤ 0.98) handed its
near-miss case to the owner's rule, and the test did not encode that rule. The quiet 23:32 reading (0.9916, 10 of 10) failed the
test while meeting the rule Francis had kept KS-E3 on. `TestKSE3_decodeSpeed` fails only below 9 of 10. The void guard is
unchanged. Every recorded reading keeps its verdict on the bar it was read under.

E3-G3: none (one command buffer held per runner). Planted defect, run once and reverted: `SetAdapter` not dropping the pending
buffer, with a slot switch in E3-G1's sequence; or, if that stays equal, the pending buffer submitted after its runner's steps
were rebuilt must go red. Stopping rule: one pass each. Cost: G1 a minute; G2 about 4 minutes.

**Amended 2026-10-24, before any E3-G run.** `UseKVSlot` calls `SetAdapter` only when an adapter is bound, and no WebGPU adapter
fixture is at hand, so the registered plant cannot fire. The plant is instead a pending buffer recorded without the logits copy
(substituted after each `Run`), which E3-G1 must turn red. The instruments are `TestKSE3_encodeAheadIdentity` and
`TestKSE3_decodeSpeed`; the bars are unchanged.

**KS-E3 read 2026-10-24 (Mac, 12:05–12:13): SHIP.**
- **E3-G1 PASS:** llama-tiny and the 1.5B at int8int8 on WebGPU with two KV slots, 24 logit rows each (8 `Forward`, a 4-row
  `ForwardN`, 2 `ForwardNoLogits`, 4 `Forward`, 4 on slot 1, 4 back on slot 0), 0 logits differ, encode-ahead on against off. The
  plant (a pending buffer without the logits copy) turned llama-tiny red, 4,352 logits.
- **E3-G2 SHIP** (12:11–12:13, the lock held exclusively; the 1-minute load 2.02 at the start, each pair's 1.28–1.59; reclaimable
  9.9 GB, available 13.1 GB): off 24.35–24.73 ms a token, on 23.81–24.21, **median on/off 0.9796, 10 of 10 pairs faster**, about
  0.5 ms a token (the survey's estimate ~0.6). Read on main's build, without KS-E2.
Logs: `~/goinfer-logs/kse3-g1-2026-10-24.log`, `~/goinfer-logs/kse3-g2-2026-10-24.log`.

**After the second-opinion review of #216 (13:40), registered before the re-run.** The earlier amendment was wrong that no adapter
fixture exists: `buildLlamaTinyLoRAFixtureGPU` makes one, so the registered plant now runs. Changes:
- `Run` requests the staging map before it records the next token, and that recording is timed apart (`TEncodeAhead`) and left out
  of `TSync`, so `TEncode`/`TSync` keep their meaning (`TestDecodeTWE_split` passes).
- E3-G1 uses a fresh resident per arm and its own inputs on slot 1. Added: an adapter bound after a pre-recorded token
  (`TestKSE3_setAdapterDropsPending`; the registered plant, `SetAdapter` keeping the stale buffer, turned all 1,024 logits red) and
  each drop path's state (`TestKSE3_pendingDroppedWhereRecorded`: `RunNoLogits`, `SetAdapter`, release). The drop moved into
  `release`, so a Context teardown frees it too. All PASS; the identity plant is still red (4,352 logits).
- **E3-G2 re-run:** the same instrument under the lock, the test refusing to run without the lock's token or under 6 GB
  reclaimable before its load, voiding and re-running a pair started above a load of 4.0 or under 2 GB reclaimable, and failing on
  the bar (SHIP iff median on/off ≤ 0.98 with at least 9 of 10 faster). The first reading (0.9796) was taken with the next token
  recorded before the staging map was requested; the re-run reads the corrected order and decides.

**E3-G2 re-run read 2026-10-24 (13:42–13:45, under the lock; "== start 13:42:07 reclaimable_mb 9161"): SHIP.** off 24.46–25.95 ms a token, on
23.75–24.88; **median on/off 0.9764, 10 of 10 pairs faster**, 2 void starts (a load spike to 6.5) re-run; each valid pair started at
a load of 2.11–3.74 and 2.5–4.0 GB reclaimable. Log: `~/goinfer-logs/kse3-g2-rerun-2026-10-24.log`.

**After the second review of #216 (14:20), registered before a further re-run.** The guard read the void rule's terms once, at each
pair's start, so a load spike or a memory drop during a timed run was not caught. The re-run reads them just before and just after
each arm's timed run (after the resident is built and warmed), and voids the pair if any of the four reads fails, up to 15 times; an
unreadable load or memory reading fails the test (`sysctl` on macOS, `/proc/loadavg` on Linux). The bar is unchanged (SHIP iff the
median on/off ≤ 0.98 with at least 9 of 10 faster); the previous readings stay in the record and this one decides. Also:
- `kse3KeepPendingForTest` is gone from `SetAdapter`. The registered plant now lives in `TestKSE3_setAdapterDropsPending`: it
  detaches the pending buffer before `SetAdapter` and hands it back after, which is what a `SetAdapter` that kept it would leave.
- That test fails, not skips, on a load error once the hardware is gated; the identity test's plant runs only on
  `GOINFER_KSE3_PLANT=1` and the adapter test's only on `=2`.
- Added `TestKSE3_kvSlotWithAdapter`: llama-tiny with two KV slots and an adapter bound, 4 steps on slot 0, 4 on slot 1 with its own
  inputs, 4 more on slot 0; every logit with encode-ahead on equals off.

All four E3-G1 tests PASS (0 logits differ); `GOINFER_KSE3_PLANT=1` reds only the identity test (4,352 logits), `=2` only the adapter
test (1,024 of 1,024).

**E3-G2 per-arm-guarded re-run read 2026-10-24 (14:32–14:34, the lock held exclusively; "== start 14:32:16", 10.2 GB reclaimable
before the load): SHIP.** Every read of every arm under the line: the 1-minute load 1.06–1.77, reclaimable 3.44–4.03 GiB. off
24.68–24.95 ms a token, on 24.17–24.56; **median on/off 0.9788, 10 of 10 pairs faster**, no void pairs. The gain holds: 0.9764 and
0.9788 on the two guarded readings. Log: `~/goinfer-logs/kse3-g2-rerun2-2026-10-24.log`.

**After #214's third review (15:45), registered before a further re-run.** The same defect is in E3-G2's guard: the 1-minute load
average updates every 5 seconds as an exponential average, so the reads around a 3-second arm usually see one sample and damp a
short spike about 12×. E3-G2 now uses #214's `voidWindow`: the machine's busy cores averaged over exactly the timed decode
(`hostmem.ReadCPU`, Mach `HOST_CPU_LOAD_INFO` on macOS and `/proc/stat` on Linux, read just before the timer starts and just after
it stops, the 32-token prompt outside), void over half the cores, and reclaimable memory at both ends, void under 2 GiB. The void
cap stays 15, as registered for E3 at 14:20, and the bar is unchanged. This reading decides; the earlier ones stay in the record.

**E3-G2 read with the direct guard, 2026-10-24 (16:11–16:13, the lock held exclusively; 9.4 GB reclaimable at the start): under its
bar, PARK.** Every arm was valid by the void rule: 0.83–1.57 busy cores over its timed decode and 3.47–3.92 GiB reclaimable; no void
pairs. But the arms swung far more than in any earlier reading: off 25.16–29.22 ms a token, on 24.74–28.26 (the 13:42 and 14:32
readings held each arm within about 1.5 ms). **Median on/off 0.9825, 8 of 10 pairs faster** (pairs 0.9271–1.0726). That misses the
bar (≤ 0.98 with 9 of 10) and the owner's rule for a bit-identical change (9 of 10 faster), so by the registration KS-E3 parks. The
earlier guarded readings, 0.9764 and 0.9788 with 10 of 10, stay in the record and do not overturn the deciding one. The swing is
not in the CPU figure the guard reads; its source (another app's GPU work is one candidate) is not identified. Log:
`~/goinfer-logs/kse3-g2-rerun3-2026-10-24.log`.

**E3-G2 with a GPU-contention term, registered 2026-10-24 (16:50) before the run, on Francis's choice (c) of the three put to him
(revert, keep, or re-run with a GPU-contention check).** `otherGPUWindow` (`gpu/gpuothers_test.go`) reads, at both ends of each
arm's timed decode, the GPU time every other process has accumulated: on macOS the IORegistry's `AGXDeviceUserClient` entries,
`AppUsage` `accumulatedGPUTime` in nanoseconds, the figure Activity Monitor's GPU column reads. **An arm is void if other
processes' GPU time exceeds 2% of its wall time**: the size of the effect under test, so contention of that size could produce
it alone. The CPU and memory terms (`voidWindow`) and the void cap (15) are unchanged, and the term is required on macOS.
- **The instrument is checked.** `TestOtherGPUWindow_seesAnotherProcess`: a child process decoding llama-tiny on WebGPU reads
  58.5% of a 3-second window. With nothing of ours running, 3-second windows read at most 0.4% (WindowServer).
- **Its limit:** a process that starts and exits inside one arm takes its registry entry, and its GPU time, with it, so a GPU job
  shorter than an arm can pass unseen. A contender running across the arm, such as another session's test or a system service, is
  seen.
- **The bar is unchanged:** SHIP iff the median on/off is ≤ 0.98 with at least 9 of 10 pairs faster. On SHIP, KS-E3 stays on main.
  Under it, KS-E3 parks and its revert from main is put to Francis. This reading decides; the four earlier ones stay in the record.
- **Amended 17:05, before the run, after the review of #214, #224 and #227:**  `hostmem.ReadCPU` read `host_statistics64(HOST_CPU_LOAD_INFO)`,
which XNU rate-limits for non-Apple binaries: after a few calls in a second it serves the last reply again, so an arm's end sample
could be up to a second stale (three calls in 0.4 s returned the same sample). It now reads `host_processor_info(PROCESSOR_CPU_LOAD_INFO)`,
per core, which is not rate-limited. Each state's count is differenced on its own, modulo 2^32 on macOS and clamped at 0 on Linux
(NOHZ can step idle back a tick), so busy never exceeds total. A sample pair spanning no interval now voids the arm instead of
failing the run, and counts toward the void cap. The memory term still reads `host_statistics64`, so it can be up to a second
stale; that is immaterial for a point reading against a 2 GiB line. The guarded readings above were taken on the old reading, so
this run reads the new one, alongside the GPU term.

**E3-G2 read with the GPU-contention term and the fixed CPU reading, 2026-10-24 (17:26–17:28, the lock held exclusively; 6.6 GB
reclaimable at the start).** Every arm was valid: other processes' GPU time 0.00–0.98% of each arm (the line is 2%), 0.17–1.41
busy cores, 2.81–4.33 GiB reclaimable; no void pairs. The arms held steady this time: off 24.59–24.89 ms a token, on 23.95–24.50.
**Median on/off 0.9807, 10 of 10 pairs faster** (0.9738–0.9899): 0.0007 over the median bar.
- **E3-G2's registered bar (above):** under the median bar, the owner's rule for a bit-identical change decides. Its terms are met:
  - bit-identical: E3-G1;
  - no memory is allocated (one encoded command buffer is held per runner);
  - faster on 10 of 10 pairs;
  - it changes the one decode path rather than adding a selectable second one (the off switch is a test hook).
  By that clause it ships.
- **The 16:50 registration** restated the bar as "under it, KS-E3 parks and its revert is put to Francis" and left the clause out.
  The two registrations conflict on exactly this case, so the call is Francis's.
- **Readings so far:** 0.9796, 0.9764 and 0.9788 at 10 of 10 (the load-average guard); 0.9825 at 8 of 10 (16:11, busy-core guard;
  the arms swung 25–29 ms); 0.9807 at 10 of 10 (this run, GPU term, steady arms).
Log: `~/goinfer-logs/kse3-g2-gpuguard-2026-10-24.log`.

**Decided 2026-10-24 by Francis: keep.** Asked to keep KS-E3 on main or revert it after the 17:26 reading, he answered "Keep". KS-E3
stays on main as shipped in #216, on the owner's bit-identical rule: 10 of 10 pairs faster, bit-identical, no memory allocated, one
path. The median (0.9807) is 0.0007 over the median bar.

**After the review of #227 (registered 22:25, before a re-read).** Three defects in the GPU term, all fixed:
- The reading counted as taken whenever ioreg ran, so a parse that matched nothing read 0.00% and passed. It now requires this
  process's own GPU client in the listing (a test with a device open owns one).
- It summed GPU time per process, so a client closing inside an arm could cancel another client's time and under-read contention.
  It now keys each client by its registry entry ID; a closed client counts nothing and cancels nothing.
- The ioreg call ran inside the arm's CPU window. The GPU reading is now taken outside it, before the CPU window opens and after it
  closes.
`TestParseAGXClients` checks the first two on synthetic ioreg text (dropping the own-client check turns it red). The 17:26 run
cannot show whether the per-process sum hid contention in a pair it counted valid, so E3-G2 is read once more on the fixed guard.
KS-E3 stays on main whatever it reads (Francis's decision); the reading goes in the record.
The first attempt (22:25) refused before loading: 4.4 GB reclaimable, under the test's 6 GB pre-load line, with the owner's apps
open. It is queued on the Mac's night runner (`kse3-g2-gpuguard`, the pinned binary in `~/goinfer-bench/kse3-night/`, re-queued
while under 6.5 GB reclaimable until 06:00); its reading is added here when it lands.

**E3-G2 read on the fixed GPU guard, 2026-10-24.**
- **First night attempt (22:31–22:41): no reading.** All 16 pairs were void: reclaimable memory fell from 3.6 to 1.8 GiB, and other
  processes held about 5% of the GPU in every arm. A sample at 23:31 put 5.0% on a Claude app helper process.
- **Second attempt, queued on Francis's word of 23:30 with only the Claude app open (23:32–23:34, the night runner holding the
  lock; 8.4 GB reclaimable at the start).** Every arm was valid, the quietest of any E3-G2 run: 0.20–0.34 busy cores, other
  processes' GPU 0.44–0.59% (the line is 2%), 3.85–4.01 GiB reclaimable; no void pairs. off 25.97–26.13 ms a token, on
  25.71–25.85. **Median on/off 0.9916, 10 of 10 pairs faster** (0.9846–0.9939).
- **Verdict.** It misses the median bar (≤ 0.98). The owner's bit-identical rule's terms are met again: 10 of 10 faster,
  bit-identical, no memory allocated, one path. KS-E3 stays on main by Francis's decision either way.
- **On a quiet machine the gain is about 0.2 ms a token (0.8%)**, smaller than the 0.5–0.6 ms the noisier readings showed.
Log: `~/goinfer-logs/night/runs/2026-10-24/kse3-g2-gpuguard-2.log` (the first attempt's beside it, `kse3-g2-gpuguard.log`).

**After #227's re-review (2026-10-25, 00:10): the GPU term's parse hardened; no re-run.**
- The own-client check accepted this process's client even when no GPU time parsed from it, so a renamed key or a changed format
  would have read every client as 0 ns, a silent pass. It now needs this process's client with a GPU time that parses.
  `TestParseAGXClients` adds both cases (no GPU time, a changed format); without the check it is red.
- The instrument check now attributes GPU time by process. The child's own clients must reach 5% of the window: they read 38.3%,
  so another app's load cannot pass it. It also waits for the child's GPU work to show before the window opens, keeps the parent's
  client open, and kills the child at cleanup.
- The window's wall time now includes the end sample.
The 23:32 reading stands: this process's client had GPU time throughout (the run had decoded before every window), and every other
client's time parsed (0.44–0.59%, not 0).
- `TestKSE3_decodeSpeed` still fails on the registered median bar (≤ 0.98), which the accepted quiet reading (0.9916) misses. The
  registration hands that case to the owner's rule, and the test does not encode it. Changing what the test asserts is a bar change
  and Francis's call.

**After #227's fourth review (registered 2026-10-25, committed 00:43:42, before the re-read started at 00:44:51).** A GPU client present at an arm's start and gone at
its end took its accumulated time with it. A contender that saturated the GPU for most of an arm and closed its device before the
end sample read as 0 ns.
- **The fix.** `otherGPUDelta` now reports such clients, and the arm is void: what it did is unknown. In 20 three-second windows
  sampled on the Mac at 00:41, no client vanished, so the void costs little. `TestParseAGXClients` covers it, and is red without
  the check.
- **What can still pass unseen:** a GPU client opened and closed entirely between an arm's two samples. Every client present at
  either sample is accounted for: its time if it is present at both, all of its time if it is new at the end, and the arm is void if
  it is gone at the end.
- **The instrument check** now reads through `otherGPUWindow` itself, decides its virtualised-GPU skip with its own GPU work
  running, and its child runs until it is killed.
- **Re-read registered.** The 23:32 reading was taken without the vanished-client check, so whether a client closed inside one of
  its arms is not known. E3-G2 is read again on the night runner with the same bar. KS-E3 stays on main either way (Francis's
  decision); the reading goes in the record.

**E3-G2 re-read with the vanished-client void, 2026-10-25 (00:44–00:46, the night runner holding the lock; 8.96 GB reclaimable at
the start; only the Claude app open): PASS, 10 of 10.**
- **Every arm valid:** 0.43–0.83 busy cores, other processes' GPU 0.00–0.71%, no client vanished, 3.97–4.21 GiB reclaimable; no
  void pairs.
- **Arms:** off 29.90–31.53 ms a token, on 27.51–28.69. **Median on/off 0.9166, 10 of 10 pairs faster** (0.8899–0.9456). That is
  under the median bar (≤ 0.98), so E3-G2 passes on its registered bar.
- **The size is not settled.** Both arms ran about 15% slower than at 23:32 (off 26.0, on 25.8 ms), with no thermal or low-power
  flag recorded, and the ratio moved with them, 0.9916 then and 0.9166 now. A ratio across sessions is not a ratio
  (`docs/measurements/noise-registry.md`), so each reading stands for its own session. The direction holds in every reading,
  faster on 10 of 10 in all but the swinging 16:11 one, and the size runs from about 0.8% to about 8% depending on the machine's
  state.
- KS-E3 stays on main (Francis's decision).
Log: `~/goinfer-logs/night/runs/2026-10-25/kse3-g2-gpuguard-3.log`.

**After #227's fifth review (2026-10-25), test-only.** The GPU-time parse ignored `strconv` errors, so an overflowing time read as
MaxInt64 and an unparsable pid as 0. A value that does not parse now makes the whole reading unknown: the arm cannot be judged, and
E3-G2 fails on macOS rather than reading it as quiet. `TestParseAGXClients` adds overflow cases, and is red without the check. The
instrument check (`TestOtherGPUWindow_seesAnotherProcess`) also no longer fails when an unrelated process closes a GPU client in its
window: it requires only the child's own clients to stay listed. E3-G2's own rule, voiding an arm with any vanished client, is
unchanged. The 00:44 reading stands: its log shows every client parsed (other processes 0.00–0.71%) and none vanished.

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
- **Decided (Francis, 2026-10-25):** the bit-identical step goes ahead, its gates registered first.
  - **Scope (10:35):** default on for gpt-oss only. `attendQueryP` keeps today's schedule for its families until they are timed (KS-F2b).
  - KS-O4 is approved gated: its fidelity gate is registered before any code, and Step 2 (memory) follows it.
  - mac-kapitsa; one PR for the bit-identical step, another for KS-O4.
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

**KH1-G5 amended 2026-10-23 (owner's order of 11:50: the night queue, case (a)).** The branch merged (a570f8df), so "main against
the branch" is `serve` built at 421a4347 (main just before the merge) against `serve` at a570f8df, one cold cell each in that order,
`mem0_darwin.py`'s new `metal:qwen35-9b` cell (the Metal `.giw`, `-ctx 32768`, the same 20K preamble). Its by-day fit check refuses
beside the owner's apps (the 9B's 5.96 GB mapping), so it runs at night through `docs/measurements/ks-night-2026-10/run-kh1-g5.sh`,
which exits 75 on a refusal. Still read, not gated: each cell's peak footprint and RSS during the request against its footprint
after load, the survey's estimate 4.8 → about 0.5 GiB of transient prefill scratch. A cell the server could not come up for is void.

**KH1-G5 read 2026-10-24 (08:56–09:20, by day on the owner's word; the timing lock held exclusively; not timed).** It ran from
`run-kh1-g5-day.sh`: both `serve` builds rebuilt at their commits with #203's memory reading applied, so the fit guard passed with
no bypass. Qwen3.5-9B's Metal `.giw`, one cold 20,443-token prompt per fresh server:

| build | footprint after load | peak | end | transient (peak − end) | request |
|---|---|---|---|---|---|
| 421a4347 (main before KS-H1) | 242 MiB | 6,245 MiB | 1,268 MiB | **4,977 MiB** | 714.6 s |
| a570f8df (KS-H1 merged) | 242 MiB | 1,866 MiB | 1,275 MiB | **591 MiB** | 695.4 s |

**4.86 → 0.58 GiB of transient prefill scratch; peak footprint −70%**, as the survey estimated (4.8 → about 0.5). System swap rose
464 → 1,826 MB during the first cell and was 1,666 MB after the second; the request times are context, not a timed comparison.
Logs: `~/goinfer-logs/kh1-g5-day-2026-10-24-0856/`.

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

**KS-H2 / KS-H3 readings, registered 2026-10-25 before any data (owner, 1:52 pm: measurement only, no guard changes; the numbers go to
nobara-ks-cuda for KS-H6).** Nobara, RTX 2070 SUPER 8 GB, driver 595.91.07, `main` at `d759bba19`; each run alone on the card under
the exclusive timing lock, with the desktop compositor's context (about 16 MiB) recorded if present. Harness:
`docs/measurements/ks-h2-h3-2026-10-25/harness_test.go.txt` (a scratch test, kept as text), driven by the existing
`allocSeenForTest` and `kvAllocSeenForTest` seams, which fire on the executor thread before each allocation; it reads
`cuMemGetInfo` through the resident's own context inside them.
- **H2-R1, the 7B's build** (`qwen2.5-7b-instruct-q4_k_m.gguf`, int4, one slot, its default context): free VRAM at the load's
  first allocation, at its first K/V pool allocation, and after the load returns. Reported: the drop from the first to the
  second reading against the bytes requested in between (the weights and the build's scratch), and against those requests
  rounded as `cuMemAlloc` rounds (2 MiB granularity at 2 MiB and over). Expectation from G-M1: about 406 MiB of rounding.
- **H2-R2, Qwen3-8B's KV pools** (`Qwen3-8B-Q4_K_M.gguf`, int4, one slot, `ResidentContext` 16,384, `--kv` at its `auto`): free
  VRAM just before the first pool buffer and just after the last, against the pool bytes requested (`kvAllocSeenForTest`) and
  the same requests rounded. Expectation from the survey: about 135 MiB of rounding not in the price.
- **H3-R1, H3-R2, `TestKernelLocalMemoryCensus` and `TestMoERouteFirstLaunchReservation` as they stand.** `moe_route` and
  `route_gptoss` still keep their per-thread arrays (`cuda/moe.cu:moe_route`'s `score[MOE_MAX_E]`), so these are the readings
  before the move, the baseline the move's own gate compares against; the move itself is KS-H3's code change, not this one.
  Expectation: `moe_route` 4,416 B per thread; 138,412,032 B retained after its first launch.
- **Void and stopping rule:** a reading taken while another process holds a CUDA context beyond the compositor is void and re-run;
  one reading each, since a memory reading on an idle card does not vary. Cost: about 5 minutes in all.

**KS-H2 / KS-H3 readings, read 2026-10-25, 14:28–14:41, by day under the exclusive lock.** Logs and harness:
`docs/measurements/ks-h2-h3-2026-10-25/`. Three attempts, all kept:
- **Attempt 1 (14:28, `readings-1-h2void-h3ok.log`):** H2-R1 and H2-R2 void. Another process's serve held 6.9 GB on the card, left
  over after its harness released the lock (since fixed on that harness's branch). H3-R1 and H3-R2 valid: the card held only the
  compositor by then.
- **Attempt 2 (14:32, `readings-2-h2.log`), a deviation:** both builds came up on the f32 cache. The cuda test suite's default
  turns `--kv auto`'s f16 off (`cuda/kv16_default_test.go`), so the harness did not resolve the KV as serve does. That is right for
  H2-R1's weight stage, which does not depend on the KV, and wrong for the registration's `auto`. Qwen3-8B at 16,384 was refused on
  f32 (4.83 GB of KV against 2.98 GB free).
- **Attempt 3 (14:40, `readings-3-h2-served.log`):** `KSH2_SERVED=1` turns `kvF16Auto` on for the build, so both builds resolve f16 as
  serve does. Only the compositor (17 MiB) was on the card. These are the H2 readings.

| reading | stage | requested | rounded to 2 MiB | the card's drop | rounding (rounded − requested) | drop − rounded |
|---|---|---|---|---|---|---|
| H2-R1, qwen2.5-7b q4_k_m, 16,384, f16 | weights and scratch (572 requests, 347 under 2 MiB) | 4,023.8 MiB | 4,429.6 MiB | 4,444.0 MiB | **405.8 MiB** | 14.4 MiB |
| H2-R1 | K/V pools (72 buffers) | 900.1 MiB | 1,008.6 MiB | 1,010.0 MiB | **108.5 MiB** | 1.4 MiB |
| H2-R2, Qwen3-8B q4_k_m, 16,384, f16 | weights and scratch (696 requests, 334 under 2 MiB) | 4,323.3 MiB | 4,431.8 MiB | 4,432.0 MiB | **108.5 MiB** | 0.2 MiB |
| H2-R2 | K/V pools (88 buffers) | 2,313.6 MiB | 2,448.6 MiB | 2,452.0 MiB | **135.0 MiB** | 3.4 MiB |

- **Free VRAM** (of 7,784.2 MiB; 7,277.1 free before each load):
  - H2-R1: 1,821.1 MiB after `kvAllocPools`, 1,691.1 MiB after the load returned.
  - H2-R2: 391.1 MiB after the pools, 261.1 MiB after the load.
  - 70 allocations follow the pools in each build, about 130 MiB.
- **Prices:** `kvBytesForCap` 896.0 MiB + a 3.5 MiB null slot (H2-R1), 2,304.0 + 9.0 MiB (H2-R2). Both are the requested pool bytes;
  the rounding is not in them.
- **Against the registration:** H2-R1's 405.8 MiB meets the expected 406. H2-R2's 135.0 MiB meets the expected 135. Rounding at 2 MiB
  accounts for the card's drop to within 14.4 MiB at every stage.
- **The f32-cache reading of attempt 2**, for the record: the 7B's pools requested 1,799.7 MiB, rounded 1,904.7 (105.0 of
  rounding), and the card dropped 1,906.0.
- **KS-H3, before the move** (`moe_route` and `route_gptoss` still keep their per-thread arrays on `main`):
  - `moe_route` declares 4,416 B per thread. Its naive full-occupancy product is 180,879,360 B (40 SMs × 1,024 threads), but the
    MEASURED retention after its first launch is **138,412,032 B (132.0 MiB)**, ratio 0.7652; the module load and the first other
    launch cost 0 B. Both meet the expectation.
  - `route_gptoss` declares 4,608 B per thread (180.0 MiB naive), and 21 other kernels declare 32–96 B.
  - The census's upper bound, every backing store at once, is 383.8 MiB.
  - The readings after the move belong to the move's own gate.
- **Handed to nobara-ks-cuda for KS-H6.** No guard changed here.

**KS-H4 · Metal towers keep their largest image's scratch until unload (bit-identical).**
- **Where:** `metal/grid_vision.go:gridVAccel.grow` only grows; the same holds in the other tower accelerators and aikit's
  `qwenmetal` / `visionmetal`.
- **Waste:** MEM2-G4: Qwen3-VL-2B holds 2,622.5 MiB after a 4K forward until `Close`.
- **Change:** release above a ceiling after the forward, or MEM2's optional patch ceiling.
- **Gain:** up to 2.6 GiB back on the 16 GB Mac.
- **Size:** S.

**KS-H4 gates, registered 2026-10-23 (05:30) before the code (H4-G), on the owner's plan of 5:16 am.** The first measurement:
Qwen3-VL-2B's grid tower holds its 4K scratch until `Close` (2,622.5 MiB on main, 1,515.4 after KS-D6), unchanged 10 s after the
forward. The build: after a forward whose scratch is over a ceiling (256 MiB), the grid tower releases its scratch once it has
been idle for `gridIdleRelease` (5 s). A forward in the meantime cancels the release and reuses the buffers, so a repeat image
call pays nothing. The next call after a release allocates again. Scope: `gridVAccel`, the measured tower. The other tower
accelerators and aikit's `qwenmetal`/`visionmetal` are left for their own rows.

| gate | tier | instrument | PASS iff |
|---|---|---|---|
| H4-G1 identity | 0 | the grid tiny fixtures (qwen3.5, glm-ocr, qwen2.5-vl), with the ceiling at 0 and the idle wait at 0: a forward, the release, a forward of a larger then a smaller grid; against the tower with release off | every output byte-equal; the scratch reads 0 bytes after each release |
| H4-G2 allocated after 10 s | 1, Mac, memory read, timing lock shared | `TestKSD6H4_gridTowerScratch` on Qwen3-VL-2B at 4K: Metal's allocated size net of the build right after the forward and 10 s later | 10 s later at most 16 MiB, from 1,515.4 right after |
| H4-G3 repeat call | timed, Mac, timing lock | Qwen3-VL-2B at 4K, the wall time of a repeat `Hidden` issued right after the previous one, release on against release off, two towers in one process, A B B A over 5 pairs after a warm run each; the first call after a release read, not graded | SHIP iff the median on/off ratio is ≤ 1.02 (no regression); otherwise PARK and decide |

Planted defect: the release keeps `cap` (so the next forward skips the allocation and writes into released buffers). H4-G1 goes
red. Stopping rule: one pass each. Cost: seconds; H4-G2 about 2 minutes; H4-G3 about 1 minute.

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
- **Decided (Francis, 2026-10-25, KS-O3): A.** After KS-H2 and KS-H3 read, the two margins are re-derived from measured parts,
  the 2,048-row pass's scratch included.
  - The numbers come to Francis before either guard changes.
  - nobara-ks-cuda.
- The margins can be rebuilt from measured parts after KS-H2 and KS-H3.
- **First measurement:** minimum free VRAM over a 20k prompt on Qwen3-8B at 16k and on the 26B.

**KS-H6 reading, registered 2026-10-25 before it runs (H6-R).** A reading for the owner's decision, not a gate: no guard or
margin changes in this work, and it stops with the numbers.
- **What each margin must cover:**
  - `ctxCapMarginBytes` is left free beside the weights and KV at the load-time fit check. It must absorb what the build allocates
    after the KV pools, the allocation floor, the one prefill pass that is not priced beyond it, and decode's transients.
  - `slotMarginBytes` is left free when the 26B's expert cache is sized, after everything else. It must absorb the floor and the
    transients that come after the slots.
- **The parts and where each comes from:**
  - **The allocation floor:** cited, not re-measured. `TestAllocFloor` drains the device, and it skips while the desktop compositor
    holds a context. Its record (`docs/code-notes/cuda.md`, "TestAllocFloor") puts the refusals at 148–155 MiB free at any request
    size.
  - **The build after the pools:** kv4b-grade's KS-H2 readings (#251): 70 allocations, about 130 MiB, on the 7B and on Qwen3-8B at
    16k.
  - **The moe_route reservation:** KS-H3's baseline, 132.0 MiB retained after its first launch. The 26B's build forces it before it
    reads free VRAM for the slots.
  - **The prefill pass inside the margin:** `PrefillScratchBytes(512)` plus `PrefillSpanScratchBytes(512, context)` per model. Under
    KS-B8 (#250) a wider pass is taken only from VRAM beyond the margin.
  - **The run's transients (new, `TestKSH6_marginParts`):** each model loaded as serve loads it (`--kv auto`, f16 where cleared;
    the 26B at int4 with its expert cache). The test records the free VRAM after the load. It then prefills a long prompt and
    decodes 64 tokens while a child `nvidia-smi --query-gpu=memory.used -lms 20` samples the card. The prompts are 15,000 tokens on
    Qwen3-8B at context 16,384, and on the 26B its context less 512. The reading is the minimum free VRAM over the run, its drop
    below free-after-load, and the pass width the run used, including any learned narrowing.
- **The derivation:** for each margin, the sum of its parts against 384 MiB, per model. The owner gets the measured headroom left at
  the run's minimum: free VRAM above the floor, or a refusal.
- **Out:** DECISION NEEDED with the numbers and the options (keep 384; a different constant; a margin computed from the parts at
  load). Nothing is changed here.
- **Cost:** two loads. Qwen3-8B about 2 min (a 15,000-token prefill and 64 tokens). The 26B about 5 min (its load from
  safetensors and the prompt). By day, with the timing lock shared: memory, not time, is read.

**H6-R read 2026-10-25, by day** (`night.py`, the lock exclusive; driver 595.91.07; only the desktop compositor's 17 MiB on the card
before each load). Logs: `docs/measurements/ks-h6-2026-10/`.
- **The first run (82a14f63a, 16:23)** read Qwen3-8B. The 26B was refused because the test had not set serve's
  `-moe-cache-experts`, which the registration names. **The second run (e790fc51a, 18:25)** read the 26B with the cache.

| | Qwen3-8B q4_k_m, context 16,384, f16 KV | Gemma 4 26B-A4B, int4, expert cache (30 slots), context 4,096 |
|---|---:|---:|
| the margin that binds | `ctxCapMarginBytes`, 384 MiB | `slotMarginBytes`, 384 MiB |
| free after the load (cuMemGetInfo) | 261.1 MiB | 385.1 MiB |
| the prompt | 15,000 tokens, passes of 512 rows, no narrowing | 3,584 tokens, passes of 512 rows, no narrowing |
| a 512-row pass's scratch | 112.0 MiB | 77.8 MiB |
| the run's peak rise above the post-load figure (nvidia-smi, every 20 ms) | 112 MiB (961 samples) | 142 MiB (1,950 samples) |
| the run's minimum free | **≈ 149.1 MiB** | **≈ 243.1 MiB** |
| above the allocation floor (148–155 MiB) | **about 0** | about 90 MiB |

- **`ctxCapMarginBytes` on Qwen3-8B at 16k: used up.**
  - The margin's 384 MiB goes to about 130 MiB of build allocations after the KV pools (KS-H2, #251), so 261 MiB is left after the
    load.
  - The 512-row pass then takes 112 MiB, which leaves about 149 MiB: the allocation floor. Decode added nothing above the pass.
  - The parts sum to about 130 + 112 + 150 = 392 MiB against 384. The run fit with no headroom: any more scratch would have refused
    an allocation, and the prefill would have narrowed its passes.
- **`slotMarginBytes` on the 26B: about 90 MiB to spare.**
  - The cache is sized last, so the 385 MiB free after the load is the margin itself. The `moe_route` reservation (KS-H3, 132 MiB)
    is taken before that figure is read.
  - The run took 142 MiB: the 78 MiB pass and about 64 MiB of MoE and decode transients. About 243 MiB stayed free, some 90 MiB
    over the floor.
  - Each expert slot costs about 106 MiB on this load (30 slots, 3.1 GB), so the spare is under one slot.
- **Not covered:**
  - The 26B at a long context: its build's default is 4,096 positions, so a 20k prompt (the survey's first measurement) is not
    possible without raising `-ctx`, which takes slots away.
  - KS-H3's after-move reading, which would give back its 132 MiB.
- **The instrument after the run:** `TestKSH6_marginParts` is split into `TestKSH6_marginPartsQwen3` and `TestKSH6_marginParts26B`,
  with the same body, because the heavy tier runs one large checkpoint per process (`cmd/gate/heavy_groups.go`). Qwen3-8B also gets
  its row there.
- **Seen in passing, not part of the reading:** beside the 26B, every tiny self-test probe resident declined for room ("0.40 GB is
  free on the device beside the weights (plus 384 MB reserved...)"). The 26B's own resident was unaffected.


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
