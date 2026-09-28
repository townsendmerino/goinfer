#!/usr/bin/env python3
"""TE5(b) calibration: does an in-process A/B predict the served A/B for the same change?

Read-only and stdlib-only. It runs and times nothing. Every number below was copied from a record
already on disk, and each change cites where its numbers came from. The script prints the
concordance table and the summary that te5b-concordance-2026-09-28.md (beside it) quotes.

    python3 docs/measurements/test-efficiency-2026-09/te5b_concordance.py

Conventions, fixed before the table was assembled:

  * SPEEDUP. Every ratio is a speedup: > 1 means the new arm is faster. In-process ratios are
    old ms / new ms (or new tok/s / old tok/s); kernel benches are old kernel time / new kernel
    time. Where a record states a time ratio new / old (e.g. "fused / f32 kernel time 0.904"), it is
    inverted here and the change's ip_dir says so. For latency metrics (a stall), speedup =
    old / new time.
  * RESOLVED. A side resolves a direction when
      - its recorded per-pair / per-rep range (lo, hi) lies wholly on one side of 1.0, or the arms'
        run ranges do not overlap (bounds min(new)/max(old) .. max(new)/min(old)); or
      - no spread is recorded and |ln r| >= ln(1 + NOISE), NOISE = 3.5% (the between-session drift
        CLAUDE.md documents for nobara); or
      - the row forces it (ip_res / sv_res) with the record's own reason (e.g. a documented ~10%
        run-to-run drift for unpaired CPU runs, or a same-session control cell).
    Pair-sign counts that split (L1 0.5B: 3 of 5 pairs below 1) do not resolve.
  * VERDICT per row: AGREE (both sides resolve, same direction), DISAGREE (both resolve, opposite
    directions), UNRESOLVED (either side does not resolve). The raw sign of the two point estimates
    is reported too.
  * k = ln(served) / ln(in-process): the fraction of the in-process effect, in log terms, that shows
    served. k = 1 perfect transfer; 0 < k < 1 smaller served; k > 1 larger served; k < 0 opposite
    signs. A KERNEL-ONLY bench times a fraction of the token, so k < 1 is the expected shape for it
    (Amdahl), not a failure. The ratio of ratios (served / in-process) is printed too.
  * IN-PROCESS CLASS: "kernel" = a kernel, a kernel category, or a component timed alone (ncu, a
    Go microbenchmark, a GEMV-work or attention sub-bucket); "e2e" = a whole token / step /
    prefill / generation timed in-process or through a Go benchmark. Several rows carry a second
    in-process reading (ip_alt) of the other class; k is summarised per class using the best
    e2e reading a row has.
  * TIER 1: the served A/B is same-session (interleaved, or adjacent arms in one sitting) and its
    two builds or settings differ by the change alone (or by commits the record shows do not touch
    the measured path). TIER 2: both halves exist, but the served A/B is cross-session, cross-tree,
    cross-condition, or spans more than the change. TE5(b)'s rule is graded on tier 1.
  * LOCALITY: "kernel" = inside a kernel, or a different kernel / launch configuration for the
    same op; "outside" = scheduling, batching, fork/join and barrier count, dispatch count,
    allocation / cache state, host overhead; "mixed" = both in one change.
"""

import math
import statistics

NOISE = 0.035           # the resolution threshold for a side with no recorded spread
DO_NOTHING_FLOOR = 0.022  # largest served "effect" on identical code with non-overlapping runs (see NOTES)

M = "docs/measurements/"
PG = M + "cpu-decode-peer-gap-2026-09-27/"
TPG = "docs/tasks/task-cpu-decode-peer-gap-2026-09.md"
TIQ = "docs/tasks/task-int4-weight-quality-2026-09.md"


def C(name, commits, machine, locality, tier, ip_instr, ip_class, ip_dir, ip_src, sv_instr, sv_src, caveat,
      ip_alt_class=None, ip_alt_instr=None):
    return dict(name=name, commits=commits, machine=machine, locality=locality, tier=tier, ip_instr=ip_instr,
                ip_class=ip_class, ip_dir=ip_dir, ip_src=ip_src, sv_instr=sv_instr, sv_src=sv_src, caveat=caveat,
                ip_alt_class=ip_alt_class, ip_alt_instr=ip_alt_instr)


CHANGES = {
    # ============================== TIER 1: kernel-local ==============================
    "L1-f16-amd64": C(
        "L1 f16 group scales, CPU int4 (row-widening f16 kernel)",
        "S0b 79555115 (in-process); build 2669bf11 vs 8940eaca (served); merged 5c85f7c0",
        "nobara, Ryzen 7 3700X, CPU int4", "kernel", 1,
        "cpuDecodeAB paired in-process ABBA, 5 pairs, depth 128 (TestS0F16Scales_AB, test-hook f16/f32 switch)",
        "e2e", "OFF ms / ON ms", PG + "s0b-gate2-ab.log:57,113,169; " + TPG + ":240-244",
        "bench_peer.py goinfer vs goinfer_old vs Ollama, one session, depth 128, 3 runs",
        PG + "gate5-served.json; " + TPG + ":321-327",
        "In-process arms are the S0b prototype (the f32 copy still resident, prefill tiles f32); served is the full "
        "build. The record: 'The 0.5B's in-process 0.968x does not show served (1.016x)' (" + TPG + ":327)."),
    "L1-v1500-arm64": C(
        "L1 as merged on arm64 (aikit v1.50.0): scalar Go f16 widen per scale",
        "new 5c85f7c0 vs old 3cd62e6d (served); the 'scalar' arm of aikit's arm64-f16-widen bench",
        "MacBook M1 Pro, CPU int4 (row4)", "kernel", 1,
        "aikit BenchmarkW4A8Row4F16Fix, M=1 row4 matmul over a cold bank, -count 8, median, scalar vs f32",
        "kernel", "scalar/f32 kernel TIME in the record; inverted", PG + "arm64-fix-kernel-bench.log; " + TPG + ":515-520",
        "bench_peer.py, two order-reversed passes, geometric mean, depth 128, 3 runs",
        PG + "arm64-speed-served.json, arm64-speed-served-pass2.json; " + TPG + ":410-414",
        "Kernel bench ran ~90 min after the served passes, same Mac, same day. The 1.5B in-process figure is the "
        "geometric mean of its gate/up and down shapes. ip_alt = the interleaved re-run of the same arms.",
        ip_alt_class="kernel", ip_alt_instr="BenchmarkW4A8Row4F16FixInterleaved (" + PG + "arm64-fix-kernel-bench-interleaved.log)"),
    "L1-fix-arm64": C(
        "L1 arm64 fix B: NEON widen + fused f16 row4 decode kernel (aikit v1.50.1)",
        "aikit 2fd6f59 via goinfer e351fad4 vs 3cd62e6d; 5d223da3", "MacBook M1 Pro, CPU int4 (row4)", "kernel", 1,
        "aikit BenchmarkW4A8Row4F16Fix, -count 8, median, fused vs f32", "kernel",
        "fused/f32 kernel TIME in the record; inverted", PG + "arm64-fix-kernel-bench.log; " + TPG + ":515-524",
        "bench_peer.py pass 1 (fix -> Ollama -> old), depth 128, 3 runs; pass 2 skipped by owner amendment",
        PG + "arm64-fix-served.json; " + TPG + ":530-534",
        "Kernel bench at loadavg 1.5-2.8 by day. ip_alt = the interleaved drift-check run (7B 0.949 vs 0.870).",
        ip_alt_class="kernel", ip_alt_instr="BenchmarkW4A8Row4F16FixInterleaved"),
    "R18-metal": C(
        "R18 Metal decode GEMV, MLX-shaped rows kernels (i2244), bit-identical",
        "serve-metal-622b1f9b vs 65ffe328; confirmation afd77749", "MacBook M1 Pro, Metal W4A8", "kernel", 1,
        "TestR18InSequence (testhooks), int4-GEMV work per token, 7 paired reps", "kernel",
        "current work / candidate work", M + "metal-decode-gemv-r18-2026-09-26.md:125-130 (alt :219)",
        "bench_peer.py, three engines interleaved cell by cell, 3 runs x 8 x 64",
        M + "metal-decode-gemv-r18-2026-09-26.md:241-246",
        "Confirmation 13:26-13:38, served 15:06-15:35, same day. Served spread <= 0.5 tok/s.",
        ip_alt_class="e2e", ip_alt_instr="wired production vs shipped, full token GPU ms (:219)"),
    "R18b-metal": C(
        "R18b Metal decode GEMV, masked half-staged form (h4244), bit-identical",
        "serve-metal-c5d7e310 vs 9bafd1f3", "MacBook M1 Pro, Metal W4A8", "kernel", 1,
        "TestR18InSequence (testhooks), int4-GEMV work per token, 7 paired reps", "kernel",
        "production work / candidate work", M + "metal-decode-gemv-r18b-2026-09-26.md:61-66",
        "bench_peer.py, three engines interleaved cell by cell, 3 runs x 8 x 64",
        M + "metal-decode-gemv-r18b-2026-09-26.md:108-113",
        "ip_alt combines R18's wired full token (r18 :219) with R18b's per-depth saving (r18b :61-66); the "
        "record's own 1.5B d128 token reading is 11.33 -> 10.58 ms = 1.07x (r18b :119).",
        ip_alt_class="e2e", ip_alt_instr="full token GPU ms, derived"),
    "R17-metal": C(
        "R17 Metal decode attention at depth, attention_fa_blk (GQA groups 6/7)",
        "7df881f5 vs dde11d93", "MacBook M1 Pro, Metal", "kernel", 1,
        "TestR17AttentionProto, in-sequence attention = full - no-op, arms interleaved rep by rep "
        "(confirmation 7 reps at 1.5B@3900; exploratory 5 reps elsewhere)", "kernel", "current / prototype attention ms",
        M + "metal-decode-attn-r17-2026-09-25.md:71-77, 433-439",
        "bench_peer.py, same session, three engines interleaved, 3 runs x 8 x 64",
        M + "metal-decode-attn-r17-2026-09-25.md:517-522",
        "The 128-key cells (block kernel not engaged) read 0.98 / 1.00: the run's noise floor, +-2%.",
        ip_alt_class="e2e", ip_alt_instr="full token ms in the same test (:435, :82-83)"),
    "R2-metal": C(
        "R2 Metal attention_fa as the default decode attention past 1536 keys",
        "7135283f; measured at 344b9514", "MacBook M1 Pro, Metal, 1.5B", "kernel", 1,
        "metal depth bench (TestZZ_metalDepthBench), 3 runs per arm interleaved, each its own process, best-of-5",
        "e2e", "attention_fa tok/s / shipped tok/s", M + "r2-attn-fa-speed-2026-09-21.md:44-49",
        "bench_peer.py, one invocation per arm, Ollama inside each as the drift control",
        M + "r2-attn-fa-speed-2026-09-21.md:62-70",
        "In-process depth 4000 vs served 3900. The record: 'reproduces the synchronous depth bench's ratio to "
        "the third digit'. ip_alt = the deleted tight-interleaved kernel probe (r2-attn-fa-2026-09-19.md:83).",
        ip_alt_class="kernel", ip_alt_instr="kernel-pair probe, min-of-40, S=14"),
    "R6-cuda": C(
        "R6 CUDA flash-decode lane attn_decode_fa (S=16)",
        "d08eefa1 / c0fda79b; default-on 643ea0ac", "nobara, RTX 2070 SUPER, driver 595.91.07", "kernel", 1,
        "TestFlashDecodeKernelLadder, ncu gpu__time, one layer, exact split-KV vs lane", "kernel",
        "exact us / lane us", M + "attn-decode-fa-ladder-2026-09-20.md:16-17",
        "bench_splitkv.py, fresh serve per arm, arms adjacent with alternating order, idle-gated",
        M + "attn-decode-fa-served-2026-09-20.md:18,21",
        "Kernel ladder on synthetic f32 KV while a CPU reference build loaded the host (ncu device time is "
        "immune, per the record)."),
    "vsum-spike-cuda": C(
        "CUDA flash-decode V-sum split spike (S=4)", "spike binary 2026-09-13 (opt-in, never default)",
        "nobara, RTX 2070 SUPER; D7 (Qwen2.5-7B) at depth 8000", "kernel", 1,
        "ncu, attention kernels total, 32 launches median, spike vs bit-identical split path", "kernel",
        "baseline us / spike us (447.5 / 318.1)", M + "vsum-split-spike-2026-09-13.md:11,15",
        "two serve arms, the same 8000-token prompt, 48 greedy tokens, one pair", M + "vsum-split-spike-2026-09-13.md:65-72",
        "Served is one pair (an earlier unlogged pair read +15.6%); both arms emit degenerate text."),
    "fused-rms-qkv-cuda": C(
        "CUDA fused_rms_qkv rows-per-warp (fewer redundant prologues), bit-identical", "74813033",
        "nobara, RTX 2070 SUPER", "kernel", 1,
        "TestFusedQKVRowsBench, ncu, rows-per-warp as a runtime parameter, random int4 weights of each geometry",
        "kernel", "original us / new us", M + "fused-rms-qkv-2026-09-21.md:24,26 (alt :47)",
        "bench_peer.py, same session, three engines, one sweep (n=2)",
        M + "fused-rms-qkv-2026-09-21/served-three-way.json; fused-rms-qkv-2026-09-21.md:38-44",
        "ip_alt = in-situ D7 GPU time per token, ncu of each build (not one process). The 0.5B keeps the original "
        "kernel and is a do-nothing control, not a row (see NOTES).",
        ip_alt_class="e2e", ip_alt_instr="ncu in situ, per build"),
    "glu-quant-cuda": C(
        "CUDA glu_quant launched at 1024 threads instead of 256 (bit-identical)", "a5ac99d2",
        "nobara, RTX 2070 SUPER, D7", "kernel", 1,
        "ncu of a served decode window, before and after builds (kernel time)", "kernel", "old us / new us",
        M + "d7-decode-breakdown-2026-09-21.md:34",
        "bench_peer.py, same session, three engines, one sweep (n=2)",
        M + "d7-decode-breakdown-2026-09-21/served-three-way.json; d7-decode-breakdown-2026-09-21.md:43-51",
        "The in-process leg is a per-build profile, not a same-process A/B. Only the D7 has an in-process "
        "reading; the 1.5B's served 1.055x has none.",
        ip_alt_class="e2e", ip_alt_instr="D7 GPU time per token, ncu per build"),
    "fused-rms-gu-wave-cuda": C(
        "CUDA fused_rms_gu wave rule for rows-per-warp (bit-identical)", "wave-rule follow-up to 74813033",
        "nobara, RTX 2070 SUPER, D7", "kernel", 1,
        "ncu in situ on D7, GPU time per token, per build", "e2e", "old ms / new ms",
        M + "fused-rms-gu-diagnosis-2026-09-21.md:60",
        "bench_peer.py, same session, one sweep (n=2)",
        M + "fused-rms-gu-diagnosis-2026-09-21/served-three-way.json; fused-rms-gu-diagnosis-2026-09-21.md:66-68",
        "The record: 'its GPU-time gain in situ was 2.7%; end to end it shows smaller'."),
    "G35-webgpu": C(
        "WebGPU quantize row max-abs: serial lane-0 scan -> 64-lane tree reduce (bit-identical)",
        "30008c03", "nobara, WebGPU on RTX 2070 SUPER, 1.5B int8int8", "kernel", 1,
        "real-model decode, in-process, interleaved A/B, best-of-6 x 48 tokens", "e2e", "new tok/s / old tok/s",
        "docs/QUEUE.md:1579", "gpu/cmd/serve, streaming chat, best of 4, new arm started twice around the old",
        "docs/QUEUE.md:1589-1596",
        "The record: 'Note the level shift: 118.4 in-process vs 104.5 through the server on the same kernel.'"),
    "G36-webgpu": C(
        "WebGPU decode attention split over keys instead of head dims", "262d4d17",
        "nobara, WebGPU on RTX 2070 SUPER, 1.5B", "kernel", 1,
        "ablation profile, same session, whole token at pos 512", "e2e", "old ms / new ms",
        "docs/QUEUE.md:1725-1727", "gpu/cmd/serve, server-to-server, best of 2, dim-split re-measured after key-split",
        "docs/QUEUE.md:1694-1702",
        "Position mismatch: in-process is one token at pos 512; served '512 tokens' is a generation whose "
        "inter-token rate averages positions up to 512. ip_alt = attention alone, 9.2x.",
        ip_alt_class="kernel", ip_alt_instr="attention ms at pos 512"),
    "R8-vision-cuda": C(
        "R8 CUDA SigLIP tower fused non-causal attention (BM=128)", "64a39a72 / 6baed301",
        "nobara, RTX 2070 SUPER, gemma-3-4b-it tower", "kernel", 1,
        "cuda/vision_tower_timing_test.go, tower seconds, arms in separate processes alternating, 2 rounds",
        "e2e", "exact s / fused s", M + "vision-tower-mma-2026-09-21.md:14",
        "serve --backend cuda, per-image request time, arms N (fused) vs E (exact) in one pre-registered run",
        M + "vision-tower-downstream-2026-09-21.md:8",
        "Per-image served time includes text prefill, 64 generated tokens and host round trips, not only the tower."),
    "S01-prefill-arm64": C(
        "S-01/S-01b register-blocked W4A8 tiles, CPU prefill (aikit v1.31.0 -> v1.32.0; the bump is the adoption)",
        "897fb18d", "MacBook M1 Pro, CPU int4, 1.5B", "kernel", 1,
        "aikit's own single-core kernel microbenchmark (24.1 -> 69.5 GMAC/s), as quoted in 897fb18d", "kernel",
        "new GMAC/s / old GMAC/s", "commit 897fb18d (kernel figure from aikit's SIMD-audit task doc, S-01)",
        "bench_peer_prefill.py, three arms interleaved within each cell, order rotated per depth, TTFT tok/s",
        "commit 897fb18d",
        "The commit: 'The 2.88x kernel win compresses to ~1.6x end-to-end, which is what S-02's untouched fork/join and "
        "S-06's serial transcendentals predict.' K=512 is the noisy cell (int4 spread up to 10.5%)."),
    "MC3-S4-metal": C(
        "MC3 S4: qkv and gate|up as per-row production GEMVs at B <= 2 (calibrated kernel choice)",
        "7fa344b2 vs 915bf6fc", "MacBook M1 Pro, Metal", "kernel", 1,
        "TestMC3Step_throughput, in sequence, depth 128, B=2 step before -> after", "e2e", "before ms / after ms",
        M + "concurrency-mc3-s4-2026-09-27.md:78,80", "bench_w7_plain.py, 2 clients, 3 interleaved pairs",
        M + "concurrency-mc3-s4-2026-09-27.md:9-10",
        "The before step is from the earlier 7B W7 record's run. The record explains the larger served 1.5B gain by "
        "skipped packing dispatches and the served mix: 'Neither was separated.'"),
    # ============================== TIER 1: mixed ==============================
    "q4k-vs-int4-cuda": C(
        "CUDA --quant q4k (native Q4_K GEMV + per-32 activations) vs today's int4",
        "gemv_q4k_g32 v2 53f1a395, PR #3 210e7307; served d5077650", "nobara, RTX 2070 SUPER, CUDA resident", "mixed", 1,
        "BenchmarkResidentDecode (cuda), 6 paired rounds, alternating, separate processes (7B); same binary, same "
        "session, 2 reps (1.5B, lever-1 fusion check)", "e2e", "q4k tok/s / int4 tok/s",
        TIQ + ":351 (7B), :456 (1.5B); " + M + "q4k-lever1-2026-09-26/fusion-check.txt",
        "bench_peer.py, S1 (q4k) and S2 (int4) back to back, each with its own peers, 3 runs",
        M + "q4k-peer-2026-09-26/s1-q4k.json, s2-int4.json; " + TIQ + ":398,401",
        "A quant-mode change: +9-12% weight bytes and a new GEMV (kernel), and on the 1.5B the per-32 path turning "
        "fusion off (~23%, launch count). Sessions back to back; llama.cpp moved <= 0.3% between them."),
    "R9-linux": C(
        "R9 Linux fixes: activation fan-out off + grouped-attention Go fallback off (non-arm64)",
        "714ad587 in HEAD aecfc9be vs 3ea2f93d (served)", "nobara, Ryzen 7 3700X, CPU int4", "mixed", 1,
        "in-process before/after runs of the shipped defaults (unpaired)", "e2e", "before ms / after ms",
        M + "cpu-decode-attribution-2026-09-22-linux.md:123-129",
        "bench_peer.py, goinfer HEAD vs goinfer_old 3ea2f93d vs Ollama, one session, n=2",
        M + "cpu-peer-reanchor-2026-09-22.json; cpu-peer-reanchor-2026-09-22.md:56-58",
        "The served span also has an aikit v1.46->v1.47 bump (arm64-only S-05) and a 0.5B regression the record "
        "could not explain. The 0.5B has no in-process arm and is not a row. ip_alt = the product of the two paired "
        "per-knob ABBAs (1.189 x 1.118; independence assumed, not measured).",
        ip_alt_class="e2e", ip_alt_instr="TestR9_cpuTuningAB per-knob ABBA product"),
    # ============================== TIER 1: outside the kernel ==============================
    "fused-gateup-amd64": C(
        "Fused gate+up+SwiGLU decode fork/join (one barrier per layer, the activation inside it)",
        "a79ef544 (served build d88ef960, its pre-rebase twin) vs ec6fec95; only docs commits between",
        "nobara, Ryzen 7 3700X, CPU int4", "outside", 1,
        "TestCPURoofline_fusedGateUp, paired in-process ABBA, 5 pairs", "e2e", "plain ms / fused ms",
        M + "cpu-decode-roofline-2026-09-23.md:124-126",
        "bench_peer.py phase A, same session, interleaved with Ollama, depth 128, n=2",
        M + "cpu-decode-roofline-served-2026-09-23.json; cpu-decode-roofline-2026-09-23.md:139-141",
        "The record: 'The served gain on the 0.5B and 1.5B is larger than the in-process A/B predicted ... and I "
        "have not found why' (:143)."),
    "G26-sampler-cuda": C(
        "Sampler, anchor ca29d6c -> HEAD f9f8833 (746 commits; P10 scratch-buffer reuse among them)",
        "ca29d6c vs f9f8833; record 28957403", "nobara, RTX 2070 SUPER, CUDA resident, optFwd off in both arms",
        "outside", 1,
        "BenchmarkG26SampleTemp1_{32k,152k}, go test -bench, 5 runs per tree, each tree its own build", "kernel",
        "anchor ns/op / HEAD ns/op", M + "g26-sampler-bench.log; docs/QUEUE.md:1003-1008",
        "bench_peer.py temp1.0_notrunc cell, anchor then HEAD sessions back to back (n=8 at 152k, n=15 at 32k)",
        M + "g26-152k-{anchor,head}.json, g26-anchor-n15.json, g26-head-nooptfwd-n15.json; docs/QUEUE.md:1003-1019",
        "The greedy control moved +0.3%, so the served delta is the sampled tail. The record's mechanism: 'In a tight "
        "loop the scratch buffer stays hot and the allocator serves from a warm free-list; in decode the same buffer "
        "is cold behind an ~8 ms forward. The loop ... inverted the comparison.' An allocation / cache-state effect."),
    "MC3c-S1-cpu": C(
        "MC3c step 2 S1: fused q||k||v and gate||up W4A8 batch in the batched CPU step",
        "000efe2e vs 2c1d89ec; pre-registered 5f7480e0", "nobara, Ryzen 7 3700X, CPU int4, 7B", "outside", 1,
        "TestCPUBatchS1_fusedVsUnfused, test-hook flip in-process, 5 reps, B=4", "e2e", "fused / unfused tok/s",
        M + "concurrency-mc3c-s1-2026-09-27.md:50-51", "bench_w7_plain.py, 4 clients, 3 pairs old new new old old new",
        M + "concurrency-mc3c-s1-2026-09-27.md:74-78", "Three of five fork/joins per layer removed; kernel unchanged."),
    "MC3-metal": C(
        "MC3 batched decode on Metal (matrix-unit batched matmuls in a shared step)", "MC3 build vs MC1 serialized",
        "MacBook M1 Pro, Metal, 1.5B", "outside", 1,
        "MC3 S1 step in sequence: aggregate = B x production / step(B), paired per rep", "e2e",
        "batched aggregate / one stream", M + "concurrency-mc3-s1-2026-09-26.md:71",
        "W7 through serve, 3 interleaved pairs at 4 clients; 2 clients reported",
        M + "concurrency-mc3-2026-09-26.md:92,103",
        "The in-process quantity is a fixed-B step; served is a workload aggregate with prefills and admission."),
    "MC3-7B-stack-metal": C(
        "MC3 stack (MC3+S2+S3+chunked prefill) on the 7B vs the pre-MC3 serialized build",
        "current stack vs serialized, 2026-09-27", "MacBook M1 Pro, Metal, 7B", "outside", 1,
        "TestMC3Step_throughput on the current step, in sequence, 7 reps, same session as the W7 run", "e2e",
        "aggregate vs one stream", M + "concurrency-mc3-7b-w7-2026-09-27.md:107-109",
        "W7 through serve, 3 interleaved pairs at 4 clients; 2 clients reported",
        M + "concurrency-mc3-7b-w7-2026-09-27.md:74,77",
        "The record: '4 clients reach 1.785x end to end, against 1.94-2.00x in sequence' (:127)."),
    "MC3-S3-metal": C(
        "MC3 S3 step polish: per-row kernels as one multi-row dispatch", "731f4f4e vs 4954f978",
        "MacBook M1 Pro, Metal, 1.5B", "outside", 1, "GPU step time in sequence, before vs after S3a+S3b", "e2e",
        "before ms / after ms", M + "concurrency-mc3-s3-2026-09-27.md:56-58",
        "W7 through serve, 3 interleaved pairs at 4 and 2 clients", M + "concurrency-mc3-s3-2026-09-27.md:90,93",
        "The saving is dispatch count and serial per-row work; kernel bodies byte-identical."),
    "MC3-cuda": C(
        "MC3 on CUDA: batched multi-request decode", "7a44a58e vs 01bcb640", "nobara, RTX 2070 SUPER", "outside", 1,
        "MC3 CUDA S0: batched M-row pass with every row's exact logits vs M sequential decodes, depth 1024, best of 5",
        "e2e", "sequential ms / batched ms", M + "concurrency-mc3-cuda-s0-2026-09-27.md:34-38",
        "bench_w7_plain.py, 3 interleaved pairs at 4 clients (1.5B); one pair each for 1.5B 2c and 7B 4c",
        M + "concurrency-mc3-cuda-2026-09-27.md:15,67-68",
        "S0 is a proxy (one sequence's rows at consecutive positions). The record: 'The 7B keeps ~90% of its "
        "step-level gain. The 1.5B keeps ~79%'."),
    "MC3c-step1-cpu": C(
        "MC3c step 1: -max-concurrent N independent CPU workers", "MC3c build at -max-concurrent 4 vs 1",
        "MacBook M1 Pro, CPU int4, 1.5B", "outside", 1,
        "MC2 J8 cell: N independent decode workers on one model vs serial x1, in-process", "e2e", "aggregate ratio",
        M + "concurrency-mc2-2026-09-26.md:58", "bench_w7_plain.py, one session; 4 clients three times, 2 clients once",
        M + "concurrency-mc3c-2026-09-26.md:11,66-68", "MC2 ran before the serve path existed."),
    "MC3c-step2-cpu": C(
        "MC3c step 2: batched CPU decode behind serve's admission (vs step 1's workers)", "2c1d89ec vs 0bc06f90",
        "nobara, Ryzen 7 3700X, CPU int4, 7B", "outside", 1,
        "MC2 Linux 7B cell: batched B vs J8 N workers, per rep, median of 5 (test-only prototype)", "e2e",
        "batched / workers aggregate", M + "concurrency-mc2-2026-09-26.md:180",
        "bench_w7_plain.py, 3 pairs at 4 clients; one 2-client pair", M + "concurrency-mc3c-step2-2026-09-27.md:66,74",
        "The record: served 2.19x is 'the trigger's 2.25-2.41x less the per-turn costs batching does not touch'."),
    "chunked-prefill-metal": C(
        "Chunked prefill under MC3 (256-token chunks)", "d4293d41 vs 731f4f4e", "MacBook M1 Pro, Metal, 1.5B",
        "outside", 1,
        "TestMC5_chunkCost: a 3000-token prefill alone, whole vs chunked, 3 reps interleaved (after the grading)",
        "e2e", "whole / chunked (time); stall = whole pass / longest pass", M + "chunked-prefill-2026-09-27.md:92-97",
        "bench_prefill_stall.py, 3 interleaved pairs", M + "chunked-prefill-2026-09-27.md:76,78",
        "Two quantities: the decoders' stall (the target) and the cost (in-process: prefill total; served: cell wall)."),
    # ============================== TIER 2 ==============================
    "row4-arm64": C(
        "W4A8 row4 repack + kernel on arm64 (with its dispatch fix)", "1c271213, 0bb63a14",
        "MacBook M1 Pro, CPU int4, 1.5B", "kernel", 2,
        "stub probe, canonical vs row4 in one harness, 64 decode steps (total time per token)", "e2e",
        "canonical ms / row4 ms", "docs/completed/task-w4a8-neon-bandwidth.md:896-901",
        "bench_peer, after vs the campaign's Step 0 before (another session)",
        "docs/completed/task-w4a8-neon-bandwidth.md:966-968", "Cross-session served.",
        ip_alt_class="kernel", ip_alt_instr="W4A8-only time per token in the same probe"),
    "lmhead-w8a8-arm64": C(
        "int4-mode LM head: weight-only Q8 -> full W8A8 (kernel selection)", "a11c56bf",
        "MacBook M1 Pro, CPU int4, 1.5B", "kernel", 2,
        "isolated LM-head shape (K=1536, N=151936): MatmulBTQ8 vs MatmulBTW8A8Into", "kernel", "W8A8 GB/s / Q8 GB/s",
        "docs/completed/task-w4a8-neon-bandwidth.md:927-931", "bench_peer, two separate runs, vs the post-row4 before",
        "docs/completed/task-w4a8-neon-bandwidth.md:1000-1004", "Cross-session served; not bit-identical."),
    "A1-attn-arm64": C(
        "A1 decode attention: acc64 AV + 8-wide QK kernels + head threading", "task-attention-decode-cost A1",
        "MacBook M1 Pro, CPU, 1.5B depth 128", "mixed", 2,
        "attendBatchedHeads depth curve, before vs after (c)+(b)+(a), attention only", "kernel",
        "before ms / after ms", "docs/completed/task-attention-decode-cost.md:465-470",
        "bench_peer method, after vs the 2026-08-22 diagnosis session", "docs/completed/task-attention-decode-cost.md:478-484",
        "Cross-session served. The record's Amdahl check predicted 21.55 tok/s against 21.52 measured (:494-498)."),
    "R06-amd64-xtree": C(
        "R-06 W4A8 q/k/v (+gate/up) batch, one fork/join", "e4e04992 / 4e933525 (served 09-04); in-process 09-23 tree",
        "nobara, Ryzen 7 3700X, CPU int4, 1.5B", "outside", 2,
        "TestCPURoofline_w4a8Batch paired ABBA, 5 pairs (plain -> batch)", "e2e", "OFF ms / ON ms",
        M + "cpu-decode-roofline-2026-09-23.md:124",
        "bench_peer.py goinfer-only cell, GOINFER_W4A8_BATCH toggled, n=10 paired", "docs/tasks/task-recompute-audit.md:532-543",
        "19 days and the R9 fixes apart: the in-process tree has fewer barriers per layer than the served one."),
    "splitkv-A2-cuda": C(
        "CUDA split-KV decode attention default-on at nKeys >= 256 (kernel choice + extra launches)",
        "a4932832 / 26ae07da (in-process 08-04); served P6a 08-09 at 686c9f8, driver 595.58.03",
        "nobara, RTX 2070 SUPER, 1.5B", "mixed", 2,
        "TestSplitKVCrossover (tight in-process ForwardArgmax loop, best-of-3 minimum); TestDecodeDepthThroughput at 2048",
        "e2e", "ON / OFF tok/s", "docs/tasks/task-decode-splitkv-attention.md:174-176; docs/ollama-chase.md:315-317",
        "P6a: 48 cells, a fresh serve per cell, ON / OFF, decode-only client timing", "docs/legacy-benchmarks.md:1062",
        "Cross-session and cross-build. The record: the test 'times a tight in-process ForwardArgmax loop and takes "
        "best-of-3 minimum. Both choices flatter split-KV against real serving - the loop hides the per-token CPU "
        "dispatch a real request exposes' (docs/legacy-benchmarks.md:1072-1078)."),
    "M26-batched-prefill-cuda": C(
        "CUDA MoE (Gemma-4-26B) takes the batched prefill path", "4ee59e15 / 654fa481 / 9453430e",
        "nobara, RTX 2070 SUPER, M26", "outside", 2,
        "in-process paired within one process, M=512/2048 (not interleaved, not at depth 8000)", "e2e",
        "old ms / new ms", "docs/queue-performance.md:166; docs/benchmarks.md:1903-1905",
        "bench_peer W3 cell wall-clock at depth 8000, re-run at 9453430e against an earlier cell",
        "docs/benchmarks.md:1882,1903-1905",
        "Cross-session served; the served cell also contains four decodes (unchanged), diluting a prefill-only change."),
    "P2b-host-sampler-cuda": C(
        "P2b deterministic parallel host normalization for temperature sampling", "686c9f8 (after) vs ed81e13 (P1, before)",
        "nobara, RTX 2070 SUPER, driver 595.58.03", "outside", 2,
        "host-only sampler microbenchmark, 16 cores", "kernel", "old ns/entry / new ns/entry",
        "docs/completed/plan-still-slow.md:277-279",
        "decode-only, client-timed from the first streamed token, 8 completions x 2 runs, temperature-only t=1.0",
        "docs/completed/plan-still-slow.md:281-288", "Interleaving of the before/after binaries is not stated."),
    "R1-W4F16-metal": C(
        "R1 Metal W4F16 decode lane (killed)", "ebe50556", "MacBook M1 Pro, Metal, 1.5B", "kernel", 2,
        "TestZZ_metalDepthBench, once per lane setting (separate processes), min-of-5", "e2e", "W4F16 / W4A8 tok/s",
        M + "w4f16-decode-speed-2026-09-21.md:74", "bench_peer.py, two separate invocations, Ollama inside each",
        M + "w4f16-decode-speed-2026-09-21.md:52-59", "Not interleaved; Ollama moved ~2% between the invocations."),
    "L2L3-prefill-cuda": C(
        "CUDA prefill: fused attention (L2) + tensor-core GEMM (L3)", "f966fa0c", "nobara, RTX 2070 SUPER, 1.5B",
        "kernel", 2, "TestPrefillTTFT, batched arm, per-lever env arms", "e2e", "baseline ms / both ms",
        M + "prefill-l2l3-phase2-2026-09-05.md:40-43", "bench_peer_prefill.py, exact arm and fast arm as two runs, TTFT tok/s",
        M + "prefill-l2l3-phase4-peer-2026-09-05.md:57-62",
        "The record: 'The arm-vs-arm comparison is across two runs (~3.5% drift on this box)'."),
    "R3-prefill-floor-metal": C(
        "R3 Metal short-prompt prefill floor 256 -> 64 (batched vs per-token command buffers)", "a9ec3452",
        "MacBook M1 Pro, Metal, 1.5B", "mixed", 2,
        "TestR3_startPosSpeed, one resident, 6 interleaved reps, startPos 512", "e2e", "sequential ms / batched ms",
        M + "r3-startpos-speed-2026-09-21.md:50-51", "bench_peer_prefill.py, exact vs batched, interleaved per depth, startPos 0",
        M + "metal-prefill-floor-2026-09-20.md:50-51",
        "Different days, startPos and prompts; the record: 'not directly poolable'."),
    "optfwd-cuda": C(
        "Optimistic forward for sampled decode (on vs off)", "6a4e0ae; in-process ff809596; served 28957403",
        "nobara, RTX 2070 SUPER, CUDA resident", "outside", 2,
        "cuda optfwd end-to-end benchmark (port of metal/optfwd_bench_test.go), qwen2.5-coder-0.5b, synthetic "
        "token-id prompt", "e2e", "off ms / on ms", "commit ff809596",
        "bench_peer.py, phi3-mini, depth 128, GOINFER_NO_OPTFWD toggled, n=6 temperature ladder",
        "docs/QUEUE.md:606-615",
        "Different model and prompt on the two sides (0.5B synthetic vs phi3-mini real); the in-process record itself "
        "warns that its synthetic prompt makes T=0.7 'a weak place to judge the feature'."),
    "dflash-drafter": C(
        "--drafter block speculation (DFlash), Qwen3-4B with the server's default thinking template",
        "7c485d31", "GPU resident (the spec's Model.Generate / GPU-argmax path); machine not restated", "outside", 2,
        "gate 3 re-run against Model.Generate as the baseline", "e2e", "spec tok/s / baseline tok/s",
        "docs/spec/08-dspark-dflash.md:2241-2245",
        "a served A/B, cited as 0.83x", "docs/spec/08-dspark-dflash.md:2247",
        "The served A/B's raw record was not found. The first in-process reading, against a slower-than-production "
        "baseline, was 0.98x (ip_alt): the correction is what brought in-process into line with served.",
        ip_alt_class="e2e", ip_alt_instr="gate 3 against the old PrefillLastNArgmax(M=1) baseline"),
    "R16-prefill-gemm-metal": C(
        "R16 Metal prefill GEMM redesign (prototype 4)", "b4e1f2c0 / 1fd9d95e", "MacBook M1 Pro, Metal, 1.5B",
        "kernel", 2, "GEMM category in sequence, 7 paired reps", "kernel", "current ms / prototype ms",
        M + "metal-prefill-gemm-s2-2026-09-25.md:193-200",
        "bench_peer_prefill.py after wiring vs the morning's cell h (another session), TTFT tok/s at K=512",
        M + "metal-prefill-gemm-s2-2026-09-25.md:270-275", "Cross-session served; Ollama as the common control.",
        ip_alt_class="e2e", ip_alt_instr="full prefill replay GPU ms"),
}


def inv(x):
    return 1.0 / x


def gm(*xs):
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


# One row per (change, cell). Speedups. *_lo/_hi = range or non-overlap bounds; None = not recorded.
ROWS = [
    # L1 f16, amd64
    dict(c="L1-f16-amd64", cell="0.5B d128", ip=0.968, ip_lo=0.960, ip_hi=1.027,
         ip_note="pairs 0.968/1.017/0.967/0.960/1.027: 3 of 5 below 1",
         sv=1.016, sv_lo=1.0069, sv_hi=1.0184, sv_note="runs 48.4-48.9 vs 48.0-48.1 tok/s"),
    dict(c="L1-f16-amd64", cell="1.5B d128", ip=1.059, ip_lo=1.055, ip_hi=1.066, sv=1.047, sv_lo=1.0432, sv_hi=1.0528),
    dict(c="L1-f16-amd64", cell="7B d128", ip=1.090, ip_lo=1.083, ip_hi=1.094, sv=1.089, sv_lo=1.0889, sv_hi=1.0900),
    # L1 v1.50.0 scalar widen, arm64
    dict(c="L1-v1500-arm64", cell="0.5B d128", ip=inv(1.815), ip_res=True, ip_why="all 8 scalar samples slower than all 8 f32",
         ip_alt=inv(1.694), sv=0.544, sv_lo=0.5331, sv_hi=0.5584),
    dict(c="L1-v1500-arm64", cell="1.5B d128", ip=gm(inv(1.402), inv(1.418)), ip_res=True, ip_why="every sample slower",
         ip_alt=gm(inv(1.193), inv(1.391)), sv=0.554, sv_lo=0.5329, sv_hi=0.5699),
    dict(c="L1-v1500-arm64", cell="7B d128", ip=inv(2.879), ip_res=True, ip_why="every sample slower",
         ip_alt=inv(3.117), sv=0.435, sv_lo=0.3403, sv_hi=0.4961),
    # L1 fix, arm64
    dict(c="L1-fix-arm64", cell="0.5B d128", ip=inv(0.904), ip_res=True, ip_why="every fused sample faster",
         ip_alt=inv(0.904), sv=1.041, sv_lo=1.0259, sv_hi=1.0541),
    dict(c="L1-fix-arm64", cell="1.5B d128", ip=gm(inv(0.905), inv(0.898)), ip_res=True, ip_why="every fused sample faster",
         ip_alt=gm(inv(0.898), inv(0.898)), sv=1.053, sv_lo=1.043, sv_hi=1.063),
    dict(c="L1-fix-arm64", cell="7B d128", ip=inv(0.870), ip_res=True, ip_why="every fused sample faster",
         ip_alt=inv(0.949), sv=1.080, sv_lo=1.0662, sv_hi=1.0801, sv_note="record means 18.9 / 17.5; runs give 1.075"),
    # R18
    *[dict(c="R18-metal", cell=cell, ip=ip, ip_lo=lo, ip_hi=hi, ip_alt=alt, sv=sv, sv_res=True,
           sv_why="spread <= 0.5 tok/s on 20-85 tok/s cells")
      for cell, ip, lo, hi, alt, sv in [
          ("1.5B d128", 1.171, 1.151, 1.191, 12.99 / 11.30, 1.135), ("1.5B d2048", 1.218, 1.189, 1.299, 14.06 / 12.37, 1.133),
          ("1.5B d3900", 1.221, 1.211, 1.238, 14.85 / 13.05, 1.124), ("7B d128", 1.317, 1.305, 1.329, 45.67 / 35.96, 1.265),
          ("7B d2048", 1.337, 1.322, 1.343, 49.50 / 39.68, 1.240), ("7B d3900", 1.332, 1.300, 1.383, 52.19 / 42.37, 1.226)]],
    # R18b
    *[dict(c="R18b-metal", cell=cell, ip=ip, ip_lo=lo, ip_hi=hi, ip_alt=base / (base - save), sv=sv, sv_res=True,
           sv_why="spread <= 1.0 tok/s")
      for cell, ip, lo, hi, base, save, sv in [
          ("1.5B d128", 1.079, 1.057, 1.107, 11.30, 0.749, 1.073), ("1.5B d2048", 1.121, 1.103, 1.158, 12.37, 0.823, 1.071),
          ("1.5B d3900", 1.136, 1.110, 1.155, 13.05, 0.875, 1.064), ("7B d128", 1.102, 1.039, 1.106, 35.96, 2.840, 1.085),
          ("7B d2048", 1.106, 1.044, 1.124, 39.68, 2.910, 1.077), ("7B d3900", 1.112, 1.101, 1.121, 42.37, 2.890, 1.069)]],
    # R17
    dict(c="R17-metal", cell="1.5B d2048", ip=4.874 / 1.770, ip_note="exploratory, 5 reps", sv=1.22, sv_res=True,
         sv_why="68.1-69.0 vs 56.2 tok/s; 128-key control +-2%"),
    dict(c="R17-metal", cell="1.5B d3900", ip=3.50, ip_lo=3.4, ip_hi=3.7, ip_alt=20.785 / 14.713, sv=1.41, sv_res=True,
         sv_why="65.2-65.6 vs 46.4 tok/s"),
    dict(c="R17-metal", cell="7B d2048", ip=13.601 / 4.898, ip_note="exploratory, 5 reps", sv=1.18, sv_res=True,
         sv_why="19.9 vs 16.9 tok/s; 128-key control +-2%"),
    dict(c="R17-metal", cell="7B d3900", ip=3.49, ip_lo=3.1, ip_hi=3.7, ip_alt=69.85 / 52.30, sv=1.33, sv_res=True,
         sv_why="18.9 vs 14.2 tok/s"),
    # R2
    dict(c="R2-metal", cell="1.5B d4000 / d3900", ip=44.9 / 37.8, ip_lo=44.9 / 39.6, ip_hi=45.0 / 37.8, ip_alt=1.21,
         sv=46.4 / 39.1, sv_lo=46.4 / 39.1, sv_hi=46.4 / 39.0),
    # R6
    dict(c="R6-cuda", cell="1.5B d3900", ip=146.3 / 42.3, sv=1.557, sv_note="194.1 / 124.7 tok/s; 194.2 on a second run"),
    dict(c="R6-cuda", cell="7B (D7) d8000", ip=439.4 / 107.7, sv=1.543, sv_note="61.1 / 39.6 tok/s"),
    # V-sum spike
    dict(c="vsum-spike-cuda", cell="7B (D7) d8000", ip=447.5 / 318.1, sv=44.84 / 38.60, sv_note="one pair"),
    # fused_rms_qkv
    dict(c="fused-rms-qkv-cuda", cell="7B (D7) d128", ip=50.5 / 32.5, ip_alt=12.78 / 12.31,
         sv=1.0359, sv_lo=1.0355, sv_hi=1.0364),
    dict(c="fused-rms-qkv-cuda", cell="1.5B d128", ip=15.9 / 11.5, sv=1.0187, sv_lo=1.0175, sv_hi=1.0200),
    # glu_quant
    dict(c="glu-quant-cuda", cell="7B (D7) d128", ip=35.7 / 13.6, ip_alt=13.39 / 12.78,
         sv=1.0453, sv_lo=1.0448, sv_hi=1.0458),
    # fused_rms_gu wave rule
    dict(c="fused-rms-gu-wave-cuda", cell="7B (D7) d128", ip=12.20 / 11.87, sv=1.0197, sv_lo=1.0194, sv_hi=1.0199),
    # G35 / G36 WebGPU
    dict(c="G35-webgpu", cell="1.5B, 128 tokens", ip=118.4 / 104.8, ip_res=True, ip_why="both arms reproduced exactly",
         sv=104.4 / 95.9, sv_lo=104.3 / 95.9, sv_hi=104.5 / 95.9),
    dict(c="G36-webgpu", cell="1.5B pos 512 / 512 tokens", ip=19.609 / 7.431, ip_alt=13.676 / 1.491,
         sv=122.8 / 68.3, sv_lo=122.8 / 69.0, sv_hi=122.8 / 67.6),
    # R8 vision
    dict(c="R8-vision-cuda", cell="gemma-3-4b tower / img0", ip=25.9575 / 4.080, ip_res=True, ip_why="spread < 0.02 s",
         sv=28.2 / 6.05, sv_lo=27.9 / 6.3, sv_hi=28.5 / 5.8),
    # S-01 CPU prefill tiles
    dict(c="S01-prefill-arm64", cell="1.5B K=512 (n=9)", ip=69.5 / 24.1, sv=1.788),
    dict(c="S01-prefill-arm64", cell="1.5B K=3900", ip=69.5 / 24.1, sv=1.618),
    # MC3 S4
    dict(c="MC3-S4-metal", cell="7B B=2 / 2 clients", ip=65.29 / 58.76, sv=1.121, sv_lo=1.093, sv_hi=1.152),
    dict(c="MC3-S4-metal", cell="1.5B B=2 / 2 clients", ip=17.20 / 16.87, sv=1.059, sv_lo=1.053, sv_hi=1.103),
    # q4k vs int4
    dict(c="q4k-vs-int4-cuda", cell="7B d128", ip=0.890, ip_lo=0.888, ip_hi=0.893, sv=0.893, sv_lo=0.8925, sv_hi=0.8930),
    dict(c="q4k-vs-int4-cuda", cell="1.5B d128", ip=gm(193.2 / 254.0, 188.3 / 252.8), ip_lo=188.3 / 252.8,
         ip_hi=193.2 / 254.0, sv=0.766, sv_lo=0.7602, sv_hi=0.7689),
    # R9
    dict(c="R9-linux", cell="1.5B d128", ip=74.5 / 54.5, ip_res=True,
         ip_why="unpaired, but 37% against the record's ~10% back-to-back drift", ip_alt=1.189 * 1.118,
         sv=1.348, sv_lo=1.3383, sv_hi=1.3535),
    dict(c="R9-linux", cell="7B d128", ip=222.2 / 201.9, ip_res=False,
         ip_why="unpaired; 10% is inside the record's own ~10% back-to-back drift", sv=1.089, sv_lo=1.0896, sv_hi=1.0913),
    # fused gate+up
    dict(c="fused-gateup-amd64", cell="0.5B d128", ip=1.113, ip_lo=1.089, ip_hi=1.147, sv=1.241, sv_lo=1.2319, sv_hi=1.2492),
    dict(c="fused-gateup-amd64", cell="1.5B d128", ip=1.066, ip_lo=1.039, ip_hi=1.073, sv=1.135, sv_lo=1.1277, sv_hi=1.1385),
    dict(c="fused-gateup-amd64", cell="7B d128", ip=1.029, ip_lo=1.028, ip_hi=1.030, sv=1.020, sv_lo=1.0214, sv_hi=1.0250,
         sv_note="the record's 1.020 is from rounded means; the runs give 1.023"),
    # G26 sampler
    dict(c="G26-sampler-cuda", cell="1.5B (vocab 151936), temp 1.0", ip=1358.1 / 1676.1,
         ip_lo=1334980 / 1707874, ip_hi=1407140 / 1646267, sv=1.085, sv_lo=1.0492, sv_hi=1.1375,
         sv_note="166.2 -> 180.4 tok/s; in-situ sampling step 1467 -> 1009 us (1.45x)"),
    dict(c="G26-sampler-cuda", cell="phi3-mini (vocab 32064), temp 1.0", ip=627.6 / 552.6,
         ip_lo=618601 / 575725, ip_hi=637390 / 524347, sv=117.84 / 114.40, sv_lo=1.0046, sv_hi=1.0774,
         sv_note="in-situ sampling step 0.703 -> 0.457 ms"),
    # MC3c S1
    dict(c="MC3c-S1-cpu", cell="7B B=4 / 4 clients", ip=1.058, ip_lo=1.055, ip_hi=1.062, sv=1.051, sv_lo=1.051, sv_hi=1.053),
    # MC3 Metal
    dict(c="MC3-metal", cell="1.5B B=4 / 4 clients", ip=1.84, ip_lo=1.78, ip_hi=1.95, sv=1.593, sv_lo=1.583, sv_hi=1.599),
    dict(c="MC3-metal", cell="1.5B B=2 / 2 clients", ip=1.10, sv=81.7 / 76.3, sv_note="one reported pair"),
    # MC3 7B stack
    dict(c="MC3-7B-stack-metal", cell="7B B=4 / 4 clients", ip=1.998, sv=1.785, sv_lo=1.778, sv_hi=1.790),
    dict(c="MC3-7B-stack-metal", cell="7B B=2 / 2 clients", ip=1.016, sv=1.018, sv_lo=1.017, sv_hi=1.018),
    # MC3 S3
    dict(c="MC3-S3-metal", cell="1.5B B=4 / 4 clients", ip=22.59 / 17.69, sv=1.261, sv_lo=1.259, sv_hi=1.300),
    dict(c="MC3-S3-metal", cell="1.5B B=2 / 2 clients", ip=18.85 / 17.20, sv=1.086, sv_lo=1.079, sv_hi=1.107),
    # MC3 CUDA
    dict(c="MC3-cuda", cell="1.5B B=4 / 4 clients", ip=18.58 / 10.61, sv=1.380, sv_lo=1.376, sv_hi=1.381),
    dict(c="MC3-cuda", cell="1.5B B=2 / 2 clients", ip=9.71 / 7.48, sv=236.2 / 217.6, sv_note="one pair"),
    dict(c="MC3-cuda", cell="7B B=4 / 4 clients", ip=55.13 / 27.00, sv=135.16 / 74.03, sv_note="one pair"),
    # MC3c step 1 / step 2
    dict(c="MC3c-step1-cpu", cell="1.5B N=4 / 4 clients", ip=2.002, sv=1.91, sv_lo=1.86, sv_hi=1.97),
    dict(c="MC3c-step1-cpu", cell="1.5B N=2 / 2 clients", ip=1.551, sv=69.3 / 43.9, sv_note="one cell"),
    dict(c="MC3c-step2-cpu", cell="7B B=4 / 4 clients", ip=2.41, ip_lo=2.39, ip_hi=2.44, sv=2.187, sv_lo=2.185, sv_hi=2.200),
    dict(c="MC3c-step2-cpu", cell="7B B=2 / 2 clients", ip=1.691 / 1.092, sv=1.466, sv_note="one pair"),
    # chunked prefill
    dict(c="chunked-prefill-metal", cell="1.5B stall (max decoder gap)", ip=5.37 / 1.20, sv=1 / 0.213,
         sv_lo=5.35 / 1.17, sv_hi=5.41 / 1.14),
    dict(c="chunked-prefill-metal", cell="1.5B cost (prefill total / cell wall)", ip=5.37 / 5.79, sv=1 / 1.052,
         sv_lo=24.4 / 25.8, sv_hi=24.7 / 25.7),
    # ---------------- tier 2
    dict(c="row4-arm64", cell="1.5B d128", ip=gm(1.08, 1.09), ip_lo=1.08, ip_hi=1.09, ip_alt=gm(1.31, 1.35),
         sv=23.2 / 21.68, sv_lo=22.9 / 21.68, sv_hi=23.5 / 21.68),
    dict(c="lmhead-w8a8-arm64", cell="1.5B d128", ip=7.7, sv=39.9 / 23.2, sv_lo=39.1 / 23.5, sv_hi=40.7 / 22.9),
    dict(c="A1-attn-arm64", cell="1.5B d128", ip=10.84 / 2.81, sv=21.52 / 17.0),
    dict(c="R06-amd64-xtree", cell="1.5B d128", ip=1.053, ip_lo=1.049, ip_hi=1.054, sv=1.066, sv_lo=1.065, sv_hi=1.067),
    dict(c="splitkv-A2-cuda", cell="1.5B nKeys 256", ip=1.0, ip_res=False, ip_why="'breaks even at 256'",
         sv=0.941, sv_note="block spreads ON 3.6-6.4 / OFF 0.1-0.6 tok/s"),
    dict(c="splitkv-A2-cuda", cell="1.5B nKeys 512", ip=1.02, ip_res=False,
         ip_why="'wins from 384+ (1.02 -> 1.20x at 2048)': 512 not tabulated; 1.02 is the stated low end",
         sv=0.939),
    dict(c="splitkv-A2-cuda", cell="1.5B nKeys 2048", ip=160.1 / 133.4, sv=1.191),
    dict(c="M26-batched-prefill-cuda", cell="M26 prefill / W3 cell wall", ip=47.382 / 43.681, ip_lo=1.083, ip_hi=1.085,
         sv=384.3 / 368.4),
    dict(c="P2b-host-sampler-cuda", cell="phi3-mini (32k vocab), temp 1.0", ip=1.62, sv=113.7 / 88.8,
         sv_lo=(113.7 - 5.4) / (88.8 + 6.4), sv_hi=(113.7 + 5.4) / (88.8 - 6.4)),
    dict(c="P2b-host-sampler-cuda", cell="0.5B (152k vocab), temp 1.0", ip=3.06, sv=220.2 / 97.6,
         sv_lo=(220.2 - 3.8) / (97.6 + 9.8), sv_hi=(220.2 + 3.8) / (97.6 - 9.8)),
    dict(c="P2b-host-sampler-cuda", cell="gemma3-1b (262k vocab), temp 1.0", ip=4.72, sv=134.2 / 56.6,
         sv_lo=(134.2 - 0.2) / (56.6 + 7.3), sv_hi=(134.2 + 0.2) / (56.6 - 7.3)),
    dict(c="R1-W4F16-metal", cell="1.5B d128", ip=72.4 / 69.9, sv=75.2 / 73.1, sv_lo=75.1 / 73.3, sv_hi=75.3 / 72.9),
    *[dict(c="L2L3-prefill-cuda", cell=f"1.5B K={k}", ip=ip, sv=f / e) for k, ip, e, f in [
        (128, 80.91 / 47.05, 1634.2, 3140.1), (512, 371.5 / 90.52, 1381.6, 5459.9),
        (2048, 2099.0 / 517.3, 967.6, 3740.5), (3900, 5451.0 / 1393.0, 707.1, 2715.0)]],
    dict(c="R3-prefill-floor-metal", cell="1.5B K=64", ip=914.9 / 228.1, sv=365.5 / 95.4),
    dict(c="R3-prefill-floor-metal", cell="1.5B K=128", ip=1851.4 / 448.3, sv=422.2 / 90.6),
    dict(c="optfwd-cuda", cell="T=0.2 (0.5B synthetic / phi3-mini)", ip=262.9 / 216.6,
         sv=119.34 / 118.03, sv_lo=(119.34 - 1.17) / (118.03 + 0.15), sv_hi=(119.34 + 1.17) / (118.03 - 0.15)),
    dict(c="optfwd-cuda", cell="T=1.0 (0.5B synthetic / phi3-mini)", ip=456.0 / 457.7,
         sv=111.72 / 117.89, sv_lo=(111.72 - 0.44) / (117.89 + 0.12), sv_hi=(111.72 + 0.44) / (117.89 - 0.12)),
    dict(c="dflash-drafter", cell="Qwen3-4B thinking", ip=0.82, ip_alt=0.98, sv=0.83),
    dict(c="R16-prefill-gemm-metal", cell="1.5B K=512", ip=1548.6 / 476.6, ip_lo=3.02, ip_hi=3.37,
         ip_alt=1662.6 / 599.4, sv=848.0 / 328.4),
]

# Candidates examined and excluded, grouped by the reason. Each name is one candidate change (or one
# measurement of a change) that the four sweeps (CPU records, GPU records, the ledgers, the commit log) turned up.
EXCLUDED = [
    ("in-process A/B only: no served A/B of the change exists", [
        "L1 S0 per-group f16 kernel (superseded by the S0b row kernel before any served run)",
        "L2 R-06 default-on 09-27 (r06-gate3-ab.log 1.016/1.030/1.018; both gate-5 served builds have R-06 on)",
        "R-06 on the 7B, 09-20 (two process blocks, 0.9997x)",
        "R13 group-major acc64 attention (kernel 1.53-2.39x vs in-process decode 0.99x@2048 / 1.32x@8192, which its record "
        "calls 'served'; no bench_peer run)",
        "S-05 centering fold (kernel 1.284-1.296x, in-process token 1.048-1.097x)",
        "P14 W4A8 split-half on amd64 (kernel 1.12x, BenchmarkDecode +2.10%)",
        "P-07 gatedDeltaNetStep loop interchange (isolated loop 1.53x)",
        "sampler scratch reuse P8, 08-12 (microbench 5-6% slower, reverted)",
        "sampler scratch reuse P10, 08-19 (microbench; inside G26's anchor->HEAD row, not isolated)",
        "audit #9 sampler-scratch branch (525 -> 392 us/op microbench)",
        "R15 sampler max-scan fan-out (microbench, never shipped)",
        "KV re-gather strided P1", "CPU worker pool / spin-then-park", "per-shape matmul width (MC3c)",
        "q4k Phase 1a on CPU, PR #2 (BenchmarkDecode 0.977x / 1.308x; the peer comparison ran CUDA only)",
        "q4k lever 1 narrow-row kernel", "q4k lever 3 fused per-32, PR #4 (its ~212-215 tok/s is in-process or projected)",
        "Phi-3 CUDA q4k (1.257x / 1.166x; the peer comparison ran the 1.5B and 7B only)",
        "per-32 activations Track A / H2 actquant, PR #1",
        "R19 Metal prefill attention (served TTFT owed)", "R10 WebGPU rb64 GEMM", "R11 / P20 expert-major MoE prefill",
        "C' DMA overlap (its bench_peer row was 'the next measurement')", "CUDA graphs on the 26B", "C' UploadBatch",
        "R14 CUDA drafter argmax", "theta A/Bs (WebGPU, CUDA)", "non-copy n-gram fix", "CUDA batched DeltaNet prefill",
        "Metal L2 fused prefill attention", "WebGPU batched prefill TTFT", "CUDA prefill chunk demotion", "R5 CUDA tile-128",
        "G24 A3 f32 attention", "P19 fused prefill attention", "P18 CPU MoE expert-major", "A3 prefill head fan-out",
        "P12 qwen35 projections", "G16 prefill head threading", "Metal A1 coalescing", "CUDA A1 coalescing",
        "softcap parallel-for", "q8Span SIMD widen", "PGO", "Metal split-KV port (reverted)", "Metal ForwardArgmax wiring",
        "M-10", "aikit v1.17.0 / v1.17.1 bumps", "int8 per-Workspace decode threshold (b9b215da)", "G30 CUDA g4x2 clear",
        ".giw kind 4", "bounded top-p selection", "WebGPU DP4A", "CUDA split-KV q-staging (ncu only)",
        "CUDA autoresearch rounds (ncu only)", "split-KV warp-shuffle softmax (ncu only)", "C' pin-in-place (load time)",
        "Metal pread staging", "P2 lazy Z (refuted)", "G37 constrained-decode mask (computed, not measured served)",
        "CPU speculative decoding", "perf-campaign phases 1 and 3 (the served side was the chat demo, not HTTP)",
        "serve decode-threshold fix (BenchmarkDecode)", "fork/join removal (go tool trace)", "moeMLP scratch (no A/B at all)",
        "int4 W4A8 Workspace pool (no A/B at all)", "embedResident scratch (no A/B at all)",
        "audit microbenches P-02 / P-04 / P-13", "LoRA cache and grid P-10 / P-11", "MC2 on the Mac, MC3c on darwin",
        "July cgo-free spikes (Metal Stage B, CUDA)", "June aikit v0.5.0 (BenchmarkDecode vs the chat demo)",
    ]),
    ("served A/B only: no in-process A/B of the change exists", [
        "MC3 S2 sampled tokens join steps", "MC1 multi-slot KV on Metal", "MC1 on CUDA", "MC1 on WebGPU", "MC0 CPU LRU fix",
        "S6 Metal weight aliasing", "G35 + G36 combined", "P6a split-KV re-gate as shipped", "P1 top_k=1 routing",
        "tool-union grammar", "R6 default-on 09-23 (the lane itself is the R6 row)", "R6 multi-row verify lane",
        "J6 prefix-aware admission (its own harness)", "session-LRU fix", "Linux sidecar default (CUDA instrument unclear)",
        "R7 CUDA device top-K (served reads the new build only)", "R7b Gumbel draw (served reads the new build only)",
        "GOMEMLIMIT on M35",
    ]),
    ("both halves recorded, but not an A/B of one change, or not comparable", [
        "CPU decode Phase 0 baseline (157196bb vs 411e7fc4, many changes)",
        "L1 arm64 fix vs the merged 5c85f7c0 (derived across two served sessions; not independent of the L1 arm64 rows)",
        "R-06 on the Mac 09-03/04 (in-process instrument not identified: 'a paired, interleaved before/after A/B', e4e04992)",
        "S-01 v1.31.0 vs v1.34.0 stack comparison, de0cc654 (three aikit releases; the S-01 row uses 897fb18d's single bump)",
        "S-03/S-04 NEON (served 1.172x spans 14 aikit releases; the only in-process number is a pure-Go AV step)",
        "P20 expert-major on the W3 cell, 1.57x (the served cell also carries C' DMA overlap, 'not decomposed')",
        "R16 at K=3900 (served exists only as a ratio to Ollama across sessions)",
        "MC4 spec verify on MC3 step kernels (in-process is a cost curve, not a new-vs-old ratio)",
        "fit-by-default context vs the ctx-fit fix (in-process on dense models, served on MoE models)",
        "the retired '0.5B 1.78x' (an in-process 476.1 tok/s divided by Ollama's HTTP 268; not a change A/B)",
    ]),
]

NOTES = [
    "INSTRUMENT NOISE (kernel bench). The aikit scalar-widen arm read differently in two kernel runs on the same Mac 40 s apart: "
    "-count 8 medians 1.815 / 1.402 / 1.418 / 2.879 (scalar / f32 time) against the interleaved driver's 1.694 / 1.193 / 1.391 / "
    "3.117. The 1.5B gate/up shape moved from 1.40x to 1.19x slower. The fused arm moved far less (7B 0.870 -> 0.949, the rest "
    "within 0.007). One reading of a kernel arm is not fixed to within ~15% on this box by day.",
    "INSTRUMENT NOISE (served, n=2). On identical code the served harness can 'resolve' a ~2% effect: the 0.5B keeps the original "
    "kernels in both the fused_rms_qkv and the fused_rms_gu builds, and reads 0.9855 (runs 0.9816-0.9893, non-overlapping) and "
    "0.979 (0.9746-0.9834). Rows whose served resolution rests on a range-excluding effect below 2.2% are listed in the "
    "sensitivity line.",
    "R9's 1.37x / 1.10x is quoted by cpu-peer-reanchor-2026-09-22.md as 'paired A/B ... measured in-process'; the attribution record "
    "shows two separate runs. Only the per-knob 1.189x / 1.118x are paired.",
    "R13's record calls BenchmarkDecodeAtDepth / TestZZDiagGroupedFires 'served'. They are in-process; no served A/B exists for R13.",
    "LEVEL SHIFT vs RATIO. G35 read 118.4 tok/s in-process and 104.5 through the server on the same kernel; the retired '0.5B 1.78x' "
    "divided an in-process 476.1 by a served peer (the same code served at 320.1). The absolute level does not transfer; the "
    "question here is only whether the new/old RATIO does.",
]


def resolved(r, lo, hi, force, why, floor=0.0):
    if force is not None:
        return force, why or "forced by the record"
    if lo is not None and hi is not None:
        if abs(math.log(r)) < math.log(1 + floor):
            return False, f"range excludes 1 but |effect| < {100 * floor:.1f}%"
        if lo > 1.0:
            return True, "range above 1"
        if hi < 1.0:
            return True, "range below 1"
        return False, "range straddles 1"
    if abs(math.log(r)) >= math.log(1 + NOISE):
        return True, "no spread; |effect| >= 3.5%"
    return False, "no spread; |effect| < 3.5%"


def classify(row, sv_floor=0.0):
    ipr, ipw = resolved(row["ip"], row.get("ip_lo"), row.get("ip_hi"), row.get("ip_res"), row.get("ip_why"))
    svr, svw = resolved(row["sv"], row.get("sv_lo"), row.get("sv_hi"), row.get("sv_res"), row.get("sv_why"), sv_floor)
    raw_agree = (row["ip"] > 1) == (row["sv"] > 1) if abs(row["ip"] - 1) > 1e-12 else True
    verdict = ("agree" if raw_agree else "DISAGREE") if (ipr and svr) else "unresolved"
    k = math.log(row["sv"]) / math.log(row["ip"]) if abs(math.log(row["ip"])) > 1e-9 else float("nan")
    k_alt = math.log(row["sv"]) / math.log(row["ip_alt"]) if row.get("ip_alt") else None
    return dict(ipr=ipr, ipw=ipw, svr=svr, svw=svw, raw_agree=raw_agree, verdict=verdict, k=k, k_alt=k_alt,
                rr=row["sv"] / row["ip"])


def k_e2e(row, ch, cl):
    """k against the row's best whole-token (e2e) in-process reading, or None."""
    if ch["ip_class"] == "e2e":
        return cl["k"]
    if ch.get("ip_alt_class") == "e2e" and cl["k_alt"] is not None:
        return cl["k_alt"]
    return None


def mr(xs):
    xs = [x for x in xs if x is not None and not math.isnan(x)]
    if not xs:
        return "n/a"
    return f"median {statistics.median(xs):.2f}, range {min(xs):.2f} to {max(xs):.2f} (n={len(xs)})"


def main():
    for r in ROWS:
        assert r["c"] in CHANGES, r["c"]
    out = [(r, CHANGES[r["c"]], classify(r)) for r in ROWS]

    hdr = (f"{'change':<24} {'cell':<36} {'loc':<7} {'cls':<6} {'in-proc':>8} {'served':>8} {'sv/ip':>6} "
           f"{'k':>6} {'k_alt':>6}  verdict   (in-process side | served side)")
    for tier in (1, 2):
        label = "same-session served, single change" if tier == 1 else "cross-session / cross-tree / cross-condition served"
        print(f"\n=== TIER {tier}: {label} ===\n{hdr}")
        for r, ch, cl in out:
            if ch["tier"] != tier:
                continue
            ka = f"{cl['k_alt']:.2f}" if cl["k_alt"] is not None else ""
            kk = f"{cl['k']:.2f}" if not math.isnan(cl["k"]) else "n/a"
            sides = f"{'res' if cl['ipr'] else 'UNRES'}: {cl['ipw']} | {'res' if cl['svr'] else 'UNRES'}: {cl['svw']}"
            flag = "" if cl["raw_agree"] else "  [point estimates point opposite ways]"
            print(f"{r['c']:<24} {r['cell']:<36} {ch['locality']:<7} {ch['ip_class']:<6} {r['ip']:>8.3f} "
                  f"{r['sv']:>8.3f} {cl['rr']:>6.3f} {kk:>6} {ka:>6}  {cl['verdict']:<10}({sides}){flag}")

    def summary(label, rows):
        n = len(rows)
        chs = {r["c"] for r, _, _ in rows}
        ag = sum(k["verdict"] == "agree" for _, _, k in rows)
        dis = sum(k["verdict"] == "DISAGREE" for _, _, k in rows)
        un = n - ag - dis
        raw = sum(k["raw_agree"] for _, _, k in rows)
        print(f"\n--- {label}: {n} rows over {len(chs)} changes")
        print(f"    point-estimate sign agreement: {raw}/{n} ({100 * raw / n:.0f}%)")
        print(f"    agree {ag}, DISAGREE {dis}, unresolved {un}; resolved-only agreement {ag}/{ag + dis}"
              f" ({100 * ag / max(1, ag + dis):.0f}%)")
        for loc in ("kernel", "mixed", "outside"):
            sub = [(r, k) for r, c, k in rows if c["locality"] == loc]
            if sub:
                a = sum(k["verdict"] == "agree" for _, k in sub)
                d = sum(k["verdict"] == "DISAGREE" for _, k in sub)
                print(f"    locality {loc:<7}: {len(sub):>2} rows / {len({r['c'] for r, _ in sub}):>2} changes: "
                      f"agree {a}, DISAGREE {d}, unresolved {len(sub) - a - d}")
        agr = [(r, c, k) for r, c, k in rows if k["verdict"] == "agree"]
        kern = [k["k"] for _, c, k in agr if c["ip_class"] == "kernel"]
        e2e = [k_e2e(r, c, k) for r, c, k in agr]
        print(f"    k on AGREE rows, kernel-class in-process reading: {mr(kern)}")
        print(f"    k on AGREE rows, best whole-token in-process reading: {mr(e2e)}")
        rr_e2e = [r["sv"] / (r["ip"] if c["ip_class"] == "e2e" else r["ip_alt"]) for r, c, k in agr
                  if c["ip_class"] == "e2e" or (c.get("ip_alt_class") == "e2e" and r.get("ip_alt"))]
        print(f"    served / in-process ratio of ratios on AGREE rows, whole-token readings: {mr(rr_e2e)}")
        for loc in ("kernel", "mixed", "outside"):
            kl_ = [k_e2e(r, c, k) for r, c, k in agr if c["locality"] == loc]
            if any(x is not None for x in kl_):
                print(f"      whole-token k, locality {loc:<7}: {mr(kl_)}")
        return dict(n=n, ag=ag, dis=dis, un=un)

    t1 = [x for x in out if x[1]["tier"] == 1]
    t2 = [x for x in out if x[1]["tier"] == 2]
    summary("TIER 1", t1)
    summary("TIER 2", t2)
    summary("ALL", out)

    kl = [(r, k) for r, c, k in t1 if c["locality"] == "kernel"]
    strict_bad = [(r, k) for r, k in kl if k["verdict"] != "agree"]
    res_bad = [(r, k) for r, k in kl if k["verdict"] == "DISAGREE"]
    nch = len({r["c"] for r, _ in kl})
    print(f"\n=== TE5(b) RULE on tier 1, kernel-local changes: {len(kl)} rows over {nch} changes ===")
    print(f"  STRICT (an unresolved side counts as a disagreement): {len(strict_bad)} failing row(s) -> "
          f"{'KILL (b)' if strict_bad else 'no kill'}")
    for r, k in strict_bad:
        print(f"    {r['c']} {r['cell']}: in-process {r['ip']:.3f}, served {r['sv']:.3f}; {k['ipw']} | {k['svw']}")
    print(f"  RESOLVED-ONLY (both sides resolve a direction): {len(res_bad)} disagreeing row(s) -> "
          f"{'KILL (b)' if res_bad else 'no kill'}")
    for r, k in res_bad:
        print(f"    {r['c']} {r['cell']}: in-process {r['ip']:.3f}, served {r['sv']:.3f}")
    ag = [(r, c, k) for r, c, k in t1 if c["locality"] == "kernel" and k["verdict"] == "agree"]
    kk = [k["k"] for _, c, k in ag if c["ip_class"] == "kernel"]
    ke = [k_e2e(r, c, k) for r, c, k in ag]
    ke = [x for x in ke if x is not None]
    print(f"  magnitude, agreeing rows, kernel-class reading: k {mr(kk)}; "
          f"within 2x (0.5-2): {sum(0.5 <= x <= 2 for x in kk)}/{len(kk)}")
    print(f"  magnitude, agreeing rows, whole-token reading: k {mr(ke)}; "
          f"within 2x: {sum(0.5 <= x <= 2 for x in ke)}/{len(ke)}; within 1.25x (0.8-1.25): "
          f"{sum(0.8 <= x <= 1.25 for x in ke)}/{len(ke)}")

    print("\n=== every resolved sign disagreement, any tier or locality ===")
    for r, c, k in out:
        if k["verdict"] == "DISAGREE":
            print(f"  {r['c']} {r['cell']} [{c['locality']}, tier {c['tier']}]: in-process {r['ip']:.3f}, served {r['sv']:.3f}")
    print("=== every row whose point estimates point opposite ways ===")
    for r, c, k in out:
        if not k["raw_agree"]:
            print(f"  {r['c']} {r['cell']} [{c['locality']}, tier {c['tier']}]: in-process {r['ip']:.3f}, served "
                  f"{r['sv']:.3f} -> {k['verdict']}")

    print(f"\n=== sensitivity: served resolution also requiring |effect| >= {100 * DO_NOTHING_FLOOR:.1f}% "
          f"(the do-nothing reading) ===")
    for r, c, k in out:
        k2 = classify(r, DO_NOTHING_FLOOR)
        if k2["verdict"] != k["verdict"]:
            print(f"  {r['c']} {r['cell']} [{c['locality']}, tier {c['tier']}]: {k['verdict']} -> {k2['verdict']}")

    nex = sum(len(names) for _, names in EXCLUDED)
    print(f"\n=== candidates examined: {len(CHANGES) + nex} = {len(CHANGES)} included changes + {nex} excluded ===")
    for why, names in EXCLUDED:
        print(f"  excluded, {why} ({len(names)}):")
        for nm in names:
            print(f"    - {nm}")
    print("\n=== notes ===")
    for n in NOTES:
        print("  - " + n)
    print("\n=== sources ===")
    for cid, ch in CHANGES.items():
        print(f"  {cid} [{ch['locality']}, tier {ch['tier']}]: {ch['name']}\n      commits: {ch['commits']}; {ch['machine']}\n"
              f"      in-process ({ch['ip_class']}): {ch['ip_instr']} [{ch['ip_dir']}] <- {ch['ip_src']}")
        if ch.get("ip_alt_instr"):
            print(f"      in-process alt ({ch['ip_alt_class']}): {ch['ip_alt_instr']}")
        print(f"      served: {ch['sv_instr']} <- {ch['sv_src']}\n      caveat: {ch['caveat']}")


if __name__ == "__main__":
    main()
