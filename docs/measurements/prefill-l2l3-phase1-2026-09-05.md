# CUDA prefill L2 — fused attention: SHIPS at 1.70× end to end (2026-09-05)

**`attn_fused` clears its pre-registered band. Attention category 3.76×, end to end 1.70× on
S at K=3900, against a ship bar of ≥1.4×.** The kernel is correct against an independent
reference, the win is attributable (the gemv category is unchanged), and Amdahl accounts for the
whole distance between the two ratios. It is **NOT the default**: it ships behind
`GOINFER_CUDA_FAST_PREFILL=1` and stays there until §3's reference gate runs on CUDA (Phase 3).

## Provenance

| | |
|---|---|
| box | `nobara-pc`, RTX 2070 SUPER (8 GB, sm_75), driver **595.91.07**, Nobara 44, kernel 7.2.0-202.fc44 |
| goinfer | `9eb6ef43` + this change; baseline arm measured at `9eb6ef43` production code (only the test-only TTFT env knobs were applied at baseline time) |
| model | **S** = `~/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf`, int4, 28L nH=12 nKV=2 hd=128 — **local NVMe**, not `/srv/models` |
| second model | **D7** = `~/models/qwen2.5-7b-instruct-q4_k_m.gguf`, int4, 28L nH=28 nKV=4 hd=128 inter=18944 |
| harness | `cuda/prefill_decomp_test.go` (categories), `cuda/prefill_ttft_test.go` (end to end) |
| method | same session, arms ~15 min apart, box otherwise idle; best-of-3 per cell (the harnesses' own rule) |
| thermal | GPU 51–55 °C throughout, 1905 MHz, no clock drop; loadavg 0.05–2.4 |
| logs | `~/goinfer-logs/prefill-l2l3/p0-*.log`, `p1-*.log`, `ncu-attn.csv` (durable, outside the worktree) |

## 1. The result

**Attention category** (`TestPrefillDecomp`), and the **gemv category beside it as a control** —
L2 does not touch the weight term, so gemv must not move, and it does not:

| K | attn before | attn after | **attn ratio** | gemv before | gemv after | catSum before → after |
|---|---|---|---|---|---|---|
| 128 | 4.176 ms | 2.170 ms | 1.92× | 74.68 ms | 75.38 ms | 83.9 → 82.9 ms |
| 512 | 53.43 ms | 12.95 ms | **4.13×** | 307.2 ms | 309.0 ms | 373.2 → 334.1 ms |
| 2048 | 831.6 ms | 210.1 ms | **3.96×** | 1.238 s | 1.237 s | 2.122 → 1.497 s |
| **3900** | **3.025 s** | **805.2 ms** | **3.76×** | 2.353 s | 2.351 s | 5.480 → **3.255 s** |

**End to end** (`TestPrefillTTFT`, batched arm) — the cell the band is written on:

| model | K | before | after | **e2e ratio** | band |
|---|---|---|---|---|---|
| S | 128 | 80.91 ms | 85.98 ms | **0.94×** | — (see §3) |
| S | 512 | 371.5 ms | 329.9 ms | 1.13× | — |
| S | 2048 | 2.099 s | 1.471 s | 1.43× | — |
| **S** | **3900** | **5.451 s** | **3.213 s** | **1.70×** | **SHIPS** (≥1.4×) |
| D7 | 512 | 1.428 s | 1.323 s | 1.08× | second model, not a gate |
| D7 | 2048 | 7.304 s | 5.646 s | 1.29× | second model, not a gate |

**The two instruments agree.** From the decomposition, attention saves 3.025 − 0.805 = 2.220 s, so
end to end should be 5.451 − 2.220 = 3.231 s (1.69×). `TestPrefillTTFT` measured **3.213 s
(1.70×)** independently. Amdahl explains the whole gap between 3.76× and 1.70×: attention was
55.2% of prefill, and 1/(0.448 + 0.552/3.76) = 1.70.

**D7 wins less, for a structural reason, not a defect.** Its `inter=18944` FFN makes the weight
term a larger share, so there is less attention to remove. This is the same Amdahl arithmetic
pointing the other way, and it is why L3 — not more L2 tuning — is what moves D7.

## 2. Correctness

`TestAttnFused_vsExact` (150 seam cases: hd ∈ {64,128} × M ∈ {1,7,64,65,512} × startPos ∈
{0,5,1024} × {mha, gqa, window, sinks, window+sinks+gqa}) and `TestAttnFused_vsF16Reference`
(6 shapes) and `TestAttnFused_attendsKeysBeforeStartPos`: **156/156 pass.**

The logic gate is the one that matters: `attn_fused` scored against **exact f64 arithmetic on its
own inputs rounded to f16** — the operands it actually receives — lands at cosine **0.99999707 to
1.00000000** across every shape, multi-tile included. Operand precision is factored out by
construction rather than budgeted for, so anything left would be logic.

Against `attn_batched` the worst per-(row,head) deviation is **4.86e-4 of max |V|**, under the
1e-3 bar.

### 2.1 The pre-registered tolerance was mis-derived, and both "defects" were in the test

Recorded because the shape of the mistake is more reusable than the result:

**First bar, wrong denominator.** The original bar was "max |Δ| ≤ 1e-3 of *the row's own* max
|ctx|". It failed widely — worst cosine 0.9642, worst row-relative delta 1.6. The kernel was fine.
With synthetic V of quasi-random sign the context vector is a near-cancelling weighted average, and
measured here `|ctx|/rms|V|` ran from **1.02 down to 0.000211** — a cancellation to one part in
4700. The cosine gap tracked that conditioning **monotonically** (1.02 → 0.99999997; 0.0099 →
0.99996; 0.00021 → 0.9642). Rounding the *inputs alone* rotates such a row 27%, so no kernel of any
quality could meet a bar scaled by that row's own |ctx|. The bar is now relative to **max |V|**,
which is the scale the output is drawn from and does not collapse; cosine is retained but applied
only to rows above a stated conditioning floor, with the excluded count reported so it cannot
silently become a bar that tests nothing.

**Second, the reference did not model the algorithm.** With the conditioning confound removed, the
multi-tile cases still read 0.9992–0.9994 against a **global-max** f16 reference while single-tile
cases read 0.99999996. That reads exactly like an online-rescale defect. It was not: the kernel
rounds each tile's *provisional* weights `exp(s − m_running)` to f16 and then rescales the f32
accumulator, so the f16 rounding happens at a different scale than a global-max reference applies.
Rewriting the reference to walk the same 64-key tiles in the same order with the same running state
moved those cells to **0.99999979 / 1.00000000**.

**The general lesson, which is CLAUDE.md's minimal-repro rule running in reverse.** That rule warns
a shrunk repro can be minimal in the dimension that HIDES a defect. Here a synthetic repro was
pathological in a dimension that MANUFACTURED one — twice — and each time the false signal was
specific, reproducible, and looked mechanistic. What settled it both times was an oracle that
shares the candidate's own approximations: comparing two things that differ in more than one way
tells you the distance, never which one is wrong. That is §3.1's correction applied at the kernel
level rather than the model level.

## 3. The negative, at full value: K=128 is 0.94×, and there must be a floor

At K=128 the fused path is **slower — 80.91 ms → 85.98 ms**. It is not noise-free at that scale
(±3 ms is ~3.5% drift on this box) but the direction is consistent with the mechanism: attention is
**5.0%** of prefill at K=128, so there is nothing to win, while a 64-row query tile still stages
64 keys' worth of K/V per block. The win begins by K=512 (1.13×) and grows monotonically.

So the default, when it comes, **must engage above a prompt-length floor** — exactly as §3's Floor
clause requires and as `--cpu-fast-attention` already does at 512. On this evidence the floor sits
between 128 and 512; it is set from data in Phase 3, not picked here.

## 3.1 Greedy stream divergence: 0% → 80%. Reported, NOT gating, and here is why that is not a dodge

`TestPrefillDivergenceRate` (real S, 50 prompts × 48-token prompt × 128-token greedy generation),
batched-prefill-then-decode vs sequential-prefill-then-decode:

| arm | streams diverged | first divergent token |
|---|---|---|
| `GOINFER_CUDA_FAST_PREFILL=0` (exact) | **0 / 50 (0%)** | — |
| `GOINFER_CUDA_FAST_PREFILL=1` (fused) | **40 / 50 (80%)** | min 0, mean 19.3 of 128 |

**This is the expected consequence of the kernel being non-bit-identical, and §3 makes it
explicitly non-gating** — "it measures reproducibility, not quality". It is the same measure that
wrongly gated Metal's batched prefill at 54% (`metal/backend.go:236`), which §2.3 and §3.1 record as
the wrong gate for exactly this reason: CUDA decode is not held to it either, only to the 3%
near-tie parity rule.

Two things stop that from being a convenient dismissal:

1. **The 0% arm proves the instrument works.** The exact path shows zero divergence over the same
   50 streams, so an 80% reading is a real property of the fused kernel and not a noisy harness.
2. **It is measured at a prompt length the floor would exclude.** The test uses 48-token prompts.
   §3 requires the fast path to engage only above a measured floor, and §3 of this doc puts that
   floor between 128 and 512 — so in a default-on world this configuration would run the EXACT
   path. The divergence number that matters for shipping is the one at prompt lengths above the
   floor, under §3's reference gate, and that is Phase 3's measurement, not this one.

**What this does NOT establish is whether the fused path is as GOOD, only that it is different.**
That question has one answer and it is Phase 3: both arms scored against a CPU f32-activation
reference. Nothing here should be read as evidence either way.

## 4. ncu, for the record — the win is real and the kernel is nowhere near a ceiling

`attn_fused_hd128`, K=2048, grid (12, 8), block 128, median of 3 launches:

| metric | value |
|---|---|
| tensor pipe utilisation | **1.72%** of peak |
| achieved occupancy (warps active) | **12.6%** |
| SM throughput | 6.98% |
| DRAM throughput | 8.3% |
| L1TEX throughput | 9.8% |

**Nothing is saturated.** The kernel is still latency-shaped, just far less so than `attn_batched`.
The most likely first cause is grid size, and it is arithmetic rather than a guess: the grid is
12 × 8 = **96 blocks**, and at 178 registers/thread only **2 blocks fit per SM**, so 40 SMs hold 80
concurrently — **1.2 waves**, with a large tail. A 32-row query tile would double the block count.

**Not acted on here, deliberately.** The band is met; tuning a kernel before its band is met is how
the sibling GEMV accumulated five recorded misattributions. This is a named lever for later with a
counted reason, not a conclusion.

## 5. What this does not claim

- **The default has not changed.** `GOINFER_CUDA_FAST_PREFILL` is off unless set; `attn_batched`
  remains the exact path, is what spec-decode verify and the parity gates run, and serves every
  shape the fused kernel declines (hd ∉ {64,128}, M < 16, module not loaded).
- **1.70× is one model, one card, one quant, one depth.** D7 is 1.29× at its deepest measured cell.
- **No fidelity claim.** §3's reference gate has not been run on CUDA. Nothing here says the fused
  path is as good as the exact path on real prompts; it says it is arithmetically correct against
  its own operands and 1.70× faster. Those are different claims and Phase 3 owns the second.
- The ncu figures are three launches of one kernel at one depth, not a sweep.
