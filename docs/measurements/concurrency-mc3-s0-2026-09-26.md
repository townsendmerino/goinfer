# MC3 S0 — a bit-identical matrix-unit batched GEMV on Metal: 8 sequences for 1.0–2.7 GEMVs, every output bit-equal to production (2026-09-26)

S0 (exploratory, no graded claim) for MC3 of [`task-concurrency-2026-09.md`](../tasks/task-concurrency-2026-09.md),
Metal batched decode. The earlier S0 ([`concurrency-mc2-2026-09-26.md`](concurrency-mc2-2026-09-26.md), MC3 S0) found
that bit-identical *scalar* small-M kernels barely amortise (≤ 1.30× on gate/up at M = 4). It concluded that a paying
MC3 needed the matrix units and a fidelity gate. **This S0 removes the second half of that conclusion.** A
`simdgroup_matrix` kernel can reproduce the production decode GEMV's arithmetic exactly, and it pays.

**Result.**
- Test-only 8-row kernels cover all five decode matmul categories: qkv, o, gate/up, down and the int8 LM head.
- Each carries 8 sequences for **1.0–2.7× the time of one production GEMV**, so 8 sequences run **2.9–8.0× faster
  than 8 sequential GEMVs**.
- **Every output is bit-identical to production's GEMV on the same row, on every shape of the 1.5B and 7B.**
- So MC3 can keep MC2's identity gate, and no fidelity gate is needed.
- One structural fact limits small batches: an 8 × 8 fragment costs the same at M = 2 as at M = 8.

## Setup

- M1 Pro 16 GB, macOS 26.6.2, goinfer `1b5b8b5e` plus the test files.
- Standalone per-dispatch GPU time, median of 7.
- The S0 bench's SLC-defeating weight rotation: at least 256 MB of weight copies cycled. The LM head exceeds the SLC on
  its own and runs unrotated.
- Random int4 weights with f16 group scales and random int8 activations with per-row scales, as production quantises.
- Baselines:
  - production's decode kernels on one row: `gemv_w4a8_sa_rows<R>` (the R18b H form, with production's R: qkv 4,
    o 2, gate/up 4), `gemv_w4a8_resid_staged4` (down) and `gemv_w8a8_coal` (LM head);
  - "× 8 GEMVs" is that time × 8, what 8 sequences cost today.
- Machine at load1 1.5–2.5 throughout. These are exploratory numbers, not graded.

Tests: `TestMC3S0PrefillGEMMSmallM` (`metal/gemm_smallm_mc3_test.go`); `TestMC3MMA8`, `TestMC3MMA8Down`,
`TestMC3MMA8LMHead` (`metal/gemm_mma8_mc3_test.go`); `TestMC3SimdSumTree` (`metal/simdsum_tree_mc3_test.go`).
All are gated on `GOINFER_METAL_MC3=1`. Logs: [`concurrency-mc3-s0-2026-09-26/`](concurrency-mc3-s0-2026-09-26/).

## 1. The prefill GEMM does not fit: flat from M = 1 to 64

`gemm_w4f16_store`, fast prefill's f16 MMA GEMM, runs every one of its 64-token tile's MMAs regardless of M. Its time
is flat across M = 1–64 at **12–14× one decode GEMV**. It overtakes M sequential GEMVs only at M ≈ 13–16.
Example, 1.5B gate/up: 1.29–1.34 ms at every M, against a 0.108 ms GEMV. It is also f16-activation numerics. It is not
a starting point ([`mc3-s0-mma.log`](concurrency-mc3-s0-2026-09-26/mc3-s0-mma.log)).

## 2. An 8-row kernel, operands built in registers

`mc3_mma`'s output tile is one 8-token fragment column: C^T[feature][token] = W[feature][k] · A^T[k][token].
- **Weights never touch threadgroup memory.** The left operand is written straight into registers through
  `simdgroup_matrix::thread_elements()`. Lane (fm, fn) holds feature fm at k = fn and fn + 1, two adjacent nibbles of
  one packed word. The lane→element mapping is MLX's, and the test checks it on the device first.
- **Group sums are exact.** Weights enter as nibble − 8 and activations as the int8 quantised values, both exact in
  half. A 32-k group's four MMAs therefore accumulate the integer group sum in f32 exactly (|gi| ≤ 32·8·127 < 2²⁴).
  Each lane then applies `acc += float(gi) * scale`, production's own per-group product.

The first cut sums groups in plain order, so it matches production to f32 summation-order rounding (relative error
≤ 5e-7, 20–40% of outputs bit-equal). It measures what the route costs
([`mc3-mma8-explore1.log`](concurrency-mc3-s0-2026-09-26/mc3-mma8-explore1.log)):

| shape (N × K) | production GEMV | best 8-row variant | 8-row ÷ GEMV | **vs 8 GEMVs** |
|---|---:|---:|---:|---:|
| 1.5B qkv 2048 × 1536 | 0.020 ms | 0.035 (fb4/s4) | 1.7 | **4.67×** |
| 1.5B o 1536 × 1536 | 0.017 | 0.027 (fb2/s4) | 1.5 | **5.16×** |
| 1.5B gate/up 17920 × 1536 | 0.108 | 0.217 (fb4/s4) | 2.0 | **3.99×** |
| 7B qkv 4608 × 3584 | 0.068 | 0.152 (fb4/s4) | 2.3 | **3.55×** |
| 7B o 3584 × 3584 | 0.061 | 0.116 (fb4/s4) | 1.9 | **4.18×** |
| 7B gate/up 37888 × 3584 | 0.434 | 1.063 (fb4/s4) | 2.5 | **3.26×** |

(fbF/sS: F feature blocks of 8 per simdgroup; S simdgroups split K.)

## 3. Bit-identical: reproduce production's reduction, not just its products

Production's lane L sums groups g = L, L + 32, … as `acc += float(gi) * scale`, then `simd_sum`s the 32 lanes. Two
kernels reproduce that exactly:
- **`mc3_mma_bi`** gives each residue r = g mod 32 its own partial with the same chain. It transposes the partials
  through threadgroup memory and `simd_sum`s each output with residue L on lane L: production's own reduction input.
  It is robust by construction and costs one `simd_sum` per output.
- **`mc3_mma_bt`** does the reduction as `simd_sum`'s own addition tree:
  - `TestMC3SimdSumTree` measured that tree **on this GPU** as exactly the xor butterfly 1, 2, 4, 8, 16. Over 65,536
    random 32-lane vectors spanning 24 binades, it differed on 0. The 16, 8, 4, 2, 1 butterfly differed on 62% and a
    sequential sum on 73% ([`simdsum-tree.log`](concurrency-mc3-s0-2026-09-26/simdsum-tree.log)).
  - Residues map r = 8·simdgroup + slot. Levels ^1/^2/^4 pair slots in registers, and ^8/^16 pair simdgroups through a
    256-byte exchange (`#pragma clang fp reassociate(off)` on both).
  - Adjacent slots are adjacent groups, so a lane's weight loads walk a row contiguously.
- **`mc3_mma_btd`**, the down projection: production's down kernel sums per 8-k *word*, not per group. So each k
  block gets its own zero-initialised fragment (that MMA is exactly the word's integer sum), and residues are words.
- **`mc3_lm_mma`**, the int8 LM head: production sums the whole row as one int32. A 32-k slab's four MMAs stay below
  2²⁴ (32·128·128), so each lane converts the slab's exact integer and adds it to an int32. That gives the same integer
  in any order, so the logits are **bit-identical by construction**. Inside a slab the k order is permuted identically
  in both operands, so each lane's weights are 8 contiguous bytes.

Every variant below reported **0 differing outputs** out of 8 × N on every shape
([`explore3`](concurrency-mc3-s0-2026-09-26/mc3-mma8-explore3.log),
[`down`](concurrency-mc3-s0-2026-09-26/mc3-mma8-down-explore1.log),
[`lmhead`](concurrency-mc3-s0-2026-09-26/mc3-mma8-lmhead-explore1.log)):

| shape | production GEMV | bit-identical 8-row (best) | 8-row ÷ GEMV | **vs 8 GEMVs** | `mc3_mma_bi` (robust) |
|---|---:|---:|---:|---:|---:|
| 1.5B qkv | 0.021 ms | 0.037 (bt fb2) | 1.8 | **4.48×** | 2.76× |
| 1.5B o | 0.017 | 0.029 (bt fb2) | 1.7 | **4.68×** | 2.95× |
| 1.5B gate/up | 0.112 | 0.244 (bt fb2) | 2.2 | **3.67×** | 2.13× |
| 1.5B down 1536 × 8960 | 0.090 | 0.138 (btd fb2) | 1.5 | **5.23×** | — |
| 1.5B LM head 151936 × 1536 | 1.457 | 1.517 (lm fb4) | 1.04 | **7.68×** | — |
| 7B qkv | 0.068 | 0.155 (bt fb2) | 2.3 | **3.51×** | 2.49× |
| 7B o | 0.061 | 0.115 (bt fb2) | 1.9 | **4.24×** | 2.84× |
| 7B gate/up | 0.435 | 1.167 (bt fb4) | 2.7 | **2.98×** | 2.06× |
| 7B down 3584 × 18944 | 0.345 | 0.579 (btd fb4) | 1.7 | **4.77×** | — |
| 7B LM head 152064 × 3584 | 3.542 | 3.533 (lm fb4) | 1.00 | **8.02×** | — |

Reproducing production's order costs 0–12% against the order-free first cut, which ran in a separate process (e.g.
1.5B gate/up 0.244 against 0.217 ms). The robust `simd_sum`-per-output form costs 41–72% more than `mc3_mma_bt`.

**The one fragility.** `mc3_mma_bt` and `mc3_mma_btd` are bit-identical only where `simd_sum` is that butterfly. It was
measured on the M1 Pro only; other Apple GPU generations are unmeasured. A production kernel therefore gates itself:
- run `TestMC3SimdSumTree`'s comparison once at build time;
- fall back to the `mc3_mma_bi` form (robust, 2.1–3.0×) where the tree differs.

The LM head needs no such gate, because it is integer-exact.

## 4. What the fragment shape costs small batches

An 8 × 8 fragment computes 8 token columns whether or not 8 sequences exist. So a kernel above costs the same at
M = 2, 4 and 8, and its gain is M / (8-row ÷ GEMV):
- **M = 4:** 1.5–2.7× on the int4 projections, 3.9× on the LM head.
- **M = 2:** 0.7–1.3×, a loss on the larger shapes. At M = 2 the 1.5B's projections net to about even (gate/up 0.91×,
  down 1.33×), and the LM head gains 1.9×.

There is no way to pack two k ranges into one fragment's columns, since they share the left operand. So below M ≈ 3 a
batched step should either use production's per-row kernels for the int4 projections, or accept the wash.

## 5. Projected whole token (a projection, NOT a measurement)

The 1.5B at depth 128, from R18b's wired production figures
([`metal-decode-gemv-r18b-2026-09-26.md`](metal-decode-gemv-r18b-2026-09-26.md)):
- full token 10.46 ms (GPU);
- int4 GEMV work 7.39 ms, of which gate/up is 3.29;
- LM head 1.46 ms (standalone here);
- the remaining 1.61 ms is assumed to be entirely per-sequence (attention, norms, quantisation, argmax): an upper
  bound.

The cost ratios are the bit-identical ones above, taken standalone. R18 and R18b both found standalone gains that
in-sequence grading moved, so this is an estimate to plan by:

| batch | step time | aggregate vs 1 stream | step ÷ single step |
|---:|---:|---:|---:|
| 2 | ~18.4 ms | ~1.14× | ~1.76 |
| **4** | **~21.6 ms** | **~1.94×** | **~2.07** |
| 8 | ~28.1 ms | ~2.98× | ~2.69 |

Depth erodes this, as on the CPU. Per-sequence attention does not amortise: at 3900 the 1.5B's non-GEMV time per
sequence is ~3.9 ms against 1.6 at depth 128.

Against MC3's registered gates:
- **Identity** (MC2's choice, kept): the kernels meet it. The rest of the step can run production's own per-row
  kernels, as MC2 did on the CPU.
- **W7 4-client aggregate ≥ 1.2× MC1's:** a B = 4 step at ~1.9× one stream projects to clearing it.
- **p99 ≤ 1.5× MC1's 1-client figure (hard):** a B = 4 step costs ~2.1× a single step. Each batched request then
  takes ~2× a lone one, which **projects to failing the bar as written**.
  - Under the same 4-client load, MC1's p99 is ~4× a lone request, because it serialises. MC3 projects to about half
    of that.
  - MC3c met the same J8 bar at the same offered load (0.51–0.52×) and failed it against a lone request (2.0–2.2×).
  - The reading the gate is held to is the owner's call, before the build.

## Prior art (the sweep MC3's registration requires)

Read from source, MLX [`09e67c68`](https://github.com/ml-explore/mlx/tree/09e67c68) and llama.cpp
[`2b129ccf`](https://github.com/ggml-org/llama.cpp/tree/2b129ccf), 2026-09-27 UTC.

**MLX.**
- `QuantizedMatmul::eval_gpu` (backend/metal/quantized.cpp) takes the vector path below `get_qmv_batch_limit(K, N)`.
  By GPU generation and K,N bucket:
  - 13–14 non-Ultra: 14 / 10 / 6;
  - 15–16: 13 / 15 / 13;
  - ≥ 17: 33 / 25 / 13.
- Below the limit:
  - **`qmv_fast`** runs one activation row per threadgroup, re-reading the weights for every row.
  - **`qmv_wide`**, adapted from llama.cpp's `mul_mv_ext` and enabled on generation ≥ 15 only, shares register
    dequantisation across tiles of ≤ 5 rows.
- Above the limit, `qmm_t` / `qmm_splitk` run 32 × 32 × 32 `simdgroup_matrix` tiles, with weights dequantised into
  threadgroup memory.
- MLX PR #3791 puts the vector/MMA crossover at M ≈ 10–13 on M3 Max and M5 Max for large shapes.

**llama.cpp** (ggml-metal).
- `kernel_mul_mv_ext_*` covers ne11 = 2–8 (4–8 for K-quants). Its weights are dequantised into registers and shared
  across up to 5 columns, with no matrix units.
- `kernel_mul_mm` (MMA with a 64 × 32 tile, weights staged in threadgroup memory) takes over only above ne11 = 8.
- An open PR (#25377, M4 Pro) measures a **64 × 8 MMA tile beating `mul_mv_ext` from bs = 5**: 1.59× at 5, 2.02×
  at 8. That is the same regime S0 finds.

**What S0's kernels do that neither does.**
- Neither project builds a weight fragment in registers. MLX's `thread_elements()` use (steel `BaseMMAFrag`, the lane
  mapping used here) always loads fragments from staged threadgroup memory, and llama.cpp never calls it.
- Neither is built to match its own single-row kernel bit for bit. Their batched kernels are separate kernels with
  their own summation order. Whether their outputs happen to agree was not checked.
- Both switch to matrix units late (M > 8 to 13+) because their tiles are 32 or 64 rows. An 8-row tile is what moves
  the crossover to M ≈ 3 here.

## What S0 decided and did not

- **Decided:** the matrix-unit route pays and is bit-identical, so MC3 keeps the identity gate. The kernel design is
  in-register operands, exact integer fragments, and production's reduction order reproduced.
- **Not measured:**
  - in-sequence cost;
  - multi-sequence attention over MC1's slots;
  - batched norm, rope and quantisation;
  - scheduling;
  - any end-to-end number;
  - the tree on other Apple GPUs.
