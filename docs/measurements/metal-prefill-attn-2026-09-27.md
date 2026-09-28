# Metal prefill attention at depth (R19): S0 puts attention at 55% of a K=3900 prefill, at ~0.30 TFLOPS; the peers' kernels show why (2026-09-27)

R19 of [`red-october.md`](../tasks/red-october.md). After R16, Metal's K=512 TTFT is level with Ollama's (0.950), but
K=3900 reads 0.511 (8278 against 4225 ms,
[`metal-prefill-gemm-s2-2026-09-25.md`](metal-prefill-gemm-s2-2026-09-25.md) § "Measured through serve"). The band and
preconditions were registered in `red-october.md` § R19 before any prototype.

## S0: where a prefill's time goes, on the current build

`TestMetalPrefillDecomp` with leave-one-out:
- `GOINFER_METAL_DECOMP=1 GOINFER_METAL_DECOMP_K=512,3900 GOINFER_METAL_DECOMP_LOO=1`;
- qwen2.5-coder-1.5b int4 (the `.metal.giw`), M1 Pro;
- medians of 5 paired reps, in sequence;
- 2026-09-27 17:49–17:56 PDT.

The replica's logits are bit-identical to `PrefillLast`'s at both lengths. Log: [`s0-decomp.log`](metal-prefill-attn-2026-09-27/s0-decomp.log).

| in-sequence cost | K=512 | K=3900 |
|---|---:|---:|
| **attention (fused)** | 108.8 ms (18.3%) | **4429.3 ms (55.1%)** |
| GEMM qkv | — | 230.3 (2.9%), 2.99 TFLOPS |
| GEMM o (+residual) | — | 183.5 (2.3%), 2.81 |
| GEMM gate/up | 272.5 (45.7%), 2.90 TFLOPS | 2063.0 (25.7%), 2.92 |
| GEMM down (+residual) | 141.9 (23.8%), 2.78 | 1039.2 (12.9%), 2.90 |
| full replay (GPU) | 595.7 ms | 8040–8060 ms |

- **Attention does ~1.3 TFLOP at K=3900** (28 layers × 12 heads × causal 3900² × head dim 128, QKᵀ and PV) in 4.43 s,
  which is **~0.30 TFLOPS**. The same GPU runs these GEMMs at 2.8–3.0 TFLOPS.
- **Parity with Ollama's 4225 ms**, with today's GEMMs (3.53 s), needs attention at ~0.6–0.7 s: about **6.5×**.

## The prior-art read (no runs)

Three kernels were compared:
- **goinfer:** `metal/prefill.go` `attention_prefill_fused` (kernel at 334–444, dispatch at 885–931).
- **MLX 0.32.0:** `steel_attention` (Homebrew `include/mlx/backend/metal/kernels/steel/attn/`). At f16, head dim 128,
  it is instantiated as `steel_attention_float16_bq32_bk16_bd128_wm4_wn1` (kernel names in `mlx.metallib`).
- **ggml 0.22.0:** `kernel_flash_attn_ext`, read from the Metal source embedded in Homebrew's `libggml-metal.so`, as
  R16's read did. It is not committed here, being third-party source.

| | goinfer | MLX steel | ggml FA |
|---|---|---|---|
| query rows per threadgroup | 32: 4 independent simdgroups × 8 rows | 32: 4 simdgroups × 8 rows | 8 rows; 4 simdgroups split keys and output dims |
| K/V block | 32 keys | 16 keys | 64 keys |
| K/V staging | none: each simdgroup re-reads from device | Kᵀ and V in padded threadgroup memory, shared by 32 rows | none |
| O accumulator | threadgroup `oScr` (16 KB), rescaled by 8 lanes after every head-dim tile | registers, 8×8 fragments | threadgroup, pulled into registers once per block |
| row max / sum | 8 of 32 lanes, serial, `exp` | registers + 2 `simd_shuffle_xor`, `exp2` with log2e folded | 32 lanes × 2 columns, `simd_max` / `simd_sum` |
| causal | mask tested on every element of every block | blocks past the diagonal skipped; mask only on the diagonal | a block-map pre-pass |
| barriers per 64 keys | ~68 | 16 | 3 |
| threadgroup memory | 22.9 KB | 14.8 KB | 10 KB |

**Why goinfer is ~10× slower, ranked.** K/V reuse is last, since ggml reuses K/V as little as goinfer and is still fast.
1. **O goes through threadgroup memory on every head-dim tile.** The PV partial is stored, barriered, rescaled
   serially by 8 lanes, and barriered again: 16 times per 32 keys, against 128 MMAs of useful work.
2. **The softmax runs serially on a quarter of the lanes, through scratch.** The kernel's comment says a fragment has
   no per-row view. MLX works per row through the lane map (each lane holds 2 adjacent elements of one row). goinfer's
   MC3 kernels already rely on that map (`mc3_frag`), measured and checked on this GPU.
3. **A long serial chain with ~68 barriers per 64 keys.** Nothing overlaps within a simdgroup.
4. **22.9 KB of threadgroup memory per threadgroup** (16 KB of it O) limits how many threadgroups are resident. This
   is inferred, not measured.
5. **K/V are not staged.** MLX's 32-row staging is headroom beyond the first four.

**The template is MLX's steel attention.**
- Its strides fit goinfer's data directly: Q row stride = the packed qkv stride, K/V row stride = kvDim.
- It is f16 in and out.
- It handles causal attention analytically from a query offset, which is `startPos` here.
- Two things to adapt:
  - goinfer's sliding `window`, which MLX lacks: a lower block bound, and masking on the window's edge blocks;
  - ragged tails.
- An option neither peer takes: pack a GQA group's 6 query heads into one threadgroup, for 6× more K/V reuse.

## The prototype, confirmed and wired (R19 SHIPS)

**Exploratory run 1** ([`r19-explore1.log`](metal-prefill-attn-2026-09-27/r19-explore1.log)): the first prototype,
`attention_prefill_steel`, built from the read above.
- In sequence at K=3900 it read **7.32×** the fused kernel (4432 → 606 ms), and at K=512 6.25×.
- Against the fused kernel on the same inputs: relative L2 6.4e-5, cosine 0.999999998.
- It was selected as the candidate.

**The confirmation run** ([`r19-confirm-1.5b.log`](metal-prefill-attn-2026-09-27/r19-confirm-1.5b.log)): a fresh
process with the candidate alone, 7 paired reps, 2026-09-27 18:08–18:17 PDT.

| | fused | prototype | speedup |
|---|---:|---:|---:|
| **attention in sequence, K=3900 (graded)** | 4444.5 ms | 617.8 ms | **7.204×** (every pair 7.15–7.43×) |
| attention in sequence, K=512 | 110.0 ms | 16.6 ms | 6.62× |
| attention alone, K=3900: sustained / after 2 s idle | 4203.6 / 4205.4 ms | 576.0 / 583.1 ms | |

**Graded against R19's band: SHIP** (≥ 1.25×). The preconditions:
1. **Fidelity, set A's §3.2 pooled gate, passes on both models with the new kernel as production's prefill
   attention.** `TestPrefillGateVsReference` ran with the `.int4.metal.giw` bundles and the tokenizers from their
   `.gguf`s (a new `GOINFER_METAL_MODEL[_D7]_TOKENIZER` in the test), 2026-09-27 18:29–19:01 PDT
   ([`gate-setA-steel.log`](metal-prefill-attn-2026-09-27/gate-setA-steel.log)).
   - A first launch stopped at once: the test read its tokenizer from the model path, and a `.giw` has none
     ([`gate-setA-steel-attempt1-tokenizer.log`](metal-prefill-attn-2026-09-27/gate-setA-steel-attempt1-tokenizer.log)).

   | model | pooled verdict (K = 256 / 512 / 1024) | hard flips, fast vs exact | agreement, fast vs exact |
   |---|---|---|---|
   | S (1.5B) | **SHIPS** (critA, critB, critC true) | 10 vs 12 | 93.39% vs 92.76% |
   | D7 (7B) | **SHIPS** (critA, critB, critC true) | 35 vs 35 | 88.18% vs 87.71% |

   Per cell, mean KL to the CPU f32 reference, fast against exact:
   - S: 0.0327 / 0.0375 (K=256), 0.0326 / 0.0343 (512), 0.0391 / 0.0410 (1024), and at the K=3900 confirm cell
     0.0419 / 0.0457 with hard flips 5 / 7;
   - D7: 0.0761 / 0.0795, 0.0670 / 0.0689, 0.0793 / 0.0863.

   The prefill lane with the new kernel is at least as close to the reference as the exact sequential path, in every
   cell.
   - **The baseline run on the retired fused kernel was deferred** by the owner's next priority. It was stopped
     seconds in, and it is owed; it would also close R16's owed set-A run.
2. **No burst dependence:** 1.01 after idle at K=3900. At K=512 the new kernel is slower after idle (22.5 against
   16.4 ms), so the sustained number it is graded on is the better one.
3. **K=512 does not regress:** 6.62× faster.
4. **The do-nothing arm ran in the same session.**

**Wired into production** (`metal/prefill.go`): head dim 128 runs `attention_prefill_steel`, and every other head dim
keeps `attention_prefill_fused`. A test-only `prefillSteelAttnOff` restores the fused kernel for baselines. Checked on
the wired build ([`r19-wired.log`](metal-prefill-attn-2026-09-27/r19-wired.log),
[`metal-suite-r19.log`](metal-prefill-attn-2026-09-27/metal-suite-r19.log)):
- the decomposition replica still reproduces `PrefillLast`'s logits bit for bit at K=512 and 3900;
- production against the retired fused kernel reads 7.35× in sequence at K=3900;
- the tagged Metal suite: 183 pass, 0 fail;
- **`PrefillLast` wall at K=3900: 8171 → 4316 ms**, and at K=512: 596 → 526 ms.

**Owed:**
- **Served TTFT against Ollama** (`scripts/bench_peer_prefill.py`, cell h's protocol). In process, K=3900 now takes
  4.32 s against Ollama's served 4.23 s from 2026-09-25. That is a projection, not the served claim.
- The fused-kernel baseline gate run.
