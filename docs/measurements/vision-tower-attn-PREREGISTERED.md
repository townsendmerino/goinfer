# PRE-REGISTERED — R8 phase A: a fused non-causal attention kernel for the CUDA SigLIP tower

**Written 2026-09-21 BEFORE the kernel exists. Not edited after any result was seen.** Record: `vision-tower-mma-2026-*.md`. Brief: `docs/tasks/red-october.md` R8.

## Step 0 result that sets the order (measured, not assumed)
`TestVisionTowerTiming` (committed driver) on the real gemma-3-4b-it tower, 896^2 = 4096 patches, 27 layers, int8: **26.06 s** resident (3 runs 26.045-26.064; the 2026-09-08 record says 26.1), cosine vs the CPU int8 reference **0.914**.
ncu `gpu__time_duration`, one forward, 306 launches: **`attn_img_batched` 22.26 s = 87.1%** (27 launches, 824 ms each); **`gemv_w8a8_batched` 3.27 s = 12.8%** (163 launches, 20 ms each); layernorm/gelu/quant kernels 0.016 s total; kernel time 25.54 s of the 26.06 s wall, so the 55 host round-trip adds + im2col + transfers are ~0.5 s (2%).
The brief's build order (GEMM first, attention second, device add third) is therefore inverted by the measurement: **attention is the wall; the int8 GEMM is 12.8%; the device add is <= 2%.**

## Phase A: what is built
`cuda/attn_fused_vit.cu`: the R5 kernel shape (`attn_fused`, mma.sync m16n8k8 f16, online softmax, K/V staged per block) with (i) NO causal/window/sink logic (every row attends all M keys), (ii) head dim 72 zero-padded to 80 (padded K/V/Q lanes are 0 so they add nothing to any dot product; output lanes >= 72 are not stored), (iii) query-tile height BM a template parameter, arms BM in {64, 128} (BN = 64). New PTX module; `attn_fused.ptx` untouched. Wired into `VisionEncoder.ForwardPatches` in place of `attn_img_batched` (which stays, selectable by `GOINFER_CUDA_VISION_ATTN=exact` as the kill switch).
Precision note: like `attn_fused` this rounds Q, K, V and the softmax weights P to f16 (accumulating in f32); `attn_img_batched` is f32 throughout. That is a real numeric change, gated below, not assumed benign.

## Gates (all required before it is default)
1. **Kernel logic** (extend the `attn_fused` reference test style): worst per-row cosine >= 0.99999 vs an f64 reference computed on f16-rounded operands and walking the kernel's own tiles, for nH in {4, 16}, hd = 72, M in {17, 70, 130, 517}, both BM arms. Logic defects (a wrong mask edge, a padded lane leaking) fail here.
2. **BM arms bit-identical to each other** (non-causal, per-row order is BM-independent): `math.Float32bits` equal on those shapes. If not, the arms are compared and the choice is recorded.
3. **Tower level, on the real checkpoint, two input patterns** (the existing sin/cos pattern and a second one at different frequencies; the baseline arm's outputs saved before the kernel is wired): cosine(new resident, saved pre-change resident) **>= 0.98** on both, AND cosine(new resident, CPU int8 reference) **>= 0.894** (baseline 0.914 minus 0.02) on the first pattern and >= the baseline's own value minus 0.02 on the second. **0.95-0.98 vs the baseline output is AMBIGUOUS -> parked (not default)**, recorded with per-layer localisation as the next step. Below 0.95 is a fail.
4. The downstream `GenerateVL` first-token identity check the P6 record used: run if a harness exists at build time; if not, recorded as NOT RUN (it is not silently replaced by the cosine).

## Speed decision rule (tower seconds, `TestVisionTowerTiming`, median of 5 after a discarded warm-up, arms in separate processes alternating, idle box; the pre-change arm is the do-nothing arm)
- **Phase A ships as default** if all gates hold AND tower <= 13 s (>= 2.0x). R8's own band then reads the outcome: **<= 8 s ships the R8 goal, 8-13 s parked-for-the-goal (the kernel still ships; phase B is scoped), > 13 s killed** (attention was not the wall after all).
- Attention-alone projection, so a miss is a finding: if the fused kernel reaches the text kernel's ratio on its own (3.76x on attention), tower = 3.27 + 22.26/3.76 + 0.5 ~ 9.7 s; a kernel at the 128-row tile's 2.4x-over-that reading (~9x) would give ~6.2 s. These are projections, not results.
- **Phase B (int8 `mma.sync` GEMM, R8 build item 1) is scoped only if the tower is still > 8 s after phase A**, against the then-remaining ~3.3 s of GEMV; the device-side add (build item 3) is <= 2% and not started unless it becomes the largest remaining item.

## Not claimed
No served TTFT / Ollama `gemma3:4b` vision peer row from this phase (that is R8's separate instrument); nothing about Qwen2.5-VL's tower; no claim that f16 attention preserves downstream generation until gate 4 (or its explicit NOT RUN) is on the record.
