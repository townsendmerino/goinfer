# R8 phase A: fused non-causal attention for the CUDA SigLIP tower — 26.0 s -> 4.1 s per image, opt-in pending a fidelity call

Pre-registration: `vision-tower-attn-PREREGISTERED.md` (written before the kernel existed). RTX 2070 SUPER, driver 595.91.07, idle box, real `gemma-3-4b-it` tower (896^2 = 4096 patches, 27 layers, int8 both arms), committed driver `cuda/vision_tower_timing_test.go`. Logs: `vision-tower-baseline-2026-09-21.log`, `vision-tower-attn-2026-09-21.log`, per-kernel ncu `vision-tower-ncu-baseline-2026-09-21.csv`.

## Step 0: the brief's build order was inverted by the profile
ncu, one forward, 306 launches, 25.54 s of kernel time in a 26.06 s wall: **`attn_img_batched` 22.26 s (87.1%)**, `gemv_w8a8_batched` 3.27 s (12.8%), layernorm/gelu/quant 0.016 s, host round-trip adds + im2col + transfers ~0.5 s (2%). The brief listed the int8 GEMM first; attention is the wall.

## Speed (tower seconds, median; arms in separate processes, exact/bm64/bm128 alternating, two rounds; spread < 0.02 s)

| arm | run 1 | run 2 | speedup |
|---|---:|---:|---:|
| `attn_img_batched` (default today) | 25.959 s | 25.956 s | 1.00x |
| fused, BM=64 | 4.200 | 4.198 | 6.18x |
| **fused, BM=128** | **4.081** | **4.079** | **6.36x** |

**R8's registered band (tower seconds): <= 8 s ships -> 4.08 s is inside it, with 3.9 s of margin.** The attention-alone projection in the pre-registration was 6.2-9.7 s; the measured 4.08 s beats it. Remaining tower time ~4.1 s: GEMV 3.27 s (80%) + attention ~0.3-0.4 s + ~0.5 s host round-trips. The tower is now GEMV-bound.

## Fidelity gates (pre-registered)
1. **Kernel logic: PASS.** `TestAttnVit_logicAndBMIdentity`: worst per-row cosine vs the f64 reference on f16-rounded operands >= 0.99999972 on all 8 shapes (nH 4/16, M 17/70/130/517, hd 72 padded to 80; output buffer pre-filled with NaN so an unwritten lane would fail).
2. **BM arms bit-identical: PASS** (bm64 == bm128 on all shapes, and identical tower outputs).
3. **Tower level: AMBIGUOUS -> PARKED under the registered rule.** cosine(new, pre-change resident output) = **0.9617** (pattern A) and **0.9661** (pattern B); the registered pass line is >= 0.98, the fail line < 0.95, and 0.95-0.98 is "ambiguous, parked (not default)". max|diff| 31.0 / 23.5.
   Against the CPU int8 reference the new kernel is **as close as the old one**: pattern A 0.9133 (old 0.9141, floor 0.894), pattern B 0.9295 (old 0.9327). So the second half of gate 3 passes and the new output is NOT further from the reference than the shipped one; what the parked half measures is that the two GPU arms differ from each other by more than 0.02 in cosine, as expected of a 27-layer int8 tower where the P6 record already attributes cosine 0.91-0.96 to accumulated per-layer rounding (f16 attention operands are a new rounding source at every layer).
   I have NOT localised the 0.96 to a layer (the registered next step) and have NOT shown it is benign; the Metal R2 root-cause record (`r2-attn-fa-rootcause-2026-09-21.md`) is the precedent for exactly this shape (an int8 activation-quantisation crossing amplified through a deep tower), and it is a precedent, not evidence about this kernel.
4. **Downstream `GenerateVL` first-token identity: NOT RUN.** No committed harness exists (the P6 check was a throwaway); recorded as not run, not replaced by the cosine.

## The fidelity difference, quantified (from the saved outputs, `vision-tower-fidelity-analysis.py`; 4096 rows x 1152, both patterns)

The output has a heavy-tailed scale (rms 0.40, abs max 52.7 on pattern A): a few outlier channels (dims 725, 961, 105, 249, 502) carry most of the energy, so **the global cosine is an energy-weighted measure dominated by them**; the per-row view is the better read.

| | pattern A | pattern B |
|---|---:|---:|
| global cosine old~CPU / new~CPU / new~old | 0.9141 / 0.9133 / 0.9617 | 0.9327 / 0.9295 / 0.9661 |
| relative L2 error old-vs-CPU / new-vs-CPU / **new-vs-old** | 0.413 / 0.417 / **0.279** | 0.367 / 0.374 / **0.259** |
| per-row cosine vs CPU, median (old / new) | 0.992 / 0.992 | 0.990 / 0.990 |
| per-row cosine vs CPU, p5 (old / new) | 0.870 / 0.851 | 0.921 / 0.907 |
| per-row cosine new-vs-old: p5 / median | 0.955 / 0.996 | 0.955 / 0.995 |
| rows where new is closer to the CPU reference than old | 49.7% | 48.1% |
| mean per-row cosine vs CPU, paired new - old (+- s.e.) | -0.0009 +- 0.0006 | -0.0002 +- 0.0006 |
| max abs diff new-vs-old / old-vs-CPU | 31.0 / 31.1 | 23.5 / 29.6 |

What it says, and what it does not:
- **The two GPU arms differ from each other by LESS than the shipped arm already differs from the CPU reference** (relative L2 0.28 / 0.26 vs 0.41 / 0.37, i.e. 0.68x / 0.71x of the existing GPU-vs-CPU gap).
- **Against the CPU reference the new kernel is statistically indistinguishable from the old one**: medians equal to three digits, the paired per-row difference is 1.5 s.e. (A) and 0.3 s.e. (B), and the new arm is closer on ~half the rows. Its tail is slightly worse (p5 0.851 vs 0.870 on A, 0.907 vs 0.921 on B): a small real effect at the bad rows, not a shift of the bulk. The global-cosine losses vs CPU are -0.0008 (A) and -0.0032 (B).
- **The error lives in the same channels**: the top-5 error dims of new-vs-old and of old-vs-CPU overlap in 4 of 5 (A) and 5 of 5 (B); they hold 18-20% of the squared difference. It is the same outlier-channel amplification through 27 layers the P6 record attributes the 0.91-0.96 to, not a new failure pattern (e.g. no block of rows, no padded-lane structure).
- The worst rows are largely, not entirely, shared: the 40 worst rows vs CPU overlap between the old and new arms 30/40 (A) and 18/40 (B), the 200 worst 141/200 and 116/200; the minimum per-row cosine is 0.230 (old) / 0.225 (new) on A and 0.098 / 0.084 on B (the same worst row, 3274, on B). So the bad rows are mostly ones the shipped tower already gets wrong, but the new kernel reshuffles part of the tail.
- The norm ratio new/old is 1.007 / 0.988: no systematic scale drift from the f16 P rounding.
- **Not shown:** that any of this is benign downstream. The reference is the repo's own CPU int8 tower, not an f32/HF oracle, and gate 4 (`GenerateVL` first-token identity) is NOT RUN. All of the above says "no worse than the noise the shipped tower already carries", not "correct".
- **On the registered 0.98 line:** the bar compares two GPU arms at 0.98 while the shipped arm agrees with the CPU only at 0.91-0.93, so the line is stricter than the tower's own agreement floor. That is a mechanism for reconsidering it, and is written here as the owner's call; the registered rule was applied as written (parked) and the bar has not been moved.

## State
Shipped opt-in: `GOINFER_CUDA_VISION_ATTN=bm128` (or `bm64`); the default is unchanged (`attn_img_batched`), which is also the kill-switch value (`exact`). No default flip is made under the registered rule. Phase B (int8 `mma.sync` GEMM, R8 build item 1) is scoped only if the tower is still > 8 s: **it is not (4.08 s), so by the pre-registration it is not started**; the device-side add (build item 3) is ~2% of wall and not started.

## Not established
Served vision TTFT and the Ollama `gemma3:4b` vision peer row (R8's separate instrument) are not measured; the per-layer localisation of the 0.96; the downstream first-token check; Qwen2.5-VL's tower (out of scope).
