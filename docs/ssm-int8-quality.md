# int8 resident SSM quality — granite-4.0-h-tiny

Generation-quality eval of the shipped guarded int8 resident SSM path
(`GOINFER_SSM_RESIDENT`) vs precision references, turning the cosine-0.37 finding
(`ssm-residency-build.md`) into agreement / KL / perplexity over a distribution.
Measurement only (`gpu/ssm_int8_quality_test.go`); no production change. RTX 2070 SUPER.

## Verdict

**int8 resident is materially degraded vs f32 (66% greedy agreement, 2.4× perplexity) — too
lossy for default; ship opt-in greedy-only as today.** But the cause is NOT what it first
looked like: localization experiments (E1/D1/D2/D3, below) refute precision, int8 quant, AND
the router as the lever, and pin the gap on the **GPU mamba kernels** — every CPU-side
reproduction of the resident's arithmetic scores ≥93.6%; only swapping in the GPU
conv/ssm/gatedNorm kernels drops it to 66%. So the loss is a **fixable GPU kernel discrepancy,
not a hardware precision floor** — meaning ~93% quality at the int8 10× speed is recoverable if
the kernel is debugged. The f16 pass and a router-precision island are both dead ends (refuted).

## Step 0 — references + precision

- **R1 = f32 CPU** (cpu backend, no quant — granite loads f32 weights): absolute ground
  truth. Held-out perplexity **6.49**.
- **R2 = staged default** (webgpu, no SSM residency): the own-forward runs on CPU (f32 mamba)
  with the attention/MoE matmuls staged on GPU. Note: for granite, staged is *slower than CPU*
  (498 vs 300 ms/tok), so the practical default users get today is **R1 (f32 CPU)** — i.e.
  today's quality is f32. (R2 staged-int8 numbers below.)

## Metrics — INT8 resident vs R1 (f32), held-out 219 tokens teacher-forced

| metric | int8 resident vs f32 | bar for "acceptable default" |
|---|---|---|
| **greedy agreement** (argmax == f32) | **66.2%** | ≳98–99% |
| **mean KL(f32 ‖ int8)** | **0.93** | small (≪0.1) |
| **top-5 overlap** | 58.3% | high (≳95%) |
| **perplexity** | **15.73** (f32 6.49) → **2.4×** | negligible delta |

All four fail the default bar by a wide margin. The cosine-0.37 finding is confirmed as a
real distribution shift: **a third of greedy steps pick a different top token than f32, and
the distribution differs enough (KL 0.93) that sampled (temp>0) output is materially worse**
— greedy agreement is a *floor*, and granite at any sampling temperature loses the KL margin.

## R2 (staged default) vs R1 (f32) — the localizing result

| metric | R2 staged | int8 resident |
|---|---|---|
| greedy agreement | **95.4%** | 66.2% |
| mean KL(f32‖X) | **0.026** | 0.93 |
| top-5 overlap | 90.3% | 58.3% |
| perplexity | **6.78** (f32 6.49) | 15.73 |

R2 is **f32 mamba (CPU own-forward) + int8 attention/MoE matmuls** — and it's **near-f32**
(95% / perplexity within 4.5%). This pins the dominant error: **R2 also runs int8 MoE, yet is
fine** — so the int8 **MoE is not the problem on its own**. The difference between R2 (95%) and
int8-resident (66%) is the **mamba precision**: int8 mamba projections corrupt the SSM output,
which then cascades into the MoE router → flipped expert selection. With f32 mamba (R2) the
router input is correct, the selection is right, and int8 experts add only small error.

## Qualitative — free-running greedy, resident int8 vs f32 CPU (6 prompts)

The degradation is not just statistical; it produces wrong facts / code / loops on common
prompts, while easy high-confidence ones survive:

| prompt | f32 | int8 resident |
|---|---|---|
| "The three primary colors are" | red, yellow, and blue ✓ | **red, blue, and red** ✗ + repetition loop |
| "Write a Python … Fibonacci" | (explains correctly) | `def fib(n): return fib(n-1) + fib(n-1)` ✗ (infinite recursion) |
| "Once upon a time …" | coherent story | loops: "love of **love**", "upon his retirement … upon his retirement" |
| "The capital of France is" | Paris … Eiffel/Louvre/Notre-Dame ✓ | Paris … "Eifff Eiffel" (minor stutter), else ok |
| "boiling point of water" | 100 °C / 212 °F ✓ | 100 °C (redundant), ok |
| "photosynthesis in one sentence" | full ✓ | shorter, ok |

So int8 is usable for *easy greedy* lookups but produces factual/code errors and repetition on
anything harder — consistent with the 66% agreement / 2.4× perplexity.

## Why — SETTLED by localization experiments: it's the GPU mamba KERNELS

> The two earlier guesses in this doc — "int8 mamba projections too lossy" and (after f16
> failed) "f64-vs-f32 SSM compute + router sensitivity" — were **both wrong**. The
> localization experiments below (E1/D1/D2/D3, `decoder/ssm_precision_localize_test.go` +
> `gpu/ssm_kernel_control_test.go`) refute every precision/quant hypothesis and pin the gap
> on the GPU mamba kernels.

All runs teacher-forced vs the f32 CPU reference (R1), 219 tokens:

| control | what it changes vs R1 | agreement | meanKL |
|---|---|---|---|
| **E1** cpu f32 SSM | SSM compute f64 → f32 (exp arg + gated-norm accum) | **100.0%** | ~0 |
| **E2** cpu f32 SSM + f64 routing | + force the f64 router selection | **100.0%** | ~0 |
| **D1** cpu int8 mamba (W8A8) | int8 mamba in/out_proj + int8 activation | **97.3%** | 0.029 |
| **D2** cpu int8 mamba + f64 routing | + force the f64 router selection | 95.0% | 0.014 |
| **R2** staged | int8 attn/MoE matmuls (f32 mamba) | **95.4%** | 0.026 |
| **D3** staged + int8 mamba | **ALL** int8, mamba still on CPU (`mamba2Step`) | **93.6%** | 0.045 |
| **GPU resident int8** | all int8 **+ GPU mamba kernels** | **66.2%** | 0.93 |

Read top-to-bottom: stacking every precision/quant change the resident makes — f32→f32 SSM,
int8 mamba, int8 attn/MoE — only takes the CPU reference from 100% to **93.6%**. The single
remaining step, D3 → resident, swaps the CPU `mamba2Step` for the **GPU conv/ssm/gatedNorm
kernels**, and *that alone* drops agreement 93.6% → 66%. So:

- **Not SSM precision** — E1: f64→f32 SSM is a no-op (100%, 0/8760 router flips).
- **Not int8 weights/activations** — D1: int8 mamba projections cost ~3% (97.3%); f16 (proven
  cosine-1.0 GEMV) didn't help either — consistent, because the projections were never the lever.
- **Not the router** — D1 has 22% *benign* router flips yet 97.3% agreement; forcing the f64
  selection (E2/D2) doesn't help. The 64-expert router is robust here, not hypersensitive.
- **It's the GPU mamba kernels.** D3 (CPU mamba) 93.6% vs resident (GPU mamba) 66.2% — a ~27-pt
  drop attributable *only* to the GPU conv/ssm/gatedNorm path. This matches the earlier
  per-layer signatures (SKIPFFN mixer cosine 0.965, whole-model 0.37): a per-step discrepancy
  in the GPU mixer that **compounds over the 219-token recurrence** (it was ~0.99 at token 1).
  A kernel bug / numerical divergence — **fixable in principle, not a hardware precision floor.**

## f16 mixed-precision attempt (built + measured) — does NOT recover quality

Acting on the R2 localization, I raised the mamba `in/out_proj` to **f16** (f16 weights ×
f32 activation; experts/attention/MoE stay int8) — `gpu/mamba_f16.go`, default-guarded to
granite-resident. It **fails Gate 1**, essentially unchanged from int8:

| config | agreement | KL | top-5 | perplexity | ms/tok |
|---|---|---|---|---|---|
| f32 R1 | 100% | 0 | 100% | 6.49 | 300 (CPU) |
| **R2 staged (f32 mamba + int8 MoE)** | **95.4%** | **0.026** | 90.3% | **6.78** | 498 |
| int8 resident | 66.2% | 0.93 | 58.3% | 15.73 | **29.6** |
| **f16-mamba resident** | **63.9%** | **0.94** | 58.0% | **14.3** | 55.8 |

f16 is **no better than int8** (63.9% vs 66.2%) and **2× slower** (f16 = 2× bandwidth on those
GEMVs). Qualitatively it still loops ("red, blue, and red …"). **The f16 GEMV is proven
accurate** (`cosine 1.000` vs f32 matvec in isolation), so this is **not an f16 bug** —
**the projection precision is not the lever.** The R2-based hypothesis is refuted.

f16 failing was the first clue that the projections aren't the lever; the localization
experiments above (added later) confirmed it and found the real cause — **the GPU mamba
kernels**, not precision. (An earlier draft of this section blamed "f64-vs-f32 SSM compute +
router sensitivity"; experiment E1 refuted that — f32 SSM is a no-op at 100%.)

## Recommendation

1. **Keep it opt-in, as shipped** (`GOINFER_SSM_RESIDENT`). It's coherent + factual on simple
   greedy prompts and 10× faster (33.7 vs 3.3 tok/s CPU) — a real win for latency-over-quality
   greedy use. **Do NOT make it default**: 66% agreement / 2.4× perplexity is a regression vs
   the f32 default (CPU; granite's staged path is slower than CPU, so today's default IS f32).
2. **Do NOT serve granite resident with sampling** (temp>0): KL 0.93 ⇒ sampled output is far
   worse than the already-degraded greedy.
3. **Do NOT fund a router-precision island, and do NOT fund more f16/weight-precision work** —
   the experiments refute both premises (router robust at 97% with 22% benign flips; int8/f16
   projections cost only ~3%). The default stays int8; f16 stays behind `GOINFER_SSM_F16MAMBA`
   for the record only.
4. **The gap is a GPU mamba-KERNEL bug, which means it is potentially recoverable at int8 speed
   — a much better prize than "close as greedy-only".** D3 (CPU mamba) hits 93.6%; the resident
   (GPU mamba) hits 66.2% on identical int8 weights. So the realistic options are:
   - **(preferred, if funded) Debug the GPU mamba kernels.** Bisect the per-layer resident-mixer
     vs `mamba2Step` divergence *over multiple tokens* (it's ~0.99 at token 1 but the multi-token
     whole-model is 0.37 → the discrepancy is in the **state recurrence compounding**: suspect
     the `ssm` state update / `dA` exp / conv-window ring, not the projections). Land it and
     granite resident recovers ≈93% quality at **29.6 ms/tok (33.7 tok/s, 10× CPU)** — usable as
     a default, not just greedy-only.
   - **(fallback) Bank the opt-in int8 10× as greedy-only**, as today. No router island, no f16.
     For quality, use the f32 CPU path (3.3 tok/s) until the kernel is fixed.
