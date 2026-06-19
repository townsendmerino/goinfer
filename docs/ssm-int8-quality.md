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

## Why — CORRECTED again (Phase A): it is NOT the mamba kernels either

> Three guesses in this doc have now been refuted in turn: (1) "int8 mamba projections too
> lossy", (2) "f64-vs-f32 SSM compute + router sensitivity", and now (3) "the GPU mamba
> kernels". The D1/D2/D3 controls below correctly excluded precision/quant; they were read as
> implicating the **GPU mamba kernels** because D3 (CPU mamba) = 93.6% vs the GPU resident =
> 66%. But **that delta was never clean** — D3 runs attn/MoE through the *staged* tiled GEMM
> while the resident runs its *own* W8A8 GEMVs (the Phase-A.1 confound). Direct capture proves
> the mamba kernels are innocent:
>
> - **`gpu/mamba_realinput_test.go`** — replay REAL granite layer-0 in_proj outputs (captured
>   from a live run) through the GPU conv/ssm/gatedNorm kernels vs `mamba2Step`, 66 tokens:
>   **worst cosine 1.000000**. The kernels are bit-correct on real inputs in isolation.
> - **`gpu/mamba_resident_capture_test.go`** — capture the resident's ACTUAL per-token layer-0
>   kernel I/O from the full plan and diff vs `mamba2Step` on the resident's own proj, 60
>   tokens: conv/y/gated **all cosine 1.000000**; and the resident's proj matches `mamba2Step`
>   (cosine 0.9997, no divergence). So the mamba mixer-in-plan AND in_proj are correct — no
>   barrier/race/state bug, no wiring bug in the mamba path.
> - **`gpu/mamba_layersweep_test.go` + `staged_layersweep_test.go`** — token-0 hidden vs f32
>   after each layer. Resident and staged are **bit-identical for layers 0–9**, then the
>   resident accumulates more error, ending at **0.892** vs staged **0.951** (→ 66% vs 93.6%
>   agreement). The drift is **distributed across the deep stack**, not any single op or layer
>   (no jump at the attention layers 5/15/25/35; MoE-skipped sweep still drifts).
>
> **Verdict: there is no mamba-kernel bug to fix.** The kernels, in_proj, and out are correct.
> The residual gap is the resident's all-GPU W8A8 forward accumulating modestly more error than
> the staged (CPU-mamba + GPU tiled-GEMM) path — tiny per-layer f32/int8 differences amplified
> by int8 re-quantization across granite's 40-layer all-MoE stack, then by cross-token state.
> It is **not a single fixable kernel discrepancy**; closing it would mean reducing int8
> re-quantization sensitivity (e.g. f16 activations on the residual stream / MoE), a precision
> change — which the f16-mamba pass already showed is not worth it. **Recommendation stands:
> opt-in greedy-only int8; do not fund a kernel hunt (there is no kernel bug).**

### Original D1/D2/D3 controls (still valid — they excluded precision/quant)

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
4. **Do NOT fund a "GPU mamba kernel" debug — Phase A proved the kernels are bit-correct**
   (`gpu/mamba_realinput_test.go`, `mamba_resident_capture_test.go`: cosine 1.000000 on real
   inputs, in isolation and in-plan). The 93.6% (D3) vs 66.2% (resident) delta is the resident's
   all-GPU W8A8 forward vs the staged tiled-GEMM path — a *distributed* accumulation across
   granite's 40-layer all-MoE stack (resident/staged are bit-identical for the first ~10 layers,
   then the resident drifts: token-0 final 0.892 vs 0.951), not a single fixable op. Closing it
   would require a precision change (f16 residual stream / MoE activations), which the f16-mamba
   pass already showed isn't worth it for a tiny model.
5. **Bank the opt-in int8 10× as greedy-only** (the standing recommendation). For quality use the
   f32 CPU path (3.3 tok/s). There is no cheap correctness fix; the gap is fundamental to the
   all-GPU int8 path on this deep, every-layer-MoE architecture.
