# int8 resident SSM quality — granite-4.0-h-tiny

Generation-quality eval of the shipped guarded int8 resident SSM path
(`GOINFER_SSM_RESIDENT`) vs precision references, turning the cosine-0.37 finding
(`ssm-residency-build.md`) into agreement / KL / perplexity over a distribution.
Measurement only (`gpu/ssm_int8_quality_test.go`); no production change. RTX 2070 SUPER.

## Verdict

**int8 resident is materially degraded vs f32 — too lossy for default. Ship opt-in only
(coherent on easy prompts, 10× faster); fund the f16 mixed-precision pass.** Greedy
teacher-forced agreement with f32 is **66%** and perplexity is **2.4×** worse — well outside
the ≳98% / negligible-perplexity bar for a default path.

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

## Why

int8 (W8A8) on the **mamba projections** is too lossy: the SSM's `exp(dt·A)` amplifies int8
`dt`/`B`/`C` error, and that error feeds the 64-expert router and flips top-6 selection
downstream. Transformers tolerate int8 (~0.999); the Mamba mixer doesn't. The resident path is
*computationally correct* (cosine 0.99 vs an int8 reference) — it faithfully computes the int8
model; the int8 **mamba** is the degraded artifact.

## Recommendation

1. **Keep it opt-in, as shipped** (`GOINFER_SSM_RESIDENT`). It's coherent + factual on simple
   greedy prompts and 10× faster (33.7 vs 3.3 tok/s CPU) — a real win for latency-over-quality
   greedy use. **Do NOT make it default**: 66% agreement / 2.4× perplexity is a regression vs
   the f32 default (CPU; granite's staged path is slower than CPU, so today's default IS f32).
2. **Do NOT serve granite resident with sampling** (temp>0): KL 0.93 ⇒ sampled output is far
   worse than the already-degraded greedy.
3. **Fund the f16 pass — and target the MAMBA PROJECTIONS, not the MoE** (corrected by R2: int8
   MoE is fine with f32 mamba). f16 the mamba `in_proj`/`out_proj` (~2.1 GB f16 over 36 layers;
   int8 everything else ≈ 5.6 GB total, fits 8 GB) — keep the 64 experts int8. Expected to
   recover ≈R2 quality (95% / perplexity ~6.8) at resident speed. **Gap f16 must close (re-run
   this harness):** greedy agreement **66% → ≥95%**, perplexity **15.73 → ≈6.8**, mean KL
   **0.93 → ≤0.05**.
