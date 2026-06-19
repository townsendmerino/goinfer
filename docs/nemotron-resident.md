# Nemotron-H (nemotron_h) resident DecodeRunner port

Porting Nemotron-Nano (Nemotron-H hybrid) to the GPU full-residency DecodeRunner, reusing the
granite SSM engine (`mambaConv`/`mambaSSM`/`mambaGatedNorm`, the parity/drift/quality harnesses).
RTX 2070 SUPER (8 GB). Status: **COMPLETE (P0–P7).** Headline: **DEFAULT-on at int4 — 92.5% greedy / 99.6% top-2 agreement,
perplexity 1.677 ≈ f32 1.695; the ~7.5% disagreements are 100% benign (near-tied #2 picks, zero
confident errors). NOT granite's wall — the dense/no-MoE hybrid quantizes cleanly.**

## P1 — architecture: DENSE squared-ReLU hybrid ✓ (no MoE)

Confirmed from `decoder/forward_nemotron.go` + `decoder/arch.go`. Nemotron-H is **single-op-per-block**:
each layer is exactly ONE of `{Mamba-2 mixer, NoPE GQA attention, non-gated relu² MLP}`, applied
pre-norm with a residual add — *not* the mixer+FFN block every other family uses. Plain RMSNorm,
**no multipliers** (unlike granite's 4). **No MoE, no router** → none of granite's router-adjacent
risk by that mechanism. The served `nvidia_NVIDIA-Nemotron-Nano-9B-v2-Q8_0` is this dense variant.

**Gates `decodeRunnerEligible()` trips:** `a.nemotron != nil → false` (the own-forward gate, same as
granite/gemma4/qwen35/llama4). So Nemotron-H takes the granite pattern: a dedicated resident bridge
branch + accessor + a guarded eligibility flip, NOT the generic dense path.

**Composition / reuse map:**
| block | resident piece | status |
|---|---|---|
| Mamba-2 mixer | `mambaConv`/`mambaSSM`/`mambaGatedNorm` + state cache | **reused unchanged** (same `mamba2Params`) |
| NoPE GQA attention | resident attn GEMVs + attn kernel, **no RoPE** (SSM carries position), no QK-norm/bias, scale 1/√hd | reused (skip the rope-store) |
| non-gated relu² MLP | up(W8A8) → **relu²→int8** → down(W8A8) | **NEW — built (P2)** |
| layer structure | single-op-per-block (mixer-OR-attn-OR-mlp), pre-norm + residual | NEW wiring (per-layer skip flags) |

Genuinely-new work: (1) the relu² FFN kernel [done], (2) single-op-per-block layer wiring, (3) the
nemotron resident bridge + guarded eligibility. Everything else is reused granite/resident infra.

## P0 — fit: int8 does NOT fit 8 GB (the blocker)

The only Nemotron-Nano on the box is **9B-v2 Q8_0 = 8.9 GB**; the GPU has **7.2 GB free** (564 MB
display). int8 resident weights are ~8.3 GB even with embed+lm_head kept CPU-side — it OOMs and
falls back to staged. **So the int8 real-model quality/speed gates (P5/P6) cannot run for this model
on this box.** int4 (W4A8, ~4.5 GB) *does* fit, and the resident already supports W4A8.

Granite established the resident int8 gap is **precision-INVARIANT** (W8A8 66% ≈ f16 64% ≈ W8A16
62.6% — see `ssm-int8-quality.md`), so **int4 quality ≈ int8 quality** here: int4 is a valid proxy
for the int8 question. The real-model P5/P6 should therefore be run at **int4** (with that caveat).

## P2 — squared-ReLU FFN kernel + parity ✓

`gpu/relu2.go`: `relu2Quant` — the fused `relu²(up) → max-abs → int8` dispatch (mirrors
`swigluQuant` but unary; `relu²(x)=max(0,x)²`, so the int8 is non-negative). The up/down GEMVs reuse
the existing W8A8 path; this is the only new resident kernel.

**Parity** (`gpu/relu2_test.go`, `TestNemotronRelu2FFN_parity`): the full resident FFN
`up(W8A8) → relu2Quant → down(W8A8)` vs the CPU reference `down(relu²(up·x))` on the SAME int8
weights (so only the resident int8 *activation* quant differs), 32 tokens, non-mult-16 `inter` to
exercise padding → **worst cosine 0.999567.** The new piece is correct.

## P3/P4 — wiring + whole-model long-context parity ✓ (cosine 1.0, no drift)

`runLayer.nemoKind` (`gpu/decoderunner.go`): an early relu²-MLP branch + "skip FFN after the mixer"
for mamba/attn layers (single-op-per-block); `nemoNone` (every other family) is untouched →
byte-identical. Bridge (`gpu/residency.go`): a nemotron branch building per-layer weights by block
kind (int8 mamba via `projF32`; attn/MLP via `proj` = model-native W8A8/W4A8), ONE pre-norm
per layer, NO multipliers, **NoPE = a zeroed invFreq** (rope kernel → identity). Eligibility flips
guarded (`a.nemotron != nil → GOINFER_SSM_RESIDENT`).

**Parity** (`gpu/nemotron_resident_parity_test.go`, opt-in): resident vs CPU `runLayersNemotron`,
matched int8, the 1/16/256/1k/2k ladder → **cosine 1.000000 (maxAbs ~6e-7) at EVERY checkpoint
through 2048 tokens, ZERO drift.** The single-op routing, NoPE, relu²-FFN, and state threading
(conv ring + ssm + KV) are bit-correct over long context.

## P5/P6 — quality + speed (int4, real 9B) — THE HEADLINE

int8 doesn't fit 8 GB; the 9B fits at **int4** (int8 mamba projections + int4 attn/MLP, ~6–7 GB,
`ResidentActive=true`). Teacher-forced vs the f32 CPU reference (R1), 159 held-out tokens
(`gpu/nemotron_quality_test.go`):

| metric | int4-resident vs f32 R1 |
|---|---|
| **greedy agreement** | **92.5%** |
| **perplexity** | **1.677** (R1 1.695 — essentially identical) |
| **mean KL(f32‖int4)** | **0.058** |
| **top-5 overlap** | 88.2% |
| **speed** | **77.2 ms/tok = 12.9 tok/s** (~13× the f32 CPU path) |

Free-running greedy is coherent + factual: "Paris … the capital and largest city of France",
"2+2 → 4", **"red, blue, and yellow"** (correct primary colors — granite's int8 resident said
"red, blue, and **red**"). **This is the opposite of granite's wall** (granite int8 = 66% / KL 0.93
/ 2.4× perplexity): Nemotron-H quantizes essentially losslessly in distribution.

**Why the contrast:** granite's gap was proven NOT to be the MoE router *per se* — it's the chaotic
f32-reduction-order amplification through a deep stack, which granite's **64-expert top-6 router**
turns into hard expert-selection flips (a discrete cliff). Nemotron-H has **no router** — every
block is a smooth dense op, so the same tiny perturbations stay *smooth* (KL 0.058, ppl unchanged)
instead of flipping a discrete selection. The "no MoE router" really did matter — measured, not assumed.

## Benign-vs-harmful: the ~7.5% disagreements are 100% benign (coin-flips on ties)

Sub-95% greedy agreement WITH equal perplexity should mean the disagreements are coin-flips on
near-tied tokens, not mistakes. Confirmed (`gpu/nemotron_benign_test.go`, 227 held-out tokens, int4
vs f32, capturing f32's full per-position distribution):

| signal | value |
|---|---|
| greedy agreement | 92.5% (17 disagreements) |
| **TOP-2 agreement** (int4 pick is f32's #1 or #2) | **99.6%** (226/227) |
| f32 top-1 prob — at AGREE vs DISAGREE | 0.969 vs **0.416** (median) |
| **f32 top1–top2 margin — at AGREE vs DISAGREE** | **0.953 vs 0.069** (median) |
| int4-pick rank under f32 (disagreements) | **#2 = 16**, #3–10 = 0, >10 = 1 |
| harmful (rank>3 AND f32 margin>0.10) | **0** |

The correlation is textbook: int4 diverges ONLY where f32 was itself near-indifferent (median margin
0.069 vs 0.953), and in 16/17 cases it picks f32's *exact* #2 (the lone >10 was also a low-margin
tie). Zero confident-token errors. This is why perplexity is unchanged — swapping #1↔#2 at a
0.07-margin tie costs ~nothing. Free-running greedy is correct on factual / instruction / code
prompts ("100 degrees Celsius", "Red, Blue, Yellow", `def sum_list(numbers): return sum(numbers)`).

## Decision — FLIPPED: Nemotron-H resident is DEFAULT-on at int4

Holistic criterion MET: perplexity within a few % of f32 (1.677 vs 1.695) AND KL small (0.058) AND
disagreements overwhelmingly benign (100% — all near-tied #2 picks, zero harmful) AND free-running
coherent. The eligibility flip is **made**, GUARDED so only Nemotron changes:
`Architecture.decodeRunnerEligible()` returns true for nemotron, and `Model.DecodeRunnerEligible()`
admits it **default-on when the projections loaded int4** (`residentProjsInt4`) — int8 (unmeasured
on 8 GB; fits ≥12 GB) stays OPT-IN behind `GOINFER_SSM_RESIDENT`, and every other family is
untouched. On 8 GB an int8 load OOMs and falls back to staged gracefully (`withResidency`). int8
quality is ≥ int4 (less quant loss; nemotron is not at a precision-invariant wall), so the int8
default-flip is a measurement formality on a ≥12 GB GPU.
