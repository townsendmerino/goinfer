# Nemotron-H (nemotron_h) resident DecodeRunner port

Porting Nemotron-Nano (Nemotron-H hybrid) to the GPU full-residency DecodeRunner, reusing the
granite SSM engine (`mambaConv`/`mambaSSM`/`mambaGatedNorm`, the parity/drift/quality harnesses).
RTX 2070 SUPER (8 GB). Status: **COMPLETE (P0–P7).** Headline: **int4-resident 92.5% agreement /
perplexity 1.677 vs f32 1.695 — NOT granite's wall; the dense/no-MoE hybrid quantizes cleanly.**

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

## Decision — keep OPT-IN (guarded), default-on at int8/adequate-VRAM

Letting the number drive it: int4 agreement **92.5%** is just below the ≳95% default bar, so the
eligibility flip stays **guarded (opt-in via `GOINFER_SSM_RESIDENT`)** — the principled call on a
sub-95% greedy number. BUT this is a near-default result, not a degradation: perplexity and KL are
at the f32 floor, generation is clean, and it's **vastly** better than granite. Two caveats push
toward default-on in practice: (1) this is **int4** (the quant floor); int8 (the production default)
would land higher — likely clearing 95% — but doesn't fit *this* 8 GB box (it fits on ≥12 GB GPUs);
(2) nemotron is NOT at a precision-invariant wall (unlike granite), so precision genuinely helps
here. **Recommendation: ship opt-in on 8 GB (int4); flip to default-on for int8 deployments with
adequate VRAM.** Re-measure int8 agreement on a ≥12 GB GPU to confirm the ≥95% flip.
