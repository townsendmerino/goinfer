# Nemotron-H (nemotron_h) resident DecodeRunner port

Porting Nemotron-Nano (Nemotron-H hybrid) to the GPU full-residency DecodeRunner, reusing the
granite SSM engine (`mambaConv`/`mambaSSM`/`mambaGatedNorm`, the parity/drift/quality harnesses).
RTX 2070 SUPER (8 GB). Status: **P0/P1/P2 done; P3–P7 blocked on the 8 GB fit (see P0).**

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

## P3–P7 — remaining (next session)

- **P3/P4** single-layer then whole-model long-context parity (1/16/256/1k/2k) on the tiny nemotron
  fixture — the cheap-insurance + drift gates. Needs the single-op-per-block wiring + bridge.
- **P5 quality** — run the harness at **int4** (int8 doesn't fit). **Prediction (not assumption):**
  Nemotron-H is the *same deep hybrid + recurrent mamba-state* architecture whose int8 gap granite
  proved FUNDAMENTAL and precision-invariant (chaotic f32-reduction-order amplification, NOT the MoE
  router). Removing MoE removes one *suspect* but not the proven *cause*, so a similar wall is
  likely → expected outcome **opt-in, not default**. But the task is right that it must be MEASURED
  (the no-MoE/relu² path could differ); the int4 proxy is the way to do it on this box.
- **P6/P7** realized int4 ms/tok + tok/s vs CPU/staged; full `-tags gpu` suite green.

## Decision (so far)

Dense confirmed, no router-wall-by-MoE. The squared-ReLU kernel is built and parity-clean. The
default-vs-opt-in call is **deferred to the P5 int4 measurement** — but the granite evidence
(fundamental, precision-invariant) makes **opt-in** the likely outcome. int8-on-the-real-9B is not
measurable here (8 GB); a smaller Nemotron-Nano or more VRAM would be needed for the exact int8 number.
