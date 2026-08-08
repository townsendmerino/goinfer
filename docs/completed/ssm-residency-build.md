# Resident SSM/Mamba decode engine — build result (Granite-4.0-H)

The P0→P7 build of the resident Mamba-2 decode path (scope: `ssm-residency-scope.md`).
**Result: built, runs, generates coherently, and is 10–17× faster than the prior paths.**
On RTX 2070 SUPER / Vulkan, `-tags gpu`.

## Realized speedup (P7) — the deliverable

`TestGraniteResidentSpeedup` on granite-4.0-h-tiny (resident int8):

| path | ms/token | tok/s | speedup |
|---|---|---|---|
| CPU (own-forward) | 300 | 3.3 | 1× |
| webgpu staged | 498 | 2.0 | 0.6× |
| **resident SSM (this build)** | **29.6** | **33.7** | **10.1× vs CPU, 16.8× vs staged** |

Coherent + factual generation: *"The capital of France is"* → *"Paris. Paris is known for
its iconic landmarks such as the Eiffel Tower, the Louvre Museum"*. This lands in the scope's
predicted 6–15× range and, unlike Llama4, is **collectable on the 8 GB box** (granite-tiny fits).

## What was built (P2→P5b)

- **3 SSM kernels** (`gpu/mamba.go`), each isolation-parity-gated to f32 `mamba2Step`:
  `mambaConv` (causal-conv ring), `mambaSSM` (selective state, **zero drift @ 2048 tok**),
  `mambaGatedNorm`. One-pass composition parity green (`mamba_test.go`).
- **Resident integration** (`gpu/residency.go`, `gpu/decoderunner.go`): per-layer mixer-kind
  branch (Mamba SSM vs attention) following the C3/C4 precedent; build-once `{conv-ring, ssm}`
  state reset per generation; in/out_proj as W8A8; the 4 Granite multipliers folded
  (ResidMul→residual-add weight scales, EmbMul→embedding, LogitScale→lm_head, AttnScale→attn);
  attention + MoE-every-layer reuse the existing resident infra. Non-Mamba models are
  byte-identical (the branch is `isMamba`-guarded; full `-tags gpu` suite green).
- **Accessor bridge** (`decoder/residency.go`): `GraniteResidentParams` / `GraniteMambaLayer`
  / `GraniteMambaWeights`.

## Parity: correct at int8, diverges from f32 (precision, not a bug)

`TestGraniteResidentParity` (opt-in, `GOINFER_SSM_PARITY=1`) characterizes the resident-vs-CPU
logit cosine. Findings from the bring-up isolation seams:

- **vs f32 CPU: cosine ~0.37** whole-model. This is *not* a bug — it's int8 (W8A8) precision.
  Ruled out via seams: int8 weights *and* activations matched (`GOINFER_SSM_Q8CPU`) →
  unchanged; bind-offset slicing → identical; multipliers (`GOINFER_SSM_NOMUL`) → confirmed
  the folds are correct (identity multipliers expose the same error, masked by ResidMul·emb).
- **vs int8 CPU reference (`GOINFER_SSM_CPUQ8`): cosine 0.99 @ tok1** (argmax matches) — the
  engine is **computationally correct**; it reproduces the int8 result.
- The dominant f32 gap is the MoE: granite loads f32 weights and has **64 experts / top-6**;
  int8 router logits shift the expert *selection* vs f32 (mixer-only is 0.965, the MoE drops
  it to 0.37). The SSM's `exp(dt·A)` adds residual int8-activation sensitivity. Transformers
  tolerate int8 (~0.999); granite's MoE+SSM do not.

**Net:** int8 granite generates coherent, factually-correct text and is 10× faster, but its
logits diverge from f32 granite more than a transformer would.

## Eligibility (P6): GUARDED, not default

`decodeRunnerEligible()` admits granite **only under `GOINFER_SSM_RESIDENT`** — opt-in, so the
default production path stays the f32 staged/CPU one (the loose f32 parity shouldn't silently
become the default). Set the env to get the 10× resident path at int8 quality.

## Follow-up to make it default-able: f16 weights

A tight f32 gate (and default eligibility) needs higher precision on the int8-sensitive
weights — **f16 for the MoE experts/router + the mamba projections** (granite-tiny: ~int8
3.5 GB + ~2 GB f16 ≈ 5.6 GB, fits 8 GB). That's an f16-weight GEMV path (the W8A8 path is the
current GEMV); a contained follow-up. Bring-up seams (`GOINFER_SSM_*`) are kept env-gated to
drive it.
