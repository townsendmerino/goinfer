# Metal backend — model-architecture coverage & the admission bug

> What the cgo-free Metal resident decoder (`metal/`) supports today, what it silently
> mis-runs, and the effort to add more architectures. Scoped 2026-07-16. Companion to
> `task-metal-cgofree-spike.md`.

## The core structural fact (and a live correctness bug)

`metal/backend.go`'s `BuildResident` gates on `decoder.Model.DecodeRunnerEligible()` and adds
**no Metal-specific check**. But `decodeRunnerEligible` (`decoder/residency.go:99`) was written
for the **richer WebGPU/CUDA `gpu.DecodeRunner`**, which handles QK-norm, partial/scaled RoPE,
sliding-window, MLA, MoE. `metal/model.go` implements only the **plain dense Qwen2/Llama subset**.
So the predicate is *too permissive for Metal*: some archs pass the gate and run with features
**silently dropped → wrong logits**.

**⚠️ Bug, live today:** `--backend metal` on these produces silently-wrong output:
- **Qwen3** (dense) — per-head QK-RMSNorm ignored.
- **Mistral** (and any active `SlidingWindow`) — full attention instead of windowed; wrong once `seq > window`.
- **Partial-rotary** models (some GLM-dense / Phi) — rotates the full head dim instead of `rotaryDim`.
- **Mellum** — QK-norm + sliding-window both ignored.

**Companion fix (do regardless of roadmap):** tighten Metal admission so class-(b) archs
*decline* (fall back to the correct CPU/staged path) instead of mis-running — check `QKNorm`,
active `SlidingWindow`, `RotaryDim < HeadDim` and return `ok=false` until each feature lands.

## What `metal/model.go` bakes in (assumptions)

`BuildResident` (`metal/model.go:121-205`) + `encodeTrunkInto` (`:348-370`) hardcode one dense
block shape:
- **RoPE**: single full-head-dim NeoX table, no scaling (`r.invf` from layer 0; kernel rotates
  full `hd`, `kernels.go:233-239`). No mscale/YaRN, partial, dual, or m-RoPE.
- **QKV bias**: handled (Qwen2) — fused bias buffer, zeros when absent (`:167-172`).
- **QK-norm**: NOT handled (`residLayer` has no QNorm/KNorm; `QNorm/KNorm` never read).
- **Norm placement**: Pre-2 only (`preNorm=PreAttnNorm`, `postNorm=PreMLPNorm`, `:165-166`). No
  Gemma sandwich (post-attn/post-MLP) norms.
- **RMSNorm**: `x*rms*w` (`kernels.go:19,23`) — no Gemma `(1+w)` offset.
- **Embedding scale**: none (`ForwardEmb` copies verbatim).
- **Attention**: dense GQA, full causal, uniform scale — no logit softcap, no sliding window.
- **MLP**: gated SwiGLU/SiLU only (`kernels.go:279,283`) — no GeGLU/GELU-tanh, no non-gated ReLU².
- **FFN**: dense only — no MoE (empty `GateProj` → `int4Concat` panics → backend recovers → declines).
- **LM head**: tied or untied (both).
- **Weights**: must be int8-loaded → re-quantized to W4A8 int4. **KV cap 4096.** No output bias,
  learned positions, or MLA.

## Arch features the decoder already exposes (Metal could read these)

`Architecture` (`decoder/arch.go`): `EmbedScale` (:68), `FinalLogitSoftcap` (:72),
`AttnLogitSoftcap` (:73), `QKNorm` (:35), `SlidingWindow`/`layerIsGlobal` (:38-39),
`NormPlacement` (:24), `RMSAddOne` (:23), `Act`/`NonGatedMLP` (:27-28), `MoE`/`FirstKDense`
(:29-30, `MoEConfig` :228), `RotaryDim` (:46), `RoPELocal/GlobalBase` (:42), `ropeScaling`
(:57-58, YaRN mscale), `LogitScale` (:93, Granite), `LearnedPosEmbed`/`OutBias` (:34-36).
`LayerWeights` (`decoder/weights.go`): `QNorm/KNorm` (:38-39), `PostAttnNorm/PostMLPNorm`
(:42,53), `Router/Experts/SharedExpert` (:58-65). Ready-made resident accessors Metal ignores:
`m.RMSAddOne()`, `m.SlidingWindowResident()`, `m.LayerIsLocalResident(i)`,
`m.RopeInvFreqLayer(i)`, `m.RopeMscaleLayer(i)`, `m.MoEResidentParams()` (`residency.go:163-223`).

## Where models land today

- **Rejected before Metal (safe)**: Gemma-2/3/4, GPT-2, Llama-4, qwen3_5_moe, nemotron(non-int4),
  granite (unless `GOINFER_SSM_RESIDENT`).
- **Eligible but Metal-INCORRECT (the bug, class b)**: Qwen3, Mellum, Mistral-past-window,
  partial-rotary dense.
- **Eligible but Metal declines gracefully (correct via fallback, class c)**: all MoE
  (Mixtral/Qwen2-MoE/GLM-4-MoE), DeepSeek/Kimi MLA, non-int8 loads.
- **Eligible AND correct**: **Qwen2**, **Llama/Llama-3.2** (llama3 rope-scaling baked into the
  single inv-freq table), **Mistral within 4096 window**, **Phi-3-mini when full-rotary**.

## Effort ladder (recommended order)

| Model | Feature Metal needs | Containment |
|---|---|---|
| **Qwen3** (dense) | per-head QK-RMSNorm before RoPE | **1 small kernel** (qk_norm over head_dim, between `pSABias` and `pRope`) + 2 buffers/layer. Reuse rmsnorm math. Highest value/cost; also fixes the bug. |
| **Mistral** | sliding-window start in attention | **1 kernel tweak** — `winStart` uniform, loop `s=winStart..nKeys`. Per-layer global/local for full correctness. |
| **Phi-3** | partial RoPE | **trivial** — dispatch `rope` over `rotaryDim/2`, leave trailing dims. |
| **small MoE** | router + top-k expert dispatch + combine | **new kernel(s), isolated** (attention path unchanged). Params via `MoEResidentParams()`. Medium-high. |
| **Gemma 2/3** | sandwich norms + `(1+w)` RMS + softcaps + GeGLU + embed-scale | **structural** (5 features + loosen the gate). Defer. |

**Order:** admission fix → Qwen3 QK-norm → Mistral window → Phi-3 partial RoPE → small MoE → Gemma.

## Progress (2026-07-16)

- **Admission fix — DONE** (`metalUnsupported`): metal now declines QK-norm (until Qwen3
  below) / active sliding-window / partial-rotary / embed-scale models → correct CPU fallback
  instead of silent-wrong output. `HasQKNorm`/`PartialRotary`/`EmbedScaleResident` accessors added.
- **Qwen3 QK-norm — DONE (unit-validated; end-to-end pending)**: `qk_norm`/`qk_norm_f16` kernels
  (per-head Q/K RMSNorm, decode f32 + prefill f16), wired between QKV and RoPE, conditional on
  `m.HasQKNorm()`. Kernel bit-exact vs CPU rmsNorm (`TestQKNorm`, both `addOne` modes). qwen2.5
  parity unchanged. **No Qwen3 checkpoint on this machine (only qwen2.5 + gemma-4) → the Linux
  box should confirm end-to-end family parity.** Removed from the decline list.
- **Next**: Mistral sliding-window (attention kernel window-start), Phi-3 partial RoPE. Both are
  contained kernel tweaks but likewise need their checkpoints to validate — recommend validating
  Qwen3 end-to-end first (proves the pattern) before piling on more unvalidated families.
