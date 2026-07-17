# Gemma 3 on Metal — the likely resolution (Fable analysis, companion to the debug report)

> Companion to `gemma3-metal-debug-report.md` (status: UNRESOLVED). A deep second pass
> found that the report's central localization likely rests on a **measurement artifact**,
> and hands over a **30-minute, CPU-only, no-GPU-code decisive test** to confirm it.
>
> **The catch:** every pos-0 number in the report is measured on **token 2 = Gemma's
> `<bos>`** (`metal/gemma_parity_test.go:103`, `gemma3IDs = []int{2, …}`) — the
> **attention-sink** token, the most numerically pathological token in the model. And
> **every number in the report is a cosine; nobody ever measured a norm.**

## Why that breaks the localization

The report's chain is: layer-1 V cos = −0.047 (orthogonal) ⇒ `x` after layer 0 is
orthogonal ⇒ the bug is in layer-0's post-attention path. That inference is only valid if
the layer-1 V projection is well-conditioned. **It's the opposite.** BOS attention-sink
value vectors are *trained to carry near-zero content* (sink K = a strong recognizable
direction, sink V ≈ a no-op) — which exactly matches the observed **K=0.40 / V=−0.047**
split at layer 1 and the wild non-monotonic swings after it (0.95 → 0.06). A cosine
between two **near-zero** vectors is dominated by quant rounding noise and flops with
whichever projection you look through. So `cos(V₁_gpu, V₁_cpu) ≈ 0` is fully compatible
with `cos(x_gpu, x_cpu) ≈ 0.95` — a mildly-off residual stream, not an orthogonal one.
**Orthogonal-and-tiny (artifact) vs orthogonal-and-full-size (real break) are different
diagnoses, and one `‖·‖` print distinguishes them.**

This also explains the crux cleanly: **synthetic random-weight Gemma passes because the
attention-sink structure is a *trained* property — random weights don't have a near-zero
sink V, so the cosine stays well-conditioned.** Real weights have it; the measurement
degenerates only there.

## The argmax is a red herring — both directions

14/24 carries ~zero information at n=24. The harness's own comment
(`gemma_parity_test.go:130`) shows the **shipped, known-good** dense path scores 15/24 on
*these* ids and 20/24 on another id set — ±21 points from trajectory choice alone. Under
forced greedy lockstep on a high-margin trajectory, ~60% agreement is compatible with both
"healthy" and "badly degraded." No metric currently in the report can answer "is it
broken." The one that can is below — and needs no tokenizer.

## The decisive experiment (cheapest-first; tests the shared premise, not one branch)

Chosen over "Metal at W8A8" and "CUDA under the asymmetry" because those have confounded
outcomes (W8A8-recovers is consistent with both a bug and quant-fragility; CUDA shares the
same Go packer; both need new numeric surface / another machine). This adds **zero new
numeric code** — it stops the encode early and reads buffers that already exist over UMA.

**Phase 0 — CPU-only screen (no Metal at all, ~30 min).** Load the model twice on CPU:
`Quant:"int8int8"` and `Quant:""` (f32). Run the same 24 forced steps; compare per-layer
K/V at pos 0 **and print norms**.
- If CPU-int8-vs-CPU-f32 *also* shows degraded/erratic layer-1 V with **‖V₁‖ ≪ ‖V₀‖**
  (pre-registered prediction): the "orthogonal ⇒ Metal broken" localization collapses
  without touching the GPU — it's the BOS-sink/quant geometry, not Metal. **A confirmed.**
- If the CPU pair is clean (≥0.999 everywhere): the signature needs the bigger int4 vs int8
  perturbation or is genuinely Metal-side → Phase 1, with bug-odds raised.

**Phase 1 — Metal probe.** Test-only `encodeTrunkInto` that encodes layer 0 and stops after
a chosen dispatch (o-proj→`oO`; post-attn `rmsnorm_f32`; residual; preMLP quant; GLU;
down→`dO`; post-MLP norm; final residual); read `oO`/`dO`/`x` via `.Floats()`. Report **cos
AND ‖·‖** for all three references (GPU-W4A8, CPU-int8, CPU-f32) at each point, plus the
absmax/rms concentration ratio of every vector entering a quant. This *names the culprit
dispatch* if there is one.

**Phase 2 — tokenizer-free brokenness metric (same run).** Over the 24 forced ids, compute
each side's NLL of the actual next token. **ΔNLL(GPU−CPU) < ~0.3 nats ⇒ functionally fine
regardless of cosine; ≫ 1 nat ⇒ genuinely broken.** This is the honest gate; the 0.95
GPU-cos-CPU bar is a category error when the two sides are quantized differently.

### Decision table

| Observation | Verdict |
|---|---|
| cos(x_gpu, x_cpu8) after layer 0 ≥ ~0.9, layer-1 V ~0 with ‖V₁‖ ≪ ‖V₀‖ | **A: localization was a measurement artifact.** 0.699 is amplified accumulation (synthetic's 0.6%/layer × 34 ≈ 0.79 already ≈ 0.764). Fix = Gemma bar / outlier-aware quant, not kernels. |
| cos(x_gpu, x_cpu8) ≈ 0 at a specific probe point, **norms full-size** | **B: real bug, culprit dispatch named** (o-proj int4 / `rmsnorm_f32` / residual-hazard / MLP). |
| cos ≈ 0 at post-attn norm but ‖oO‖ pre-norm ≈ 0 both sides | **H1b:** rms-norming a trained-near-zero sublayer output amplifies each side's own noise to stream scale — not a code bug; quant policy. |
| ΔNLL small in all cases | the 0.95 bar is wrong for Gemma; gate on NLL / vs-f32. |

## Ranked hypotheses

1. **H1 — BOS-sink cancellation + int8-activation-absmax SNR collapse + sandwich no-dilution (~60%).** Not a bug. Layer-1 V(BOS) is a trained near-cancellation; both sides' quant noise (int4-vs-int8 weight ≈ 10% RMS — that's what requant-cos 0.9948 means — plus rounding) dominates a near-zero vector ⇒ cos ≈ 0 in the measurement projection, x only mildly off. Predicts ‖V₁‖ tiny; concentration ratio ≫ synthetic; **pos 3–5 KV far cleaner than pos 0.**
2. **H1b — sandwich norm of a near-zero o-proj output at BOS (~20%).** ‖o(v₀)‖≈0 ⇒ `rsqrt(ss/H+eps)` blows both sides' independent quant noise up to full stream scale ⇒ a real step-to-orthogonal in x with **no incorrect code**. Distinguished by ‖oO‖ pre-norm.
3. **H2 — a genuine data-gated Metal defect: `f32ToF16` on packed group scales (`metal/pack.go:53`) flushes scales < ~6e-8 to zero (~10%).** Whole 32-weight groups dead on GPU, alive on CPU; data-dependent, invisible to synthetic and to H9. **5-min pre-check: scan packed scale buffers for zero/subnormal halves in layers 0–1.** The one surviving concrete "real bug" candidate.
4. **H3 — int4 LM-head asymmetry contaminating the logit cosine (near-certain partial contributor, cheap fix).** `metal/model.go:264` int4-quantizes the head; CPU pins it int8; `decoder/weightmat.go:50` *documents* that int4 heads "flip the argmax and tank the cosine." Gemma's tied 262k×2560 head makes this maximal. Separates from the trunk by computing logits on CPU from Metal's final hidden. **Cheap fix: pin the Metal head at int8 via the already-compiled `gemv_w8a8_coal`.**
5. **H4 — GGUF-vs-safetensors mapping (<5%).** Can't produce a GPU-vs-CPU divergence (both read the same load); only matters via absolute quality, which Phase-2 NLL measures anyway.

## What the report missed (fix these regardless of the verdict)

- **Norms.** Not one norm in the whole report; the orthogonal-V finding is uninterpretable without ‖V₁‖. Add ‖·‖ to every comparison, permanently.
- **The reference's own fidelity.** CPU-int8 was treated as ground truth; its distance from CPU-f32 on Gemma was never measured (trust was calibrated on Qwen). Phase 0 fixes this.
- **H9 measured the wrong space** — weight-space requant cosine is isotropic; the live hypothesis is anisotropic (output-space under real activations). The kill was recorded against a broader claim than it tested.
- **BOS as the probe token.** Pos 0 eliminates RoPE/window/QK-scale (clever) but maximally exposes sink degeneracy. **Re-run the per-layer KV table at pos 3–5.**
- **The head asymmetry (H3)** was never in the hypothesis list despite the codebase flagging it as logit-critical.
- **"Coherent generation is blocked" is false** — SentencePiece *decode* needs only the GGUF's token array (id→piece), not the merges (those are for *encoding*). Seed with the existing ids, self-feed greedy 50 tokens each side, read the two texts. The report's #1 next step was never actually blocked.
- **The bar itself.** If H1 holds, the 0.95 GPU-cos-CPU gate is a category error for Gemma under int4-GPU-vs-int8-CPU (same asymmetry: ~0.04 on Qwen, ~0.30 on Gemma, no bug required). The durable gate is per-side NLL vs forced ids, or each side cos'd against CPU-f32.
