# Task: Metal Gemma 3 resident decode

Port Gemma 3 GPU-resident decode to the cgo-free Metal backend (`metal/`, `-tags metal`),
mirroring the CUDA implementation (`03e9816`) so the five Gemma features can be flipped on
for Metal honestly. **Gemma 4 stays refused** on both backends (`FeatLogitSoftcap` + its own
non-uniform forward: PLE branch, per-layer head_dim, cross-layer KV sharing).

The shared decoder foundation already landed with the CUDA port: `decodeRunnerEligible` no
longer arch-refuses sandwich norms (it's a feature-gate question now), and the accessors are
exported. Metal declines Gemma today **purely via `MissingResidentFeatures`** — so this task
is: implement the kernels, then add the five features.

## What Gemma 3 needs (derived, not guessed)

`archFeatureProfile["gemma3"]` = `{EmbedScale, GatedGELU, PerLayerRoPE, QKNorm, RMSAddOne,
SandwichNorm, SlidingWindow}`. Metal already ships **QKNorm** and **SlidingWindow**.

**Note there is NO `FeatPartialRotary`.** Gemma 3's rotary is FULL WIDTH on both local and
global layers — only the *base* differs (local 10k / global 1M). (Gemma **4**'s global layers
are partial via `GlobalRotaryDim` — a different arch with its own forward. Do not conflate:
implementing a partial-rotary global layer here would silently diverge from CPU.)

## The five features → Metal work

| Feature | Work |
|---|---|
| `FeatEmbedScale` | **None.** `decoder.embedResident` applies ×√hidden host-side, so the resident stream starts where the CPU's does. The backend receives an already-scaled `emb`. Just flip the flag. (Metal's own `Forward(id,pos)` does an internal un-scaled embed lookup — not used by the parity test, which passes `EmbedResidentForTest(tok)`.) |
| `FeatPerLayerRoPE` | Per-layer `invf` buffer from `m.RopeInvFreqLayer(l)` (replaces the single model-level `r.invf`). **The rope kernel is unchanged** — all per-layer variation rides in the table contents; width and `r.half` stay model-level. |
| `FeatRMSAddOne` | `addOne` uniform into `rmsnorm_quant` and the **final norm** (easy to miss). `qk_norm` already has it. CPU order: `(v*inv) * (1 + w[i])` — weight applied AFTER the normalize. |
| `FeatGatedGELU` | `act` selector in `swiglu_quant` (→ `glu_quant`): `ACT_GELU_TANH=0`, `ACT_SILU=1`, ordinals deliberately = `decoder.ActKind` iota, passed as `int32(m.GatedActResident())`. GELU-tanh = `0.5·x·(1+tanh(0.7978845608028654·(x + 0.044715·x³)))`. f32 on GPU vs the CPU's f64 — the same trade the SwiGLU path already ships; clears the 3% bar. |
| `FeatSandwichNorm` | New `rmsnorm_f32` (plain in-place RMSNorm of the [H] sublayer output, no fused quant, honors `addOne`) **and splitting the fused residual epilogue** — see below. |
| *(+ per-layer window)* | Metal's `uWindow` is model-level today (correct for all-local Mistral/Phi-3, **wrong for Gemma 3**, which mixes local and global layers). Bind a per-layer window: `m.LayerIsLocalResident(l) ? window : 0`. Covered by the already-declared `FeatSlidingWindow`, so it must be fixed here or Gemma 3's global layers get wrongly windowed. |

## The sandwich norm — the one structural change

Gemma norms each **sublayer OUTPUT** before the residual add:
`y = proj(...)` → `y = rms(y)·(1+w_post)` → `x += y`.

That is incompatible with Metal's fused `_resid` GEMV epilogues (`gemv_w4a8_sa_resid` /
`gemv_w4a8_resid` do `out[row] += acc·asc`), which absorb the residual add into the
projection. The sandwich path must therefore project into a scratch, norm it, then add:

```
o-proj   : pSA      (plain, → r.oO)  ; rmsnorm_f32(r.oO, L.postAttnNorm) ; pRes(r.x, r.oO)
down-proj: pGemv    (plain, → r.dO)  ; rmsnorm_f32(r.dO, L.postMLPNorm)  ; pRes(r.x, r.dO)
```

Two conveniences: `r.oO` / `r.dO` are **already allocated and currently dead** (the fused
epilogue absorbed them) — exactly the buffers this needs back, same as CUDA; and `residual`
(`x[i] += y[i]`) already exists as `r.pRes`.

Everything else in the per-layer order is unchanged. The delta vs Qwen2/Llama is precisely:
the o-proj and down-proj each go from **one fused dispatch to three**, and every RMS gains
`addOne`.

## Weights (no loader work)

`LayerWeights.PostAttnNorm` / `.PostMLPNorm` (`[HiddenDim]`, nil for Pre2) already load via
the shared Gemma tensor schema. Naming trap (shared code, already handled): for Gemma,
`post_attention_layernorm.weight` really IS the post-attn norm and `pre_feedforward_layernorm`
/ `post_feedforward_layernorm` are the MLP pair — whereas for Qwen/Llama the identically-named
`post_attention_layernorm.weight` is positionally the **pre-MLP** norm. Upload both as f32 and
validate `len == hidden` when the arch declares sandwich (a silently-missing one would drop
the norm, not error).

## Test plan (mirror the CUDA test shape)

The gate is the repo's **3% near-tie rule**, not a cosine bar: argmax must match, **or** the
CPU-logit gap between CPU's pick and GPU's pick must be ≤3% of the CPU logit **range**.
Cosine is a logged diagnostic with a `< 0.95` gross-breakage floor only — an earlier CUDA
draft asserted cosine ≥ 0.999 and that bar **failed the shipped dense path** (min 0.9936).
W4A8 int4 does not reproduce CPU int4 to 0.999.

So: a **dense control** on the same harness is mandatory — without it a Gemma cosine can't be
judged. Also `t.Fatal` if the resident path DECLINED (nil `ResidentForwardForTest`), which is
what catches a silent CPU fallback passing every assertion trivially. The GPU side must go
through `EmbedResidentForTest(tok)` — that call is what applies the embed scale, so it's the
only place `FeatEmbedScale` is under test.

Unit-level first (no checkpoint): `rmsnorm_f32` (+addOne) and the `glu_quant` act selector vs
CPU references.

## Admission (last)

Add to `ResidentBackendFeatures["metal"]`: `FeatRMSAddOne`, `FeatSandwichNorm`,
`FeatGatedGELU`, `FeatEmbedScale`, `FeatPerLayerRoPE`; update the `noOverclaim` want.

**Two derived-predicate exposures to mirror knowingly** (CUDA has both):
- `FeatSandwichNorm` is derived as `NormPlacement != NormPre2` — a *negative* predicate. Fine
  today (only two placements exist); a third would be silently admitted.
- `FeatGatedGELU` is `!NonGatedMLP && Act != ActSiLU` — so claiming it also claims an
  `ActReLU2`-gated arch, which the kernel's `else` branch would mis-run as GELU. Safe only
  because nothing else is admitted today.

## STATUS: kernels land dormant — one bug outstanding, localized

The five features are implemented and the kernels are unit-tested green, but Metal does **not
yet declare** them: on a real gemma-3-4b-it the resident output diverges from CPU, so declaring
would be an overclaim. `TestGemma3ResidentParity` skips until the declaration lands.

### What is measured, not assumed

| | result |
|---|---|
| dense control (shipped path, same harness) | 15/24 argmax, **min cosine 0.962** → PASS |
| gemma-3-4b-it | 14/24 argmax, **min cosine 0.699** → FAIL (gross breakage) |

Note argmax alone *passed* Gemma (14/24) while cosine showed 0.699 — the same trap CUDA hit
from the other side. Neither metric alone is trustworthy; both are in the test.

### Ruled OUT (don't re-pay for these)

- **Not an OOM.** The Linux box's trap #4 (an OOM wearing a parity bug's clothes) was tested
  directly: with allocations checked (`mustBuf` panics on a nil id) **no allocation failed** and
  the cosine was bit-identical. It is a real compute bug.
- **Not rope / qk-norm / window / attn-scale.** The divergence is present **at pos 0**, where
  θ=0 makes RoPE the identity and attention output is exactly `v0` (softmax over one key) —
  q, k, rope, qk-norm and the scale provably cannot affect the output.
- **Not the kernels individually.** `rmsnorm_f32` (both addOne modes), `rmsnorm_quant` (both
  addOne modes), and the `glu_act` selector (GELU-tanh *and* SiLU) all match CPU references.
- **Not the uniforms.** Verified against the live model: `uAddOne=1`, `uAct=0` (GELU),
  `sandwich=true`, 5:1 local/global pattern, embedScale=50.596=√2560, `prefillOK=false`.
- **Not the dispatch order.** Checked line-for-line against the CPU forward (model.go:440-461)
  and against `normalize`/`rmsNorm`.

### LOCALIZED — where to look next

A layer-by-layer KV bisect (compare `cache.Keys/Vals(l)` against Metal's f16 `kc/vc[l]` at
pos 0) pins it precisely:

```
layer  0: K cos 0.999515   V cos 0.998092     <- inputs to layer 0 are CORRECT
layer  1: K cos 0.403001   V cos -0.046768    <- catastrophic
```

Layer 0's KV is right, so preNorm + QKV are fine. Layer 1's KV is fed by layer 0's **output**,
so the bug is in **layer 0's post-attention path** — precisely the sublayer this task rewrote:
`o-proj → rmsnorm_f32(postAttnNorm) → residual → rmsnorm_quant(preMLPNorm) → gate|up → GeGLU →
down → rmsnorm_f32(postMLPNorm) → residual`. V cos of −0.047 is orthogonal, i.e. x after layer 0
is unrelated — a structural break, not drift.

Next step: bisect *within* that sublayer (a test hook that encodes only N layers and reads
`r.x`, or a tiny synthetic gemma3 for a fast iterate loop). Suspicion order: the sandwich
split's buffer/hazard handling (`oO`/`dO` reuse across dispatches), then the GeGLU wiring.
