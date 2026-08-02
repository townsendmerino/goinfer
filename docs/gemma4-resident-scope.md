# Gemma-4 own-forward residency bridge — scope (Phase 9a)

**MEASURE + CHARACTERIZE + DESIGN + ESTIMATE, read-only.** This is a spec, reviewable before any
code. **No kernels, no engine code, no `benchmarks.md` row** are produced here (see the Boundary at
the end). Target backend: **CUDA** (RTX 2070 SUPER), per `docs/task-gemma4-moe.md` Phase 9b.

The per-layer *machinery* analysis below (how constants and mixer-kind branches already work) is
drawn from the **WebGPU** runner (`gpu/`, the reference with the richest per-layer machinery); the
CUDA runner (`cuda/`) is analogous and the bridge *concepts* — per-layer geometry, the variant
branch, host-side softcap — apply to both. But the **kernel cost model was verified against CUDA
directly** (the 9b target), not inferred from WebGPU: a first-draft of this spec wrongly imported
WebGPU's `hd≤128` attention ceiling as if it were the CUDA cost — it is not (gemma3 proves CUDA at
hd=256). Per-backend kernel/dispatch facts are stated per backend, not generalized.

Precedent this spec is modeled on: `docs/ssm-residency-scope.md` / `-build.md` — Granite-4.0-H and
Nemotron-H are already own-forward families that got a resident bridge (a per-layer *mixer-kind*
branch bypassing the uniform-GQA gates), both `✅ resident`. This is the same shape of work.

---

## Headline verdicts

- **The gap is per-layer SHAPES, not a "non-uniform forward."** Per-layer *constants* already
  work: `runLayer.isLocal` (C6 sliding window, residency.go:427/decoderunner.go:813) and per-layer
  `invFreq`/`ropeScale` tables (C7, residency.go:256-266/428). What Gemma 4 adds is per-layer
  *geometry* — `head_dim` 256 (local) vs 512 (global), `kv_heads` 8 vs 2, and K=V global layers
  with no `v_proj` (`V = v_norm(k)`, scale-less, no RoPE). These change **buffer sizes and dispatch
  geometry**, which the runner currently freezes model-level. Narrower than "own forward" implies.
- **It is two variants alternating 5:1, not arbitrary variation** — so "encode-once" survives.
  The build-once `[]runStep` plan (decoderunner.go:14-19) already emits *heterogeneous* per-layer
  steps (mamba/MLA/attention layers dispatch differently); the fix is to let a **per-layer variant
  index** select which geometry uniform/pipeline a layer binds, instead of the single
  model-level `attnUni` shared across all layers (decoderunner.go:448-452, 518-521).
- **The per-layer variant branch has direct in-repo precedent.** `runLayer.isMamba` /
  `runLayer.nemoKind` / MLA-via-`m.mla` all branch per layer in the plan loop (decoderunner.go:721/
  730/758/784), with the differing geometry carried per-layer and the uniform-across-those-layers
  bits in a model-level param block. The Gemma-4 bridge extends that pattern from *mixer*-kind to
  *attention-geometry*-variant.
- **The attention kernel needs NO change on CUDA — measured, not assumed.** The one open kernel
  question was Gemma 4's 512-dim global head. **Resolved (`TestAttention_HeadDimWidths`, `933201c`):**
  the shipped CUDA `attention` kernel passes cosine ≈ 1.0 vs the CPU GQA reference at hd 128 / 256 /
  512 — the fixed-128-thread block decomposes 512-dim heads (4 elements/thread) correctly. `hd=256`
  was already proven independently (Gemma 3 is `✅ resident` on CUDA, `hardware-matrix.md:20`; Gemma 3
  is *not* own-forward). The kernel launches `GridX=nH, BlockX=128 (fixed), SharedMem=(nWin+128)·4`
  (cuda/resident.go:549), `hd` as an arg, shared-mem sized by *keys* — hd-parametric, as the probe
  confirms. So **CUDA is "bridge + 0 attention kernels."** (WebGPU's kernel *does* hard-cap
  `hd ≤ 128` at gpu/attention.go:52 — a real kernel rewrite, another reason it is 9d. Metal handles
  256 already, `metal/attn_shape_test.go`, but that table tops at 256 — a Mac-side item is to add the
  512 row and run it; the CUDA result makes it likely to pass, not guaranteed on a different kernel.)
- **Prize collectability on THIS box, stated plainly (the Llama-4 caveat from the SSM doc).**
  `gemma-4-E2B` dense (~2 GB q4) and the tiny MoE fixtures (`testdata/gemma4-moe-tiny`,
  `gemma4-moe-unified-tiny`) **fit the 8 GB 2070** and gate the bridge here. The **26B-A4B at int4
  is 16 GB — it does NOT fit 8 GB**, so its *resident* prize is not collectable on this card; it
  needs a bigger GPU or Metal's unified memory (Phase 9c, M1 Pro 16 GB). Collectable-here (E2B +
  tiny) develops and parity-gates the bridge; collectable-eventually (26B) is a VRAM follow-on.
- **Fixture PROPERTY coverage, not just fit (the "blind by construction" hazard).** A fixture can
  fit *and pass parity* while never exercising the feature under test — `metal/attn_shape_test.go`
  says exactly this of its hd=128 control vs an hd=256 fault. The real 26B has **256-dim local /
  512-dim global** heads; `gemma4-moe-unified-tiny` config has `global_head_dim=512` but `head_dim=16`
  local. That **16 / 512 split is partly a STRENGTH, not just an unrealism**: it is a *harsher*
  geometry contrast than the real 256 / 512, so no buffer size, stride, or dispatch dim can
  accidentally coincide between the two variants — it stresses the per-layer seam harder than the real
  model would. Its weakness is orthogonal: a 16-dim local head does not exercise the real 256-local
  path (the CUDA kernel proof covers 256, but the resident *plumbing* at 256 is untested), and it must
  be verified that its global layer actually shapes 512-dim attention weights end-to-end. So the parity
  plan must name the *properties* to cover — 256 local, 512 global, K=V (no v_proj), the 5:1
  interleave, per-layer KV switch — and confirm (or purpose-build) a fixture hitting each: the tiny
  fixture is the seam-stressor, a 256-local one is still needed for realistic-width plumbing.
- **Do not let the abstraction come out Gemma-4-shaped.** `qwen35` (3 linear : 1 full attention)
  and `llama4` (per-layer `isMoE[]`, per-layer NoPE) are the **same per-layer-variant class**.
  Implement **only** Gemma 4 and build nothing speculatively — but the per-layer variant index +
  per-layer geometry on `runLayer` is the general seam, so name it generically, not `gemma4*`.
- **Go/No-Go: GO.** The seam is a bounded extension of an existing pattern (per-layer geometry,
  following per-layer constants + the mixer-kind branch); the one kernel unknown (hd=512) is already retired (933201c), leaving the
  Gemma-4 MoE delta. Effort **M–L**, comparable to the SSM bridge. Parity-gated on fitting fixtures.

---

## Phase 0 — measure/verify (the feasibility questions, read-only)

Before designing, resolve the unknowns a paper analysis can't:

1. **The `hd=512` attention question on CUDA — DONE.** Resolved by mutating the known-good
   `validateGlue` oracle (which cosine-checks `attention` vs a CPU GQA reference, parametric in hd)
   to hd 128/256/512 — single variable, so a red is the head dim not the contract. All pass cosine
   ≈ 1.0 (`TestAttention_HeadDimWidths`, `933201c`). CUDA needs **no attention-kernel change** for
   Gemma 4. (Metal equivalent, Mac-side, still open: add the 512 row to `metal/attn_shape_test.go`.)
2. **The prize (a fitting model).** Measure `gemma-4-E2B_q4_0-it.gguf` CPU decode ms/token (the
   own-forward `runLayersGemma4` on CPU) as the resident target-to-beat. E2B fits 8 GB and is the
   collectable prize; the SSM precedent landed 10× (300 ms → 30 ms) moving own-forward CPU → resident.
3. **The parity oracle — by PROPERTY, not by model.** Confirm a fixture set that exercises each seam
   the bridge touches: 256-local heads, 512-global heads, K=V-global (no v_proj), the 5:1 interleave,
   per-layer KV switch. The tiny MoE fixtures back `TestGemma4MoEFFN_parity`; verify (or purpose-build)
   one whose global layer actually shapes 512-dim attention weights, and extend to whole-model
   per-token logits. A fixture that loads but leaves the 512 head untouched is not an oracle for it.

**Gate for the rest of this spec:** a written answer to (1) — whether the CUDA attention kernel needs
widening for 512 or only the uniform/dispatch does. Everything else is per-layer plumbing + the MoE
delta, both with precedent.

---

## Phase 1 — characterize the Gemma-4 own-forward vs the resident runner

**Layer composition** (`runLayersGemma4`, forward_gemma4.go): 30 layers, a **5:1 interleave** of
`sliding_attention` (local) : `full_attention` (global). Two attention variants:

| | local (sliding) | global (full) |
|---|---|---|
| `head_dim` | 256 (`arch.HeadDim`) | **512** (`g4.GlobalHeadDim`) |
| `kv_heads` | 8 | **2** (`num_global_key_value_heads`) |
| RoPE | base 10k, full rotary | base 1e6, **partial** (`GlobalRotaryDim`, factor 0.25) |
| V | `v_proj` | **K=V**: `V = v_norm(k)`, scale-less, no RoPE (`lw.VFromK`) |
| window | `SlidingWindow` (1024) | none |

Buffers are sized to `maxHd = max(GlobalHeadDim, HeadDim)` (forward_gemma4.go:77); per layer
`hd = headDimAt(l)`, `nKV`, `invFreq` are selected on `isGlobalLayer(l)` (forward_gemma4.go:89-95).
Plus the sandwich-norm block, the parallel dense‖MoE FFN, `layer_scalar`, `embed_scale` (√hidden),
and the **final-logit softcap 30** (host-side, `logitsFromHidden` model.go:655).

**What the resident runner lacks (the seams — from the `gpu/` survey):**

1. **Per-layer geometry.** `nKV`, `hd`, `rotary_dim` are baked **model-level**: `m.Dims()` →
   `kvDim := nKV*hd` computed once (residency.go:138/151); `newDecodeRunner(rm, hidden, nH, nKV, hd,
   inter, …)` takes scalar geometry (residency.go:752); `attnUni` hard-codes `nH/nKV/hd/group/scale`
   in **one uniform for all attention layers** (decoderunner.go:518-521); KV caches are all sized
   `ctxCap*kvDim` with the single `kvDim` (residency.go:640-652). The eligibility gate already names
   this gap: `ropeResidentCompatible` requires local/global inv-freq **same length**
   (residency.go:177-179) and the comment at :166-169 says "a per-layer rotary width — Gemma's
   global/local head-dim split — is not handled."
2. **Per-layer KV-cache sizing.** Local layers need `8·256` KV, global `2·512` — different `kvDim`
   per layer. `NewKVCacheI8(nKV, hd)` takes one geometry today (residency.go:640).
3. ~~The attention kernel at `hd=512`~~ — **not a gap on CUDA** (Phase-0 item 1, resolved: cosine
   ≈ 1.0 at hd=512, `933201c`). Left here only to note it *is* a gap on WebGPU (`hd≤128`, 9d).
4. **The Gemma-4 MoE delta** (the resident MoE path is single-branch SwiGLU): parallel dense‖MoE
   with seven norms + `layer_scalar`; a router with a **weightless** pre-norm + learned `[hidden]`
   scale + `hidden^-0.5`, unconditional renorm, learned `per_expert_scale[nE]`; **gelu-tanh** GeGLU
   experts, not SwiGLU. (CUDA already ships `FeatGatedGELU` + `FeatMoE`, so the expert activation and
   routed block exist; the parallel-branch structure + router pre-processing + per-expert scale are
   the new wiring.)
5. **The host-side final softcap.** `Forward()` returns host `[]float32` logits (decoderunner.go:995;
   residency.go:29) — a `softcap·tanh(logits/softcap)` pass after readback, before the sampler, is
   the whole job. But `DecodeRunnerEligible` currently **declines** `FinalLogitSoftcap != 0`
   (residency.go:152-154), and `FeatLogitSoftcap` conflates it with the (unneeded) attention softcap
   — both must be resolved (the 9b.2 taxonomy split; see task-gemma4-moe.md).

**What already works and must NOT be rebuilt:** per-layer `isLocal`/window (C6), per-layer
`invFreq`/`ropeScale` (C7), the routed MoE block + GeGLU experts (CUDA `FeatMoE`/`FeatGatedGELU`),
sandwich norms + (1+w) RMS + embed-scale (CUDA `FeatSandwichNorm`/`FeatRMSAddOne`/`FeatEmbedScale`),
and the build-once heterogeneous step list (decoderunner.go:14-19).

---

## Phase 2 — design the bridge (the seam, generically named)

The pattern is fixed by precedent: **accessor on `Model` → field on `runLayer` → per-layer buffer
or per-*variant* uniform → bound in the plan loop.** Extend it from constants to geometry.

1. **Per-layer geometry on `runLayer`.** Add a per-layer variant index (e.g. `attnVariant uint8`,
   generic — local/global today, extensible to qwen35/llama4) and the per-layer `hd`, `nKV`,
   `rotaryDim`, `kEqV` it selects. Move these off the `newDecodeRunner` scalar args / model-level
   `attnUni` onto the layer, exactly as `isLocal`/`ropeScale` already moved. Keep a small **set of
   per-variant geometry uniforms** (2 for Gemma-4), deduplicated by variant the way rope uniforms
   are deduplicated by `float32` scale today (decoderunner.go:467-504).
2. **Per-layer KV-cache sizing.** Size each layer's KV by its own `nKV·hd`; K=V global layers store
   K and derive V (no v_proj upload, `v_norm(k)` at attention time). This is bounded — two sizes.
3. **The attention dispatch** (kernel confirmed sufficient, Phase-0 item 1): on CUDA this is pure
   per-layer plumbing — the kernel handles hd 256 and 512 already, so each layer just binds its own
   `hd`/`nKV`/`group=nH/nKV`/`rotaryDim` uniform instead of the shared `attnUni`. No kernel change.
   (On WebGPU (9d) the `hd≤128` cap would make even 256 a kernel change — one more reason it is last.)
4. **The Gemma-4 MoE delta**, on top of the existing resident MoE: the parallel dense-MLP branch +
   joint norm + `layer_scalar`; the router pre-norm (weightless) + learned scale + root-size; the
   `per_expert_scale` post-step. Reuse GeGLU experts (CUDA `FeatGatedGELU`) and the routed kernel.
5. **Host-side final softcap.** Apply `softcap·tanh(logits/softcap)` in the model after
   `resident.Forward()` returns, mirroring `logitsFromHidden`. Split `FeatLogitSoftcap` →
   `FeatAttnLogitSoftcap` (a real per-layer kernel; **no goinfer arch needs it** — Gemma-2 only) +
   `FeatFinalLogitSoftcap` (host-side); remove the redundant arch-shape decline at
   residency.go:152-154; declare `FeatFinalLogitSoftcap` for a backend **only once the resident loop
   demonstrably applies it** (the charter: no overclaim). This is 9b.2, gated end-to-end here.

**Generality guard:** name the variant index and geometry fields generically (`attnVariant`, not
`gemma4Local`) so `qwen35` (3:1) and `llama4` (per-layer isMoE + NoPE) drop into the same seam
later. Implement only Gemma-4; add no qwen35/llama4 code.

---

## Phase 3 — feasibility, effort, go/no-go, phase sketch

**Feasible in the current runner** with bounded additions; no structural rewrite. The one genuinely
new item is the attention head_dim width (Phase-0-gated); everything else is the established
per-layer pattern + the MoE delta, both with precedent (C6/C7 constants, C3 MoE-kind, C4 MLA-kind,
SSM mixer-kind).

**Success criterion (the spine): byte-identical-to-CPU under the repo's 3% near-tie parity rule**,
on the real fixtures — the same bar §B2's CUDA numbers passed (9/10 exact argmax, 0 hard fails).
Reference = the staged/CPU `runLayersGemma4`, already the parity oracle. Gate cosine ~1.0 / bounded
maxAbs over 100+ tokens, both variants and MoE layers included, per token (attention geometry
switches every layer, so a single-position sample is insufficient — mirror the SSM drift gate).

**Effort: M–L.** Per-layer geometry plumbing + per-variant uniforms + per-layer KV sizing (bounded),
the MoE delta (parallel branch + router + per-expert scale on top of existing MoE), the host-side
softcap + taxonomy split. No attention-kernel change (hd=512 confirmed, 933201c). Bigger than a
single C-lever; comparable to the SSM bridge.

**Phase sketch (the eventual build, parity-gated — matches the SSM P0→P7 discipline):**

> **Superseded 2026-08-01 by the Split A/B decision + the CUDA-target build (P1 landed).**
> The P-items below were drafted backend-agnostic and webgpu-flavored, before four things were
> settled; corrected here rather than rewritten:
> - **P1 is DONE.** Per-layer geometry + the `{hd, nKV, half, kEqV}` value-keyed dedup shipped
>   (`b909c86` three dims + `9ec363f` kEqV key + cache guard); `TestGeomVariants_dedup` (uniform→1,
>   each of hd/nKV/half/kEqV→2) + the byte-identical resident suite green.
> - **Split into A and B.** Split A = geometry + K=V, **no MoE** (P1–P3 + the parity gate on a dense
>   two-geometry K=V fixture, `1d6172d` / `TestGemma4DenseTwoGeom_forwardParity`, CPU oracle cosine
>   1.0); Split B = the Gemma-4 MoE delta (the old P4). Landing admission + geometry + K=V + MoE in
>   one step, on the first run that carries two live variants, would make a red un-attributable.
> - **The env gate lands FIRST, not P7.** `GOINFER_GEMMA4_RESIDENT` ships with the *first* admission
>   change, so every intermediate commit is unreachable-by-default by construction (granite's
>   `GOINFER_SSM_RESIDENT` through P5b.3/P5b.4). A last-landing gate would ship a live half-path.
> - **The `FeatLogitSoftcap` split is on Split A's critical path, not P6.** The arch-predicate
>   philosophy (residency.go:148-151 — a capability a backend hasn't shipped declines via the
>   *feature gate*, not in `decodeRunnerEligible`) forces it: you cannot narrow the `gemma4` admission
>   while the `FinalLogitSoftcap != 0` decline (residency.go:152-154) still blocks it there. So the
>   taxonomy split + host-side softcap move up beside the decline-narrowing — the entanglement is
>   structural, not a scheduling choice.
> - **On CUDA, P5's features are mostly free.** `features.go`'s `cuda` entry already ships
>   `FeatSandwichNorm` / `FeatGatedGELU` / `FeatEmbedScale` / `FeatPerLayerRoPE`, so P5 is wiring
>   existing kernels, not building them — the webgpu-flavored "build the norm/activation set" reading
>   does not apply to the 9b target (that cost is webgpu's alone, 9d).

- **P0** ✅ CUDA `attention` hd **16/64/128/256/512** confirmed (`933201c` + `cf341a3`, cosine ≈ 1.0
  — the small end covers the fixture's hd=16 local layer). Remaining P0: E2B CPU ms/token (the prize
  baseline); the property-covering fixture (local + 512-global + K=V) logit oracle — **done** (`1d6172d`).
- **P1** ✅ per-layer geometry on `runLayer` (value-keyed `{hd,nKV,half,kEqV}` dedup), per-variant
  uniforms; **non-Gemma models byte-identical** (structural via shared `*attnGeom`).
- **P2** per-layer KV-cache sizing + K=V global layers (derive V, no v_proj).
- **P3** attention dispatch per variant (per-layer hd/nKV uniform; no kernel change — hd=512 proven) → single-layer
  attention parity vs CPU, both variants.
- **P4** the Gemma-4 MoE delta (parallel dense‖MoE + router pre-norm/scales + per-expert-scale +
  gelu-tanh) → FFN-sub-block parity.
- **P5** sandwich norms + embed-scale + layer_scalar wiring; whole-model resident-vs-CPU parity on
  the tiny MoE fixtures + E2B, per token to 1k+ (cosine ~1.0).
- **P6** `FeatLogitSoftcap` split + host-side final softcap + drop residency.go:152-154; parity with
  softcap on.
- **P7** flip `decodeRunnerEligible` to admit `gemma4` **guarded** (env-gated, e.g.
  `GOINFER_GEMMA4_RESIDENT`, like `GOINFER_SSM_RESIDENT`); other own-forward families stay declined;
  confirm existing resident models bit-unchanged; regenerate `hardware-matrix.md` — Gemma-4 moves
  off `CPU` **only** in the CUDA column, never inheriting `✅` from a feature flag.
- **P8** measure realized ms/token on E2B (the fitting prize) vs the P0 CPU baseline; full `-tags
  cuda` suite no-regress. (The 26B remains VRAM-bound on the 2070 — a bigger-card / Metal follow-on.)

**Eligibility flip is GUARDED, not default** — like the SSM bridge, the first landing is env-gated
so the default production path stays staged/CPU until parity + precision are proven at the shipping
quant (int8/int4), not silently promoted.

---

## Known limits (belong in the spec, not surfacing mid-build)

- **Resident context cap 4096 (CUDA).** `cudaCtxCap = 4096` (cuda/resident.go:23) — the resident KV
  holds 4096 positions; the staged path handles longer. Gemma 4's 5 **global** layers grow their KV
  with position (only the sliding layers are bounded by the 1024 window), so a prompt past ~4096
  tokens exceeds the resident cap and falls back to staged. Gemma 4 advertises 256K context, so this
  is reachable — not a bridge blocker (short-context decode is the target), but the resident/staged
  boundary must be stated, not discovered.
- **Metal threadgroup score buffer `sc[4096]` (metal/backend.go:99).** Same ceiling, harder failure:
  the Metal attention kernel's threadgroup `float sc[4096]` overflows above 4096 attended keys. Same
  global-layer growth reaches it on a long prompt on 9c. A known-limit for the Metal port, flagged
  here so it is a design input, not a 9c surprise.
- **The 512-global head: kernel PROVEN (`933201c`), whole-model still to gate.** The CUDA
  `attention` kernel is confirmed correct at hd=512, so "bridge + 0 attention kernels" is now
  measured. What remains is *end-to-end* parity — the resident path running Gemma-4's per-layer
  512-global geometry through a property-covering fixture (P5), which is bridge-plumbing correctness,
  not a kernel question.

## Boundary (explicit — what this spec does NOT do)

- **No kernels, no engine code.** This is design only. The build is P1→P8 above, each parity-gated.
- **No `benchmarks.md` row.** A Ryzen-box / 2070 Gemma-4 number is not the turbo-fieldfare
  comparison (that is Apple-silicon Metal — Phase 9c) and must not be published as one. Any speed
  number here is a diagnostic until the build lands *and* is gated.
- **Success is correctness, not speed:** byte-identical-to-CPU under the 3% near-tie rule is the
  P-gate; ms/token is a P8 follow-on, reported separately per rig.
- **Implement only Gemma-4.** `qwen35`/`llama4` are the same class and the seam is named for them,
  but no speculative code — the generality is in the *shape* of the abstraction, not in unbuilt paths.

## Fixture resident-gate pre-flight (a RULE, learned across Split A → Split B)

Before a fixture is used to gate a resident backend at quant Q, run a CPU-only pre-flight —
two CPU forwards, zero GPU. `decoder/TestQuantNoiseFloor_gemma4MoE` is the reference. What the
pre-flight should assert changed once Split B measured a control, so the history matters:

**First cut (Split A instinct): the int4-vs-f32 "noise floor."** CPU-at-Q vs CPU-f32 on the same
tokens, on the theory that a resident backend can only agree with the CPU-Q path as well as Q agrees
with f32. This caught two *representativeness* traps:
- **Too degenerate (false negative):** HF's identity init left every scaling param at 1.0, so a bug
  applying them (×1) wouldn't move the golden. `strengthen()` (seeded non-trivial norms/scales)
  fixed it — and is what made the K=V `v_norm(raw k)` ordering observable three phases later.
- **Too small (false positive):** a hidden=64 fixture manufactured a phantom 0.82 "bug" that was
  pure W4A8 int8-activation sensitivity. hidden≥256 clears it.

**The correction (Split B): the f32-floor's 0.97 bar is UNCALIBRATED — a warning signal, not a hard
gate (and NOT irrelevant).** The resident gate compares int4-to-int4 (cuda-Q vs cpu-Q). Measured
control: the Split-A dense two-geometry fixture PASSES its resident gate at cosine **0.979** while its
own int4-vs-f32 floor is only **0.880** (`NOISE_FLOOR_CKPT=../testdata/gemma4-dense-twogeom-tiny`). So
0.97 is miscalibrated — even the known-good fixture fails it. But the floor is not decoupled: it's a
CONDITIONING proxy, CORRELATED with resident parity (hidden=64 → floor bad AND gate 0.82; hidden=256 →
floor 0.88 AND gate 0.979 — both moved together). CUDA-vs-CPU-int4 is only PARTLY common-mode: same
quantized weights, but each side rounds/groups activations its own way, and how much that difference
amplifies is exactly the conditioning the floor measures. One control point fixes 0.88-was-fine *for
that fixture*, not a general threshold. So keep the floor REPORTED and demoted to a warning — if the
resident MoE gate comes back marginal, a low floor (gemma4-moe-tiny sits at 0.79) is the first suspect.

**What actually gates a resident MoE fixture: routing.** MoE has a discrete failure mode dense
doesn't — quant noise near a router tie flips the top-k, a *different computation* not a small
numeric error. The pre-flight gates on two things:
1. **Routing agreement 100%** (expert-set match CPU-f32 vs CPU-Q), and
2. **Min routing MARGIN** (top-k boundary gap = min selected prob − max rejected prob) comfortably
   above the observed quant perturbation, so the tighter cpu-Q-vs-gpu-Q gap can't flip it either.
   Gate on the margin, not on one agreement observation — agreement can be luck, a wide margin is
   robustness. `routerMarginBuf` captures it; the gate is `min int4 margin ≥ 0.02` (~2 pts of prob).

**Constructing the margin (don't amplify random ties).** A random router gives near-tied logits int4
flips ~30% of the time (gemma4-moe-tiny started at 68.8% agreement). Scaling `router.proj` does NOT
help — the perturbation scales with it too, gap-vs-noise unchanged. Because hidden ≫ tokens, the
router inputs are linearly independent, so a **least-squares fit** of `router.proj` to the fixture's
own captured router inputs reproduces ANY target logit per token exactly (winner 7 / 2nd 6 / rest 0),
giving every decision the SAME wide margin with zero cross-token contamination — the boundary that
matters is the selected *pair* towering over the rejected pair, not the winner over the 2nd, so keep
the two selected logits close and both far above the zeros. gemma4-moe-tiny now: 100% agreement, min
int4 margin **0.12** ≫ 0.02. See `construct_router_margins` in `scripts/pin_gemma4_moe_forward.py`.
