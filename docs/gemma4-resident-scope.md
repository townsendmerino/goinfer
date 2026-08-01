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
- **The one kernel unknown is `head_dim = 512`, and it is CUDA-specific — not the WebGPU ceiling.**
  On the **CUDA target (9b)**, `hd = 256` is **already proven**: Gemma 3 uses 256-dim heads and is
  `✅ resident` on CUDA (`docs/hardware-matrix.md:20`; Gemma 3 is *not* own-forward — only
  gemma4/qwen35/llama4/gptoss decline at residency.go:130). The CUDA attention kernel `attention`
  launches `GridX=nH, BlockX=128 (fixed), SharedMem=(nWin+128)·4` (cuda/resident.go:549) with `hd`
  passed as an arg and shared memory sized by *keys*, not head_dim — i.e. it **decomposes hd across
  the 128 threads** (2 elements/thread at 256), which is why 256 works. The genuinely new width is
  Gemma 4's **512-dim global head** (4 elements/thread), untested anywhere. Low-risk (the
  decomposition is hd-parametric and shared-mem is hd-independent), but **the P0 gate is a focused
  `attention` parity run at hd=512** before the plan claims "bridge + 0 attention kernels" for CUDA.
  (WebGPU's kernel *does* hard-cap `hd ≤ 128` at gpu/attention.go:52 — a real kernel rewrite, and an
  extra reason it is 9d. Metal handles 256 already, table in metal/attn_shape_test.go, but that
  table tops out at 256 — a Mac-side P0 item is to add the 512 row and run it.)
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
  local, so it exercises *neither* the real 256-local geometry *nor* (unverified) whether its weights
  actually shape 512-dim global attention. Gemma-3-resident covers 256; **nothing on this box is
  confirmed to exercise the 512-wide global head or the K=V-global path end-to-end.** The spec's
  parity plan must name the *properties* to cover — 256 local, 512 global, K=V (no v_proj), the 5:1
  interleave, per-layer KV switch — and confirm (or purpose-build) a fixture that hits each, rather
  than assuming a model that loads is a model that tests the seam.
- **Do not let the abstraction come out Gemma-4-shaped.** `qwen35` (3 linear : 1 full attention)
  and `llama4` (per-layer `isMoE[]`, per-layer NoPE) are the **same per-layer-variant class**.
  Implement **only** Gemma 4 and build nothing speculatively — but the per-layer variant index +
  per-layer geometry on `runLayer` is the general seam, so name it generically, not `gemma4*`.
- **Go/No-Go: GO.** The seam is a bounded extension of an existing pattern (per-layer geometry,
  following per-layer constants + the mixer-kind branch), plus one kernel-width question and the
  Gemma-4 MoE delta. Effort **M–L**, comparable to the SSM bridge. Parity-gated on fitting fixtures.

---

## Phase 0 — measure/verify (the feasibility questions, read-only)

Before designing, resolve the unknowns a paper analysis can't:

1. **The `hd=512` attention question on CUDA (the gate).** `hd=256` is already proven (gemma3
   resident); the CUDA `attention` kernel decomposes hd over 128 fixed threads (resident.go:549). The
   open question is the 512-dim global head. **Near-free empirical check on the 2070:** drive the
   shipped `attention` kernel at hd=512 (single head, hand-crafted q/k where the correct softmax
   weight depends on dims ≥256) against a CPU reference — if the kernel ignores the tail, the weights
   diverge. This is the one result the plan's cost model depends on: pass ⇒ "bridge + 0 attention
   kernels"; fail ⇒ a bounded kernel-width change (its own parity gate). (Metal equivalent, Mac-side:
   add the 512 row to `metal/attn_shape_test.go` and run.)
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
3. **The attention kernel at `hd=512`** (Phase-0 item 1) — CUDA proven at 256, 512 unverified;
   potentially a bounded kernel-width change (only if the P0 probe fails). NOT the WebGPU `hd≤128`
   ceiling, which is a separate, bigger 9d problem.
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
3. **The attention dispatch/kernel** (depends on Phase-0 item 1): gemma3 proves CUDA handles `hd=256`,
   so the local layers need only the per-layer uniform + `group=nH/nKV` to vary. The 512-global head
   is the P0 question: if the fixed-128-thread decomposition already covers it (likely — hd-parametric,
   hd-independent shared mem), it too is just per-layer dispatch; if not, a bounded reduction/workgroup
   widen to the max head_dim (an isolatable kernel change with its own parity gate — the one place
   this is more than plumbing). On WebGPU (9d) the `hd≤128` cap makes even 256 a kernel change.
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
softcap + taxonomy split, and — conditionally — one attention-kernel width change. Bigger than a
single C-lever; comparable to the SSM bridge.

**Phase sketch (the eventual build, parity-gated — matches the SSM P0→P7 discipline):**
- **P0** measure: the CUDA `attention` hd=512 probe (the gate — 256 already proven by gemma3); E2B CPU
  ms/token; a property-covering fixture (256-local + 512-global + K=V) logit oracle.
- **P1** per-layer geometry on `runLayer` (variant index + hd/nKV/rotaryDim/kEqV), per-variant
  uniforms; **non-Gemma models byte-identical** (guard the new path).
- **P2** per-layer KV-cache sizing + K=V global layers (derive V, no v_proj).
- **P3** attention dispatch per variant (+ kernel width change iff P0 requires it) → single-layer
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
- **The 512-global head is unproven end-to-end** (see Phase 0). Until the P0 probe + a
  property-covering fixture confirm it, "CUDA is bridge + 0 attention kernels" is a hypothesis, not a
  measured fact.

## Boundary (explicit — what this spec does NOT do)

- **No kernels, no engine code.** This is design only. The build is P1→P8 above, each parity-gated.
- **No `benchmarks.md` row.** A Ryzen-box / 2070 Gemma-4 number is not the turbo-fieldfare
  comparison (that is Apple-silicon Metal — Phase 9c) and must not be published as one. Any speed
  number here is a diagnostic until the build lands *and* is gated.
- **Success is correctness, not speed:** byte-identical-to-CPU under the 3% near-tie rule is the
  P-gate; ms/token is a P8 follow-on, reported separately per rig.
- **Implement only Gemma-4.** `qwen35`/`llama4` are the same class and the seam is named for them,
  but no speculative code — the generality is in the *shape* of the abstraction, not in unbuilt paths.
