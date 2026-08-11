# Task: Metal MoE resident decode (full-parity)

> **ARCHIVED — a record, not instructions.** This file is closed work kept for its reasoning and
> its numbers. Checkboxes record the state at the moment it was archived: an unticked box means
> "not ticked when this closed", **not** "still to do", and nothing in `docs/completed/` is
> actionable. If you need a task, use the live docs; if something here reads as an instruction to
> a future reader, it was missed at archival — see the doc-closeout rule in
> `docs/parity-coverage-policy.md`, and move it to live policy or strike it.


Add mixture-of-experts (MoE) GPU-resident decode to the cgo-free Metal backend
(`metal/`, `-tags metal`), mirroring the proven WebGPU implementation so `FeatMoE`
can be flipped on **honestly** for every MoE arch whose other needs Metal already meets:
**Mixtral, Qwen2-MoE, Qwen3-MoE, GLM-MoE**. (DeepSeek/Kimi still decline — MLA; Granite —
SSM; Mellum — YaRN/per-layer-RoPE; Qwen3.5-MoE — DeltaNet.)

## The core design (from the WebGPU reference — `gpu/moe.go`, `gpu/moe_w4a8.go`)

The router runs **entirely on-GPU** and writes `idx[k]` / `wgt[k]` device buffers; the
host records a **fixed k dispatches** per projection, and each expert kernel reads its
expert index dynamically (`e = idx[slot]; weightRow = e*rowsPerExpert + n`). The command
buffer is **static every token** — only buffer contents change.

**Why this matters on Metal specifically:** the production path `ForwardEmbPipe` is a
pre-encoding pipelined executor (`encodeTrunkInto` records dispatches value-independently
so token t+1 encodes while t runs). A host-side top-k readback between router and experts
would force a per-token sync and destroy that pipeline. The on-GPU router is **required**,
not just elegant.

## Decoder surface (all exported — usable from the separate `metal` module)

- `m.MoEResidentParams()` → `(nE, k, inter, sharedInter, sigmoid, norm, sharedUngated, scale, nGroup, topkGroup, ok)` — one call, config is uniform across MoE layers. `decoder/residency.go:181`.
- Per-layer MoE detection: **`len(lw.Experts) > 0`** (dense-prefix / non-MoE layers have `Experts == nil`; this is also how GLM/DeepSeek `FirstKDense` and Llama-4 `isMoE[]` surface — no separate field needed).
- `lw.Router` (`linalg.WeightMat [nE,H]`, **f32** — `loadMat`, kept full-precision for selection stability), `lw.RouterBias` (`[]float32 [nE]`, nil unless sigmoid routing), `lw.Experts[e].{Gate,Up,Down}` (int8 under `int8int8`), `lw.SharedExpert.{Gate,Up,Down}` (int8), `lw.SharedGate` (`[1,H]`, **f32**).

## Kernels to add (MSL in `metal/kernels.go` `allKernels`)

1. **`gemv_wf32_a8`** — f32 weight × int8 activation GEMV, one simdgroup per row.
   `out[r] = (Σ_k wf32[r*K+k]·aq[k])·asc[0]`. Used for the **router** (rows=nE) and the
   **shared-gate** (rows=1). Keeps router weights at full precision (better than WebGPU's
   int8 router); reuses the already-quantized post-norm activation `mq/mSc`.
2. **`moe_route`** — one threadgroup, single lane. Reads `logits[nE]` (+ optional
   `bias[nE]`), does softmax **or** per-expert sigmoid scoring, optional DeepSeek
   group-limiting (nGroup>1), top-k selection (O(k·nE), selection score = score+bias,
   weight = un-biased score), optional `norm_topk_prob` renorm and `routed_scale`. Writes
   `idx[k]` (u32) + `wgt[k]` (f32). Mirrors `moeRouteWGSL` + CPU `routeExperts`/`groupLimit`.
3. **`gemv_w4a8_moe`** — indexed Stage-A W4A8 GEMV, mode-0 **overwrite** (gate|up). Weight
   row = `idx[slot]*rowsPerExpert + outRow`; output overwrites scratch. For the fused
   gate|up: `rowsPerExpert = 2*inter`.
4. **`gemv_w4a8_moe_wacc`** — indexed Stage-A W4A8 GEMV, mode-1 **weighted-accumulate**
   (down). Weight row = `idx[slot]*H + outRow`; epilogue `x[n] += wgt[slot]·acc·asc[0]`
   straight into the residual. This is the combine — no separate combine kernel.
5. **`shared_gate_combine`** — `x[i] += sigmoid(gl[0])·src[i]` for the qwen2_moe gated
   shared expert. (Ungated GLM/DeepSeek shared expert reuses the plain `gemv_w4a8_resid`.)

Kernels 3/4 are small deltas on `gemv_w4a8_sa` (add `idx`/`slot`/`rowsPerExpert` inputs;
change the weight base and the epilogue). All follow the existing MSL-string → `pipe()` →
`dispatchTG` pattern.

## `residLayer` / `Resident` extension

Per-layer (MoE layers only): `routerW Buffer` (f32 [nE,H]), `routerBias Buffer` ([nE], zeros
if none), `expGuW,expGuS Buffer` (stacked all-E fused gate|up, rows `nE*2*inter`),
`expDW,expDS Buffer` (stacked all-E down, rows `nE*H`); shared: `shGuW,shGuS,shDW,shDS`
(dense-packed like the dense FFN), `shGateW Buffer` (f32 [1,H]). A per-layer `isMoE bool`.

Stacking uses the **existing `int4Concat`** (it row-concatenates same-K int8 `WeightMat`s):
`int4Concat(d, gate_0,up_0, gate_1,up_1, …)` → one gate|up buffer; `int4Concat(d, down_0,
…, down_E)` → one down buffer. No new packing primitive.

Resident-level: uniforms `uNE,uK,uInter,uInter2(=2*inter)` + route params (`sigmoid,norm,
sharedUngated,scale,nGroup,topkGroup,hasBias`), `k` constant **slot uniforms** `uSlot[0..k-1]`
(compile-time constants, set once — keeps encode value-independent), scratch `rLogits[nE]`,
`rIdx[k]`, `rWgt[k]`, `shGl[1]` (shared gate logit). Reuse `r.gu`/`r.dq` per slot (dispatches
serialize via Metal's automatic compute-hazard tracking, same as the dense path). Widen
`r.gu` to `2*max(I,inter)` and `r.dq` to `max(I,inter)` if `inter > I`.

## Dispatch sequence per MoE layer (replaces the 4 dense FFN dispatches, `model.go:391-394`)

```
pRms                     x → mq/mSc                      (post-attn norm, unchanged)
gemv_wf32_a8   nE rows   routerW × mq → rLogits[nE]      (router logits)
moe_route      1 thread  rLogits(+bias) → rIdx[k],rWgt[k]
for slot j in 0..k-1:
  gemv_w4a8_moe          expGuW[idx[j]] × mq → gu[0:2*inter]   (fused gate|up, overwrite)
  swiglu_quant           gu → dq/dSc
  gemv_w4a8_moe_wacc     expDW[idx[j]] × dq → x += wgt[j]·…    (down, weighted accumulate)
if sharedInter > 0:
  pSA gate|up + swiglu + (ungated: gemv_w4a8_resid → x) OR
  (gated: gemv → shDown scratch; gemv_wf32_a8 shGateW → shGl; shared_gate_combine → x)
```

Dense layers keep the existing 4 dispatches; `encodeTrunkInto` branches on `L.isMoE`.

## Memory (all experts resident — no paging on this path)

W4A8 = 4 bits/weight + f16 group scales (K/32 halves/row). Per MoE layer ≈
`nE·(2·inter·H + H·inter)·0.5 B` nibbles + scales. Target **Qwen1.5-MoE-A2.7B**
(qwen2_moe: H=2048, inter=1408, nE=60, k=4, shared) ≈ **~7–8 GB** at int4 → fits 16 GB
(M1 Pro). Mixtral-8x7B (47B) does **not** fit and isn't a local target.

## Test plan (bit-exact-first, the As-cap discipline)

1. **Unit test (no model)** — synthetic small MoE layer (nE=8, k=2, H=64, inter=128,
   random int8 experts + f32 router), Metal MoE FFN output vs CPU `moeMLP` within f32
   tolerance. Cover every router variant: softmax+norm (Mixtral), sigmoid+bias (GLM),
   group-limit (nGroup>1), gated shared (qwen2_moe), ungated shared (GLM). Also a router
   idx/wgt match vs CPU `routeExperts`.
2. **E2e coherence + argmax parity** — Qwen1.5-MoE-A2.7B on Metal vs CPU. (GLM-MoE e2e is
   too big for 16 GB → hand to the Linux box / CUDA mirror.)

## Admission (only after kernels validated)

Add `FeatMoE: true` to `ResidentBackendFeatures["metal"]` (`decoder/features.go:160`).
`TestResidentBackendFeatures_noOverclaim` pins declared coverage — flip only once the
unit test + Qwen1.5-MoE e2e are green.

## Phasing

Foundation (router f32 kernel + moe_route + indexed GEMVs + unit test on Mixtral-shape) →
add sigmoid/bias/groups + shared (gated/ungated) + FirstKDense branch → e2e Qwen1.5-MoE →
flip `FeatMoE` + update README coverage + release summary + Linux-box GLM/CUDA handoff.
