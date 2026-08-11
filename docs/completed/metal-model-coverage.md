# Metal backend — model-architecture coverage & the admission bug

> **ARCHIVED — a record, not instructions.** This file is closed work kept for its reasoning and
> its numbers. Checkboxes record the state at the moment it was archived: an unticked box means
> "not ticked when this closed", **not** "still to do", and nothing in `docs/completed/` is
> actionable. If you need a task, use the live docs; if something here reads as an instruction to
> a future reader, it was missed at archival — see the doc-closeout rule in
> `docs/parity-coverage-policy.md`, and move it to live policy or strike it.


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

## Update — Phi-3 + Mistral added (2026-07-16)

- **Phi-3 partial RoPE — DONE (unit-validated)**: `rope`/`rope_f16` split parameterized by
  `rhalf = rotaryDim/2 = len(invf)` (was hardcoded `hd/2`); rotates `(d, rhalf+d)` for `d<rhalf`,
  dims `[rotaryDim, hd)` pass through. `r.half = len(RopeInvFreq)` (= `hd/2` for full rotary →
  qwen2.5 unchanged). `TestRopePartial`: rotated dims match CPU, tail untouched.
- **Mistral sliding-window — DONE (unit-validated)**: `attention`/`attention_prefill` take a
  `window` uniform; derive `winStart = max(0, nKeys-window)` in-kernel (window=0 = full causal).
  `TestSlidingWindow`: windowed-over-N == full-over-last-W (maxAbs 0.0).
- Both removed from the admission decline list. qwen2.5 decode+prefill parity UNCHANGED.

**Coverage status:** admission bug fixed; **Qwen3 / Mistral / Phi-3 all code-complete + kernel-
unit-validated** on this machine. **END-TO-END family parity is UNVALIDATED** (no Qwen3/Mistral/
Phi-3 checkpoints here) — hand off to the Linux box (see the validation prompt). Remaining
declines: MoE (graceful), MLA (graceful), Gemma (structural — sandwich norms/softcaps/GeGLU/
embed-scale, deferred), embed-scale.

## Validation results (Linux box, c1c29b6) + follow-ups (2026-07-16)

Decoder-side arch detection verified on real checkpoints (the seam that feeds Metal):
- **Qwen3-1.7B — CLEAN, safe to validate on Metal.** `HasQKNorm=true`, QNorm/KNorm len==headDim
  on all layers, `RMSAddOne=false`, no window, full rotary, `DecodeRunnerEligible=true`. GGUF and
  safetensors load paths agree exactly. Mac checkpoint: `Qwen/Qwen3-0.6B` (cheapest) or the 1.7B.
- **Phi-3-mini-4k — does NOT exercise partial RoPE** (no `partial_rotary_factor` → full rotary;
  `PartialRotary=false`). **Drop Phi-3 from real-model validation; rely on the bit-exact
  `TestRopePartial` unit test.** Real partial-rotary dense families don't exist small (GLM-4.5/4.6
  are MoE; gemma4 is per-layer RoPE).
- **Phi-3 decoder BUG (below Metal, all backends):** `phi3Architecture` (registry.go:1043-)
  never sets `Architecture.SlidingWindow` from `cfg.SlidingWindow` (2047) — cf. mistral:327.
  goinfer runs Phi-3 full-attention everywhere; correct ≤2047 tokens, diverges from HF beyond.
  **Fix belongs in the decoder (phi3Architecture); Linux box owns it (has Phi-3 + can re-bless).**
  Once fixed, Metal windowing applies for free (it reads `SlidingWindowResident()`).
- **Mistral — validate with v0.1 (sliding_window=4096), NOT v0.3 (null → window never runs →
  false pass).** Mac checkpoint: `TheBloke/Mistral-7B-Instruct-v0.1-GGUF` q4_k_m (~4.1 GB).
- **Accessors: clean** (HasQKNorm/PartialRotary/EmbedScaleResident/SlidingWindow/LayerIsLocal).

**Follow-up done:** closed the **YaRN gap** — `metalUnsupported` now declines `RopeMscaleLayer(0)
!= 1` (Metal's rope applies no attention_factor; Mellum/long-ctx would run wrong). qwen2.5
(mscale=1) still admitted, parity green.

**Revised Mac validation set:** Qwen3-0.6B + Mistral-7B-v0.1 (decode + prefill parity vs CPU).
Phi-3 partial-RoPE = unit-test-only. Pass = argmax parity in qwen2.5's band, cosine ≈0.98+.

## Qwen3-0.6B end-to-end run on Metal (Mac, this machine) + the bug it caught

Downloaded `Qwen/Qwen3-0.6B-GGUF` (Q8_0) and ran the Metal parity + generation. **Caught a real
bug:** Qwen3's `head_dim` is independent of `hidden` (0.6B: `nH*hd=2048 ≠ H=1024`), so the o-proj
GEMV's `K=nHhd=2048` overflowed the Stage A staging scratch `threadgroup short As[1536]`. qwen2.5
has `nHhd==H`, hiding it. Would have broken Qwen3-1.7B (H=2048), Mistral-7B (nHhd=4096), any
large-H model. **Fixed** with dynamic `[[threadgroup(0)]]` As sizing (per-dispatch, no qwen2.5
regression) — `dispatchTG` + `setThreadgroupMemoryLength`.

**Post-fix status:**
- **Qwen3-0.6B generates coherently on Metal** ("The capital of France is Paris, and … Moscow …").
  QK-norm confirmed correct (removing it → cos 0.14). ✓ path works.
- Teacher-forced parity 15/24 cos 0.93 (vs qwen2.5-1.5b 20/24 0.989, qwen2.5-**0.5b** 19/24 0.991).
  The gap is NOT a systematic bug (coherent generation): pos-0 cos 0.44 is an arbitrary-out-of-
  context teacher-forced token, the rest cos 0.87–0.96 = int4-vs-int8 on a 0.6B Q8_0 model. The
  hardcoded parity ids were chosen for qwen2.5's vocab and are adversarial for Qwen3.
- **For full rigor, the Linux box should run Qwen3-1.7B** (bigger → tighter parity confirms
  no residual bug). Metal harness ready: `GOINFER_METAL_MODEL=<gguf> go test ./metal -run
  TestRealModel...` on a Mac. (Metal ≠ Linux; needs a Mac.)

**Coverage bottom line:** admission bug fixed; **Qwen3 works end-to-end on Metal** (the As-cap
bug that would have broken all real Qwen3/Mistral sizes is fixed); Mistral/Phi-3 kernels
unit-validated (Mistral needs a v0.1 Mac run; Phi-3 partial-RoPE unit-only). YaRN declined.

## Mistral / large-K note (2026-07-16)

Mistral-7B-v0.1 downloaded, but a full int8int8 parity run doesn't fit comfortably in 16 GB
(int8 weights ~7 GB + int4 resident ~3.5 GB + CPU-reference forwards → heavy swap). Rather than
thrash the machine, validated what a Mistral-7B run would actually stress:
- **As-cap fix at K=4096** (Mistral's nH·hd = 32·128 = 4096): `TestSAGemvLargeK` — the production
  Stage-A GEMV with dynamic As at K=4096 is **bit-exact vs CPU (cos 1.0)**. Same mechanism as the
  Qwen3 K=2048 case (validated by coherent generation), now confirmed at 4096.
- **Sliding-window path**: `TestSlidingWindow` (windowed==full-over-last-W). Note the decode
  parity test at ctx~30 wouldn't exercise the window anyway (winStart=0 until ctx>4096).
- **Dense path**: identical to qwen2.5 (validated).

So Mistral-7B's Metal components are all validated; a full end-to-end 7B coherence run wants a
Mac with >16 GB (or a smaller windowed model — none exists small). Deferred, not blocking.

## Round-2 outcomes (Linux box) + Metal refit (2026-07-16)

- **Admission taxonomy — shared** (`decoder/features.go`, Linux box): my backend-local
  `metalUnsupported` is replaced by the registry-derived `ResidentFeature` subset check —
  `MissingResidentFeatures(ResidentBackendFeatures["metal"])`, one source of truth across
  backends. Metal's declared set = {qk-norm, sliding-window, partial-rotary}; everything else
  (YaRN, embed-scale, softcaps, MoE, MLA, SSM, sandwich-norm, …) now declines automatically —
  strictly more coverage than my hand-rolled switch. qwen2.5 + Qwen3 verified admit correctly.
- **CUDA immune to the As-cap bug class** (audited): dynamic `extern __shared__` sized from
  dims; o-proj/down take K from the weight (`wt.K = Cols()`), never a dim constant. Latent limit:
  fused shared request crosses the 48 KB block cap at H≥~9216 (hard launch error, not corruption).
- **Phi-3 SlidingWindow decoder bug FIXED** (Linux box, 40e2a38): `phi3Architecture` now wires
  `cfg.SlidingWindow` + all-local `layerIsGlobal`. Since Metal reads `SlidingWindowResident()`,
  **Phi-3 windowing works on Metal for free.** (GGUF conversions legitimately drop the window →
  declare none.)
- **Decoder Qwen3 correctness confirmed independently**: `decoder/qwen3_test.go:
  TestQwen3_forwardParity` passes (QK-norm application, head_dim, weights on the CPU path). With
  Metal's coherent generation, this makes the Metal 0.6B parity gap int4-on-tiny-Q8_0 + adversarial
  ids, not a decoder/QK-norm bug. A CUDA-Qwen3-1.7B reference would fully close it, but needs CUDA
  QK-norm implemented first (offered by the Linux box).

## Qwen3 gap RESOLVED — not a Metal bug (CUDA cross-check, 420906d)

The Linux box implemented CUDA QK-norm and ran Qwen3-1.7B on CUDA, which answers the Metal
0.6B parity question conclusively — three independent lines:
1. **Ratio match across unrelated backends**: CUDA Qwen3-1.7B 5/8 = 62.5% (pre-f64-fix) ==
   Metal Qwen3-0.6B 15/24 = 62.5%. Two different kernels landing on the identical ratio points
   UPSTREAM of both — methodology/model, not the GPU kernels.
2. **Quantization ruled out**: CUDA's gate compares CUDA-int4 vs CPU-int4 (same weights), and
   still showed the looseness — so it's backend-vs-CPU numerics + adversarial teacher-forced ids
   (out-of-distribution → flat logits → near-tie argmax flips), NOT int4-vs-int8.
3. **Coherent generation** on both backends (Metal + CUDA), including Qwen3's `<think>` block.

**f64 note (not portable to Metal):** the CPU accumulates QK-norm's sum-of-squares in float64
(decoder/rmsnorm.go); on CUDA, matching that (f64 accumulate, f32 cast) recovered one near-tie
position (5/8→6/8). **Metal GPUs have no `double`** — MSL can't do f64 — so this doesn't port.
Metal's qk_norm already reduces in f32 (not f16) with a pairwise tree, which is the best
available; the residual near-tie gap is a hardware precision limit, not a bug.

**Coverage conclusion:** Qwen3 confirmed correct on Metal (+ CUDA). Admission unified on the
shared `decoder/features.go` taxonomy (refit done, 10a32f9). Qwen3/Mistral/Phi-3 coverage
complete for the dense path; Gemma/MoE/MLA/YaRN correctly decline. *(Superseded below: MoE and
Gemma 3 have since landed resident on Metal — see 2026-07-17.)*

## Resolved — MoE + Gemma 3 now resident on Metal (2026-07-17)

The two "structural, defer" rows above are done. `decoder/features.go["metal"]` now declares
nine features; `TestResidentBackendFeatures_noOverclaim` pins them.

- **MoE — resident** (`metal/moe.go`): on-GPU router + row-stacked indexed-expert W4A8 GEMVs +
  shared expert. Mixtral / Qwen2-MoE / GLM-4-MoE run resident; only YaRN-mscale (Mellum) and MLA
  (DeepSeek/Kimi) still decline.
- **Gemma 3 — resident** (`FeatSandwichNorm, FeatGatedGELU, FeatRMSAddOne, FeatEmbedScale,
  FeatPerLayerRoPE`; commits 38a2b7c + 471a418). `TestGemma3ResidentParity` live and passing
  (logit cosine 0.9117 vs CPU-int4, int4-hostile 0.88 bar), generates " Paris." coherently.

  **Root cause of the dormant residual (worth remembering for any GeGLU family):** Metal's
  `glu_act` evaluated GELU-tanh as `tanh(0.7978·(x + 0.044715·x³))` UNCLAMPED. Gemma's `<bos>`
  builds a massive-activation gate `x≈12` → the tanh argument reaches ≈73 → MSL's `tanh`
  overflows its internal `exp(2·73)` to **NaN** → `swiglu_quant` quantizes NaN to int8 0 → the
  single channel that builds the sink's massive activation is silently dropped, collapsing the
  int8 scale 32× and cratering every downstream context. SiLU has no `tanh`, so only GeGLU broke
  while SwiGLU (Qwen) shipped clean. Fix: clamp the tanh argument to ±15 (tanh saturates by
  |arg|≈9, so exact for every real input). One line.

  **Note the mechanism is precision-independent:** the NaN comes from the tanh argument
  overflowing a naive internal `exp`, which overflows in f32 too — not an f16 issue. CUDA never
  hit it because its `tanhf` intrinsic saturates large arguments correctly; a manual
  `(exp(2x)-1)/(exp(2x)+1)` on any backend would reproduce it.

  **How it was found:** a multi-turn cross-box hunt (Metal ⇄ CUDA-as-f32-oracle) that refuted
  six hypotheses — weights, o-proj chain, attention block, KV precision, sandwich norm, MLP
  down-proj — via a matched-input attention confirmer, per-sublayer/per-channel capture
  (`ForwardSubCapture`, `LayerKVForTest`, the CUDA `subCap` seam), and a shipped-kernel isolation
  test, before pinning it to the one activation call. The discipline that mattered: every CUDA
  fork checked against f32 truth, which caught two wrong verdicts (weight double-quant, then f16
  KV) before they shipped.

Remaining Metal declines: YaRN-mscale (Mellum), MLA (DeepSeek/Kimi), SSM, logit-softcap
(Gemma 2, Gemma 4 — the latter also has its own forward).
