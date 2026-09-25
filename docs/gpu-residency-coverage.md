# GPU resident-decode coverage — the why/how, not the table

This is the standing residency backlog — for each family still declined somewhere, which
predicate declines it and what shipping it would take. The coverage claim itself lives
elsewhere, generated so it can never drift from the code: [docs/hardware-matrix.md](hardware-matrix.md).

## What "resident decode" is

A resident (GPU) decode path keeps the whole model on-device and runs
`Forward(embedding, pos) → logits` as one command submission per token — no CPU interleave
between layers. Each backend (`gpu/decoderunner.go` for WebGPU, `cuda/resident.go` for CUDA,
`metal/model.go` for Metal) routes a plain `Generate` through its own resident runner whenever
the architecture is eligible (`decoder.ResidentEligible`, `decoder/features.go`) and that
backend's `BuildResident` accepts the model. Anything ineligible on a given backend falls back to
that backend's staged per-matmul path, or to CPU, gracefully — declining is never silent
mis-execution.

Every resident runner expresses ONE uniform per-layer block by default:

```
input_RMSNorm → q/k/v/o GQA attention (RoPE) → +residual
             → post_RMSNorm → SwiGLU(silu) MLP  OR  sparse MoE → +residual
```

The QUANTIZATION of that block is backend-specific, not "int8 W8A8" uniformly (N-04,
`docs/audit-metal-2026-09-12.md`): Metal has no int8 GEMV kernel and re-quantises an int8/int8int8
weight to W4A8 at build time (`metal/model.go`'s `int4Buf`) rather than running it in int8 — every
Metal-resident dense projection is W4A8 regardless of the checkpoint's own quant label.

A family is eligible on a backend only if every layer collapses onto that block, or a shape the
backend has separately declared it can express (`ResidentFeature`s below). The eligibility
predicate (`Architecture.decodeRunnerEligible`, `decoder/residency.go`) is arch-level and
backend-agnostic; `residentBackendFeatures` (`decoder/features.go`) is where each backend
declares which of those shapes it actually implements — and that declared set is exactly what
`hardware-matrix.md` is generated from.

## The gaps, as of 2026-09-12

Two kinds: a family CPU on every backend (the arch itself isn't bridged onto any resident
runner), and a family resident on some backends but not others (a specific backend is missing
one feature or geometry seam). For each, the predicate that declines it and one line on the lift.

### CPU on every backend

- **Llama 4** (`llama4_text`, `decoder/forward_llama4.go`) — declines at the arch-shape gate
  before any feature is even checked (`decoder/residency.go:474`, `case a.llama4 != nil: return
  false // own forward, not yet bridged`). Needs: **iRoPE** (per-layer RoPE/NoPE interleave —
  RoPE layers use interleaved complex-pair RoPE + a parameter-free L2 QK-norm applied after
  rope; NoPE layers skip rope and apply an attention-temperature tweak to the query), and
  **top-1 sigmoid, input-scaled MoE** routing (route by raw sigmoid logit, no softmax/group-limit,
  scale the expert *input* by the gate rather than the output — a variant of the existing MoE
  router kernel, not the current one). All three are new kernels, individually small, medium in
  aggregate; no recurrence involved.
- **LFM2.5** — also an arch-shape decline (`decoder/residency.go:475-437`, `case a.lfm2 != nil:
  ... return false`), same discipline as Llama 4: 22 of 30 layers run a gated short convolution
  with a rolling window no uniform-layer runner can express. Even past that gate, no backend
  declares `FeatShortConv` (`decoder/features.go:100`) — "no resident backend implements the conv
  OR its recurrent state."
- **Laguna** — needs `FeatAttnOutputGate` (`decoder/features.go:100`): `ctx *=
  softplus(g_proj·h)` applied before `o_proj`, plus a per-layer query head count. "No resident
  backend implements either." WebGPU's Gated-DeltaNet output gate is a similar shape (fused
  double-width q-proj + sigmoid) but not interchangeable — Laguna's is a separate `g_proj`
  through softplus, spelled differently on purpose (`decoder/features.go:54-66` explains why the
  two must not be conflated).
- **Ling 3.0** — needs `FeatKDA` (`decoder/features.go:134`), Kimi Delta Attention: a delta-rule
  recurrence structurally close to Gated DeltaNet but with a per-channel decay (one value per
  state-matrix row) where DeltaNet's is a single scalar per head. "No resident backend
  implements it."
- **Granite-4.0-H** — genuinely different from the other four: it is **opt-in, not unimplemented**.
  `decoder/residency.go:495-443` declines it unless `GOINFER_SSM_RESIDENT` is set
  (`if a.granite != nil { return os.Getenv("GOINFER_SSM_RESIDENT") != "" }`), and the
  hardware-matrix generator explicitly runs with that variable forced empty
  (`decoder/hardware_matrix_test.go:41`, `t.Setenv("GOINFER_SSM_RESIDENT", "")`) — the same
  deliberate "show the off-by-default state" choice `hardware-matrix.md`'s own footnote makes
  for Nemotron-H's int4-vs-int8 policy, just not footnoted for Granite the same way. Both
  Granite-4.0-H and Nemotron-H need `FeatSSM`, which only WebGPU declares
  (`decoder/features.go:664`); with the flag set, the same arch-level gate that admits
  Nemotron-H's Mamba-2 engine on WebGPU should admit Granite-4.0-H's too — not independently
  verified here beyond the arch gate itself, flagged rather than asserted. Not a code finding:
  the generator is working as designed, showing the real default state.

### Resident on some backends, not others

- **MLA family** — DeepSeek-V2, DeepSeek-V3, Kimi K2: resident on WebGPU, CPU on CUDA and Metal.
  Both decline on the same missing feature, `FeatMLA`, declared only for webgpu
  (`decoder/features.go:664`). Named reuse path already scoped:
  [docs/completed/task-mla-cuda-residency.md](completed/task-mla-cuda-residency.md).
- **Nemotron-H** — resident on WebGPU (default-on, int4), CPU on CUDA and Metal. Declines there
  on the same missing feature as Granite-4.0-H above, `FeatSSM` (`decoder/features.go:664`, webgpu
  only) — the Mamba-2 scan a CUDA/Metal port would need already exists on WebGPU
  (`gpu/mamba2.go`), so this is a port, not a new design.
- **Gemma 4 dense** — resident on CUDA, Metal, and (as of 2026-09-17) WebGPU too: WebGPU's
  per-layer geometry seam (`gpu/residency.go`'s `ghd`/`gnKV`/`ghalf`/`gKEqV`, mirroring
  `cuda/resident.go`'s `cudaLayer.hd/nKV` and `metal/model.go`'s `residLayer.geom`), K=V
  (`attention_k_eq_v`, a new dedicated `vNorm` WGSL kernel) and the per-layer output scalar (a
  new `scaleVec` kernel) are all now populated/implemented — closing the gap
  `residentPerLayerGeomBackends` (`decoder/features.go:430`) previously described.
  **Gemma 4 MoE (26B-A4B, the parallel dense+MoE FFN) remains resident on CUDA and Metal only,
  CPU on WebGPU** — a separate, still-open gap gated by `residentGemma4MoEOK`
  (`decoder/features.go:448`), not `residentPerLayerGeomBackends`; WebGPU implements the dense
  per-layer attention geometry but not the joint dense‖MoE FFN bridge.
  <br>**Carve-out (N-04, `docs/audit-metal-2026-09-12.md`): the E2B/E4B E-models are CPU-only on
  EVERY backend, including CUDA and Metal** — the "resident on CUDA and Metal" above describes
  only the dense 12B/26B shape. E2B/E4B add per-layer embeddings (PLE, `hidden_size_per_layer_input
  > 0`), a cross-layer shared-KV pattern, and variable per-layer FFN width; `FeatGemma4EModel`
  (`decoder/features.go:100`) is declared by NO resident backend, so `decodeRunnerEligible` declines
  all three uniformly until an E-model bridge lands (the dense bridges were built PLE-free and
  would silently skip the PLE branch if admitted). `hardware-matrix.md`'s single "Gemma 4" row
  cannot distinguish E2B/E4B from the dense shape it actually measures.
- **Command-R / Command-R7B** — resident on CUDA and Metal, CPU on WebGPU. Missing
  `FeatLayerNorm` (declared `decoder/features.go:648` for cuda, `decoder/features.go:743` for
  metal, absent from webgpu's map) — a genuinely new kernel there (mean-centered LayerNorm, no
  learned bias), plus `FeatParallelBlock` and `FeatLogitScale`, both sequencing/host-side changes
  once the norm exists.
- **Olmo 3 / Olmo Hybrid** — resident on CUDA and Metal, CPU on WebGPU. Missing
  `FeatPostOnlyNorm` (no pre-norm; the sublayer's output is normalized before the residual add —
  declared `decoder/features.go:631` for cuda, `decoder/features.go:753` for metal) and
  `FeatQKNormWhole` (QK-norm over the whole projected vector, not per head — declared
  `decoder/features.go:636` for cuda, `decoder/features.go:753` for metal), neither declared on
  webgpu.
- **SmolLM3** — resident on CUDA and Metal, CPU on WebGPU. Missing `FeatNoPE` (declared
  `decoder/features.go:617` for cuda, `decoder/features.go:753` for metal, absent from webgpu) —
  some layers skip RoPE entirely, an all-zero per-layer invFreq table rather than a new kernel.
- **Ministral 3** — resident on CUDA and Metal, CPU on WebGPU. Missing `FeatAttnTemp` (declared
  `decoder/features.go:625` for cuda, `decoder/features.go:753` for metal, absent from webgpu) —
  a post-RoPE query scale, one scalar per position, folded into the existing rope launch on the
  backends that have it.
- **GPT-2** — resident on Metal ONLY, CPU on both WebGPU and CUDA (not just WebGPU — the one
  family here where the gap isn't purely "WebGPU is behind"). Needs `FeatLayerNorm`,
  `FeatNonGatedMLP`, `FeatLearnedPos`, `FeatOutBias`. Metal declares all four
  (`decoder/features.go:743-707`). CUDA declares `FeatLayerNorm` (`decoder/features.go:648`,
  added for Command-R) and `FeatOutBias` (`decoder/features.go:636`, added for gpt-oss) but not
  `FeatNonGatedMLP` or `FeatLearnedPos` anywhere. WebGPU declares none of the four.
- **Qwen2.5-VL / Qwen3-VL on Metal support full multimodal resident decode via `decoder.ResidentMRoPE`
  and `UploadKV`.** Metal resident implements `ForwardMRoPE` (`metal/backend.go`, `metal/model.go`)
  decoupling rope rotation position from KV cache position via `uRopePos`, and `UploadKV` enables
  bridging CPU-computed image prefill into resident GPU KV cache. Parity verified in
  `metal/forwardmrope_parity_test.go` and `metal/uploadkv_parity_test.go`.

## What's not here

Perf levers (int4-expert residency, GPU speculative decode) are a different axis from coverage
and tracked separately: [gpu-next-levers-assessment.md](completed/gpu-next-levers-assessment.md). The full
per-family history behind the levers already shipped (C1–C7, the SSM engine) is archived, not
repeated here: [docs/completed/gpu-residency-coverage-2026-06.md](completed/gpu-residency-coverage-2026-06.md).
