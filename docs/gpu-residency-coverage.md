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

Every resident runner expresses ONE uniform per-layer block by default, in int8 W8A8:

```
input_RMSNorm → q/k/v/o GQA attention (RoPE) → +residual
             → post_RMSNorm → SwiGLU(silu) MLP  OR  sparse MoE → +residual
```

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
  before any feature is even checked (`decoder/residency.go:374`, `case a.llama4 != nil: return
  false // own forward, not yet bridged`). Needs: **iRoPE** (per-layer RoPE/NoPE interleave —
  RoPE layers use interleaved complex-pair RoPE + a parameter-free L2 QK-norm applied after
  rope; NoPE layers skip rope and apply an attention-temperature tweak to the query), and
  **top-1 sigmoid, input-scaled MoE** routing (route by raw sigmoid logit, no softmax/group-limit,
  scale the expert *input* by the gate rather than the output — a variant of the existing MoE
  router kernel, not the current one). All three are new kernels, individually small, medium in
  aggregate; no recurrence involved.
- **LFM2.5** — also an arch-shape decline (`decoder/residency.go:375-390`, `case a.lfm2 != nil:
  ... return false`), same discipline as Llama 4: 22 of 30 layers run a gated short convolution
  with a rolling window no uniform-layer runner can express. Even past that gate, no backend
  declares `FeatShortConv` (`decoder/features.go:97`) — "no resident backend implements the conv
  OR its recurrent state."
- **Laguna** — needs `FeatAttnOutputGate` (`decoder/features.go:96`): `ctx *=
  softplus(g_proj·h)` applied before `o_proj`, plus a per-layer query head count. "No resident
  backend implements either." WebGPU's Gated-DeltaNet output gate is a similar shape (fused
  double-width q-proj + sigmoid) but not interchangeable — Laguna's is a separate `g_proj`
  through softplus, spelled differently on purpose (`decoder/features.go:52-65` explains why the
  two must not be conflated).
- **Ling 3.0** — needs `FeatKDA` (`decoder/features.go:132`), Kimi Delta Attention: a delta-rule
  recurrence structurally close to Gated DeltaNet but with a per-channel decay (one value per
  state-matrix row) where DeltaNet's is a single scalar per head. "No resident backend
  implements it."
- **Granite-4.0-H** — genuinely different from the other four: it is **opt-in, not unimplemented**.
  `decoder/residency.go:395-396` declines it unless `GOINFER_SSM_RESIDENT` is set
  (`if a.granite != nil { return os.Getenv("GOINFER_SSM_RESIDENT") != "" }`), and the
  hardware-matrix generator explicitly runs with that variable forced empty
  (`decoder/hardware_matrix_test.go:41`, `t.Setenv("GOINFER_SSM_RESIDENT", "")`) — the same
  deliberate "show the off-by-default state" choice `hardware-matrix.md`'s own footnote makes
  for Nemotron-H's int4-vs-int8 policy, just not footnoted for Granite the same way. Both
  Granite-4.0-H and Nemotron-H need `FeatSSM`, which only WebGPU declares
  (`decoder/features.go:595`); with the flag set, the same arch-level gate that admits
  Nemotron-H's Mamba-2 engine on WebGPU should admit Granite-4.0-H's too — not independently
  verified here beyond the arch gate itself, flagged rather than asserted. Not a code finding:
  the generator is working as designed, showing the real default state.

### Resident on some backends, not others

- **MLA family** — DeepSeek-V2, DeepSeek-V3, Kimi K2: resident on WebGPU, CPU on CUDA and Metal.
  Both decline on the same missing feature, `FeatMLA`, declared only for webgpu
  (`decoder/features.go:594`). Named reuse path already scoped:
  [docs/completed/task-mla-cuda-residency.md](completed/task-mla-cuda-residency.md).
- **Nemotron-H** — resident on WebGPU (default-on, int4), CPU on CUDA and Metal. Declines there
  on the same missing feature as Granite-4.0-H above, `FeatSSM` (`decoder/features.go:595`, webgpu
  only) — the Mamba-2 scan a CUDA/Metal port would need already exists on WebGPU
  (`gpu/mamba2.go`), so this is a port, not a new design.
- **Gemma 4** (dense + MoE) — resident on CUDA and Metal, CPU on WebGPU. Not a missing
  `ResidentFeature` — WebGPU otherwise has everything Gemma 4 needs. The decline is
  `residentPerLayerGeomBackends` (`decoder/features.go:388`, `map[string]bool{"cuda": true,
  "metal": true}`): a layer's own head_dim/KV-head count genuinely differing from another's
  (Gemma 4's local/global split, head_dim 256 vs 512) needs a per-layer geometry seam CUDA and
  Metal both implement (`cuda/resident.go`'s `cudaLayer.hd/nKV`, `metal/model.go`'s
  `residLayer.geom`) and WebGPU's own twin fields exist but are never populated by its per-layer
  builder (`decoder/features.go:371-388`'s own comment has the full history, including the
  2026-09-08 near-miss this predicate was added to prevent).
- **Command-R / Command-R7B** — resident on CUDA and Metal, CPU on WebGPU. Missing
  `FeatLayerNorm` (declared `decoder/features.go:580` for cuda, `decoder/features.go:674` for
  metal, absent from webgpu's map) — a genuinely new kernel there (mean-centered LayerNorm, no
  learned bias), plus `FeatParallelBlock` and `FeatLogitScale`, both sequencing/host-side changes
  once the norm exists.
- **Olmo 3 / Olmo Hybrid** — resident on CUDA and Metal, CPU on WebGPU. Missing
  `FeatPostOnlyNorm` (no pre-norm; the sublayer's output is normalized before the residual add —
  declared `decoder/features.go:563` for cuda, `decoder/features.go:683` for metal) and
  `FeatQKNormWhole` (QK-norm over the whole projected vector, not per head — declared
  `decoder/features.go:568` for cuda, `decoder/features.go:684` for metal), neither declared on
  webgpu.
- **SmolLM3** — resident on CUDA and Metal, CPU on WebGPU. Missing `FeatNoPE` (declared
  `decoder/features.go:549` for cuda, `decoder/features.go:681` for metal, absent from webgpu) —
  some layers skip RoPE entirely, an all-zero per-layer invFreq table rather than a new kernel.
- **Ministral 3** — resident on CUDA and Metal, CPU on WebGPU. Missing `FeatAttnTemp` (declared
  `decoder/features.go:557` for cuda, `decoder/features.go:682` for metal, absent from webgpu) —
  a post-RoPE query scale, one scalar per position, folded into the existing rope launch on the
  backends that have it.
- **GPT-2** — resident on Metal ONLY, CPU on both WebGPU and CUDA (not just WebGPU — the one
  family here where the gap isn't purely "WebGPU is behind"). Needs `FeatLayerNorm`,
  `FeatNonGatedMLP`, `FeatLearnedPos`, `FeatOutBias`. Metal declares all four
  (`decoder/features.go:674-677`). CUDA declares `FeatLayerNorm` (`decoder/features.go:580`,
  added for Command-R) and `FeatOutBias` (`decoder/features.go:543`, added for gpt-oss) but not
  `FeatNonGatedMLP` or `FeatLearnedPos` anywhere. WebGPU declares none of the four.

## What's not here

Perf levers (int4-expert residency, GPU speculative decode) are a different axis from coverage
and tracked separately: [gpu-next-levers-assessment.md](gpu-next-levers-assessment.md). The full
per-family history behind the levers already shipped (C1–C7, the SSM engine) is archived, not
repeated here: [docs/completed/gpu-residency-coverage-2026-06.md](completed/gpu-residency-coverage-2026-06.md).
