# One memory-accounting path for CUDA — `Plan("cuda")` now prices what CUDA allocates (2026-09-25)

`docs/tasks/task-memory-accounting-2026-09.md`, the CUDA half of item 1 (and item 3's check). The Metal half is
`docs/measurements/memory-accounting-metal-2026-09-25.md`; this is its twin, measured on the Linux box before anything
changed.

## What was measured first (RTX 2070 SUPER, NVIDIA driver 595.91.07, int4, tree at `6d8b317e`)

Real CUDA residents built from real fixtures, at the capacity each one resolved (`r.ctxCap`). "CUDA allocates" is the
bytes of the K/V buffers the resident actually holds (`r.kc`/`r.vc`, f32 elements × 4), which equals the resident's own
fit figure `kvBytesForCap` on every row. The right-hand columns are `decoder.Model.ResidentKVBytes("cuda", ctx, …)`
before this change — Plan's per-position formula, i.e. what `Plan("cuda")` and `fit` priced.

| fixture | layout | ctx | CUDA allocates | Plan, f32 | Plan, f16 requested | Plan, i8 requested |
|---|---|---|---|---|---|---|
| llama-tiny | dense | 4096 | 4,194,304 | 4,194,304 | 2,097,152 (0.50×) | 1,179,648 (0.28×) |
| phi3-tiny | dense | 4096 | 3,145,728 | 3,145,728 | 0.50× | 0.28× |
| qwen2.5-coder-0.5b (q4_k_m) | dense | 8192 | 201,326,592 | 201,326,592 | 0.50× | 0.28× |
| mistral-tiny-window | sliding window | 4096 | 6,291,456 | 6,291,456 | 0.50× | 0.28× |
| gemma3-vl-tiny | sliding pattern | 4096 | 2,097,152 | 2,097,152 | 0.50× | 0.28× |
| gemma3-1b (q4_k_m) | sliding pattern | 8192 | 436,207,616 | 436,207,616 | 0.50× | 0.28× |
| gemma4-dense-twogeom-tiny | per-layer geometry | 4096 | 34,603,008 | 34,603,008 | 0.50× | 0.28× |
| gemma4-moe-tiny | per-layer geometry | 4096 | 34,603,008 | 34,603,008 | 0.50× | 0.28× |
| qwen35-tiny | DeltaNet hybrid | 4096 | 1,048,576 | 1,048,576 | 0.50× | 0.28× |
| **deepseek-tiny** | **MLA** | 4096 | **1,572,864** | **3,145,728 (2.00×)** | 1,572,864 | 884,736 (0.56×) |

(Granite-H and Nemotron-H did not build a CUDA resident, so there is no Mamba row.)

Two disagreements, both in kind, not rounding:

1. **CUDA allocates f32 K/V whatever precision is requested.** Nothing in CUDA's resident build reads a KV-precision
   option; `Plan` priced a requested f16 at half and i8 at ~0.28× of what CUDA builds, so `fit --kv f16`/`--kv i8` (or a
   planner asked for f16) under-counted every CUDA model. `resolveCtxCapFit`'s own call to `Plan("cuda")` passes no KV
   flags, so CUDA's default context sizing was not affected by this half.
2. **An MLA layer holds ONE latent buffer, not K and V.** `Plan` doubled a DeepSeek / Kimi-class model's KV at f32 — this
   one DID reach CUDA's default context sizing: `resolveCtxCapFit` shrank an MLA model's context to fit twice its real KV.

Also established by the same run, and pinned: CUDA gives sliding-window layers the **full** context (the window is a
mask, not a ring buffer — the same as Metal, unlike the CPU), gives a DeltaNet layer **no** cache, and sizes Gemma 4's
two geometries per layer. On those the old formula was already exact. Per-buffer 2 MiB driver rounding is within one
quantum per buffer and is not added: the resident's own fit check (`checkKVFits` via `kvBytesForCap`) does not add it
either.

## What shipped

- `ResidentKVBytes("cuda", …)` has its own branch (`decoder/residentneed.go`, `cudaKVBytes`): f32 always, K and V at the
  full context on every attention layer, one latent buffer on an MLA layer, none on a no-attention-KV layer — exact to
  `kvBytesForCap`. `Plan("cuda")` and `fit` now price what CUDA builds. Metal's branch and the webgpu/cpu formula are
  unchanged.

## Gates

- `TestResidentKVBytes_matchesCUDAAllocation` (cuda, GPU): the allocated buffers' bytes == `kvBytesForCap` ==
  `ResidentKVBytes("cuda")` at f32, f16-requested and i8-requested, on all ten fixtures above: **10/10 exact**; each
  buffer's 2 MiB-rounded allocation within one quantum. Mutation (branch removed): every fixture fails, 0.500× / 0.281×
  for the requested precisions and 2.000× for MLA at f32.
- `TestResidentKVBytes_cudaLayout` (decoder, runs in CI): the same layout on llama-tiny and deepseek-tiny without a GPU;
  red under the same mutation.
- Item 3 (slot sizing) needed nothing: `cuda/slotcap_test.go` already pins the boundary where division over-admits
  (`TestSlotCapArithmetic_search`: division 34, search 33 on the real 26B's figures, the 4-quanta 33→34 step) and proves
  it by mutation (`TestSlotCapArithmetic_mutation`).
