# Mellum2 — resident decode on an 8 GB card (measured)

How fast Mellum2 decodes, the path it takes, and the one lever that makes it faster.
RTX 2070 SUPER (8 GB, ~7.2 GB free, ~343 GB/s effective), `-tags gpu` / Vulkan.

## Config (from the registry + the checkpoint)

`model_type: mellum` — a **12B-class sparse MoE** with sliding-window attention:

| | |
|---|---|
| hidden / layers | 2304 / 28 |
| attention | GQA **32 / 4** heads, head_dim **128** |
| **MoE** | **64 experts, top-8**, moe_intermediate **896**, every layer (no dense prefix) |
| **sliding window** | **1024**, layer pattern **3:1** (21 `sliding_attention` : 7 `full_attention`) |
| RoPE | **per-layer type** — full layers **YaRN** (θ 5e5, factor 16), sliding layers **default** (θ 5e5) |
| QK-norm | per-head RMSNorm before RoPE | 
| vocab | 98304 |
| unquantized size | 23 GB (bf16) → **int8 ≈ 11.5 GB**, int4 ≈ 6 GB |

## P0 — the "deadlock" does not exist (it's slow, not hung)

`TestMellum2_windowParity` was reported as a chan-receive deadlock in `attendBatchedHeads`
(>440 s). It is **not a deadlock**: the test **passes** (argmax exact, sample-256 cosine
**0.99620** vs the bf16 oracle) in **534.6 s**. The execution path —
`runLayers` → `causalAttention` → `attendBatchedHeads` → aikit `MatmulBT` — contains **no
channel** (the sequential kernel + a `sync.WaitGroup` matmul fan-out; aikit's `parallelCols`
returns immediately at N=0, never blocks). The >440 s is simply a **12B int8 forward on the
naive CPU backend** prefilling a long (>1024) sliding-window context, one token at a time.
It correctly skips under `-short` and when the 24 GB checkpoint is absent, so it **never
blocks CI**. (Verify-before-fixing: there was nothing to fix.)

## P1 — PATH: it goes RESIDENT (not staged)

`decodeRunnerEligible()` passes: MoE is eligible, and the three Mellum-relevant levers are
all already shipped — **C1** per-head QK-norm, **C6** sliding-window residency, **C7**
per-layer-type RoPE (YaRN-on-global vs default-on-local; `ropeResidentCompatible()` holds
because both rope tables are full head_dim → equal length). So Mellum2 loads onto the
**full-residency `DecodeRunner`** at int8 — one command buffer per token, no CPU interleave.

## P2 — MEASURED decode tok/s (resident, int8)

Driven through the resident `DecodeRunner.Forward(emb, pos)` at the model's real shapes
(`gpu/mellum2_decode_bench_test.go`):

| context | tok/s | ms/token | TWrite / TEncode / **TSync** |
|---|---|---|---|
| short (pos 64–264) | **8.9** | 112 | 0% / 4% / **96%** |
| steady sliding state (pos 1100–1300) | **6.7** | 150 | 0% / 3% / **97%** |

- **GPU-bound** — TSync ≈ 96–97% (the on-GPU-blocked wall), so the decode is genuinely
  on-device; CPU encode/write is ~3–4%.
- **Context dependence** — ~25% slower at 1.2k context. The 21 sliding layers stay fixed at
  the 1024 window; the 7 full-attention layers grow with context, so the number drifts down
  until they dominate.

## int8 spills to host — int4 (shipped) is the ~2× win

int8 weights are **~11.5 GB > the 8 GB card**, so it runs resident only because the Vulkan
driver **spills the overflow to host memory** — those expert reads cross PCIe every token
(that's the 112 ms; TSync 96% absorbs the slow host reads). **int4 (~6 GB) fits entirely in
VRAM, no spill** — so it's the real win.

int4 was blocked by one gate: the resident **stacked-MoE-expert builder was int8-only**
(`gpu/residency.go` `buildStacked`; rejected `kind "int4"`). The dense W4A8 GEMV already
existed (`gpu/gemv_w4a8.go`, used by Nemotron) — the gap was the **indexed stacked-expert**
W4A8 path. That lever is now **shipped**: a `moeExpertGEMVW4` kernel (the int8 indexed kernel's
`row = e*N+n` addressing + the W4A8 nibble/f16-group-scale unpack) + `UploadStackedExpertsInt4`
+ a `w4` dispatch branch. Parity-gated bit-identical (`TestMoEExpertW4A8_parity`, cosine
1.000000 / maxAbs 0 vs the CPU twin); int8 MoE unchanged.

### Result — int4 RESIDENT (fully in VRAM), measured

| context | int8 (host-spill) | **int4 (in-VRAM)** | speedup |
|---|---|---|---|
| short (pos 64–264) | 8.9 tok/s (112 ms) | **20.9 tok/s (48 ms)** | **2.35×** |
| steady sliding (pos 1100–1300) | 6.7 tok/s (150 ms) | **13.2 tok/s (76 ms)** | **1.97×** |

int4 TWE stays GPU-bound (TSync 92–95%). So **Mellum2 now decodes resident at ~21 tok/s
(short) / ~13 tok/s (long context) on this 8 GB card** — roughly 2× the int8 spill path, by
eliminating the PCIe traffic.

Run int4: `GOINFER_MELLUM_QUANT=int4` (default in the bench).

## Fast load — prequant int4 `.giw` (measured)

The int4 *decode* is the win, but loading Mellum2 int4-resident from the 23 GB bf16
safetensors costs **~66 s** every launch (read bf16 → quantize to group-wise int4 →
upload to VRAM). `cmd/prequant` now accepts a **safetensors directory** (not just a GGUF)
and bakes the already-quantized int4 weights into a `.giw` bundle, so subsequent loads
skip the requant:

```
go run ./cmd/prequant -quant int4 -o ~/models/mellum2.int4.giw ~/models/mellum2-unq   # one-time, ~30 s / 44 GB RAM
go run ./cmd/serve  --model ~/models/mellum2.int4.giw --backend webgpu                  # no --quant; bundle is int4
```

**Measured (RTX 2070 SUPER, `gpu.TestGIWInt4_resident`): `.giw` int4 load 45 s vs direct
int4 66 s = 1.5×, decode token-IDENTICAL, both GPU-resident int4.** The win is exactly the
~21 s requant the bundle skips — *not* "seconds." The remaining **~45 s is the GPU resident
upload itself**: the int4 nibbles are unpacked + re-packed to the GPU layout and
`UploadW4A8`'d for ~12 B params, CPU-bound, paid every launch regardless of source. Reaching
seconds would need a **GPU-layout `.giw`** (store weights already in the resident packing so
the upload is a straight copy) — a deeper follow-up, tracked in `docs/task-mellum2-fast-load.md`.

The bundle's tokenizer half carries the dir's `tokenizer.json` verbatim (serve loads it via
`tokenizer.LoadJSONBytes` when the blob isn't GGUF metadata); the resident/decode path never
reads it. (Note: the GGUF→`.giw` path still drops `rope_parameters` for MLA/YaRN families —
a separate pre-existing bug; the safetensors dir path here is unaffected because it carries
the full config.json.)

## Reproduce

```
GOINFER_MELLUM_BENCH=1 GOINFER_MELLUM_QUANT=int8int8 \
  go test -tags gpu -run TestMellum2_decodeThroughput -v -timeout 40m ./gpu

# fast-load seam (prebuild the bundle first, then):
GOINFER_GIW_INT4=~/models/mellum2.int4.giw GOINFER_GIW_SRC=~/models/mellum2-unq \
  go test -tags gpu -run TestGIWInt4_resident -v -timeout 30m ./gpu
```
