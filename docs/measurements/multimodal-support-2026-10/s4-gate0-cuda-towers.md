# S4 / S2 CUDA twins, Gate 0: a desk map of the CUDA tower base (2026-10-07, nobara)

Read-only research for the CUDA image towers (docs/tasks/task-multimodal-support-2026-10.md, S4 combined block), done before any CUDA tower code.
It was produced by a delegated read of the `s2-towers` goinfer worktree (HEAD `e5e5dbdf`) and the aikit `s2-tower-exports` worktree (`64bab88`).
Spot-checked against the tree by the session that wrote this: the GEMM plan's alignment condition, `planGridTower` being hard-coded to Metal, the planner's
extra-bytes reserve, the kernel lint lists, and the three registry functions. Everything else is the reader's account, and every size and FLOP count below is
arithmetic, not a measurement. Names are given as file and function, with no line numbers, so the citation lint does not have to chase them.

## What CUDA already has

- **aikit `gpu.ViT` (CUDA, `gpu/vit.cu` through `gpu/cuda_vit.go`) is most of a base.** f32 GEMM C = A·Bᵀ (`gemm_f32_reg` and `gemm_f32_tiled`, chosen by
  `GEMMF32Plan`), f32 GEMM with bias and with bias plus residual epilogues (`GEMMF32BiasPlan`, `GEMMF32BiasAddPlan`), `add_bias`, `add_vec`, LayerNorm with
  weight and bias (f64 accumulate), weight-only RMSNorm, GELU tanh and erf, `silu_mul`, a NeoX rotate-half `rope_qk` on a fused qkv buffer, and attention kernels
  that take an explicit `scale` (`attention`, `attention_tiled`, `attention_seg`, `attention_seg_tiled`, the last two with per-patch segment ranges). There is
  no batched or strided-batch f32 GEMM.
- **aikit's `gpu/qwencuda/encoder.go` is a correct-at-real-size template** for the Qwen3.5 and GLM-OCR block loop (fused qkv GEMM with bias, `rope_qk`,
  `attention_seg`, `gemm_f32_bias_add`, `silu_mul`). It has no LayerNorm or GELU, since Qwen2.5-VL is RMSNorm and SiLU.
- **goinfer's `cuda/vision_encoder.go` is Gemma 3's SigLIP tower and is not a base:** int8 W8A8, its own device and executor goroutine, host round trips for
  residual adds, no RoPE. Reusable idioms: a scratch free-list with defer-release, recover around setup, the leak test, and `ReleaseObjects` on close.
- **`cuda/attn_fused_vit.cu`** is an f16-operand `mma.sync` fused attention specialised to head dim 72, one segment of all keys. A head-dim-64 instantiation is
  mechanical, but it rounds Q, K, V and P to f16, which may threaten a 0.9999 per-token bar over 16 to 24 layers (unverified). A speed step after an f32 baseline.
- **Kernel shipping:** `cuda/build_ptx.sh` builds each `cuda/*.cu` to `cuda/testdata/*.ptx` through NVRTC, embedded in `cuda/kernels.go`. The audited trio
  (`moe`, `glue`, `gemv_fwd`) is pinned to NVRTC 12.6.85 and must never be regenerated elsewhere; a new kernel is a new `.cu` and `.ptx` at whatever NVRTC is
  present, recorded. A new embed must be in `lintedKernels` (explicit `__fmaf_rn`/`__fmul_rn` on every float multiply-add) or in `fmaLintExempt` in
  `kernel_fma_lint_test.go`, and is covered by the PTX consistency and local-memory census tests. Do not edit aikit's `vit.cu`.

## What carries over from Metal, and what is new

- **Metal's base** is `eg2Ops` (`metal/embeddinggemma2.go`): f32 GEMM, RMSNorm, rotate-half RoPE, and attention built from matmuls in 256-row blocks (headmajor
  copy, a per-block Vᵀ, scores GEMM, row softmax, PV GEMM, scatter), so memory is one block of scores and never T². `metal/gemma4_vision.go` adds pos-add,
  ClippableLinear clamps and an axial 2-D RoPE; `metal/grid_vision.go` adds an attention scale on q, biased GEMMs, LayerNorm with bias, GELU, the host Qwen3
  position table and per-segment attention. They register through `multimodal.RegisterGemma4Tower`, `RegisterQwen3Tower` and `RegisterGlmOcrTower`
  (backend-neutral, keyed by name).
- **Metal's scalar-arena trick does not port.** The CUDA encoder's `Dispatch` binds only buffers; CUDA kernels take scalars as launch arguments. The CUDA base
  needs its own launch helper with a sticky first-error latch.
- **New kernels (small):** Gemma 4's axial RoPE, ClippableLinear clamp and clamp-copy, gelu·mul; position add (or a host-computed [np,H] table and `add_vec`,
  which also avoids uploading two [10240,768] tables). Qwen3.5 and GLM-OCR need no new kernel if qwencuda's pattern is followed. The matmul-blocked attention
  helpers (headmajor, vt_block, softmax_rows, scatter_head) are needed only if Metal's attention design is copied; the alternative is `attention_seg`
  (np ≤ 12288 by its shared-memory row) or `attention_seg_tiled` (correct, but 32-thread blocks make it slow at large np).
- **GEMM plan selection is shape-dependent, and the tiny fixtures do not reach the production branches.** `gemm_f32_reg` needs M%64, N%64, K%16; the bias plans
  need K%16 and accept any M and N. Real GLM-OCR's patch embed has K = 3·2·14·14 = 1176, not a multiple of 16, so the bias plan is empty and it needs the tiled
  GEMM plus `add_bias`. Real np is rarely a multiple of 64. This is the failure class of aikit's `visioncuda` (tiny passes, real is wrong), so the real-size gates are
  mandatory and per-branch GEMM tests at real shapes are added.
- **A factory registered with the tower registries must return an error, not panic,** so serve can name the CPU fallback; `MustBuf` panics on allocation failure,
  so setup and `Hidden` recover (as `NewVisionEncoder` and qwencuda do).

## The three towers (real shapes; f32 sizes and costs are arithmetic)

| tower | real config | f32 weights | cost | scratch |
|---|---|---|---|---|
| Gemma 4 E2B | hidden 768, 12 x 64, ffn 3072, 16 layers, patch 16, <= 2520 patches | ~670 MB (605 MB with the position add on the host) | ~1.07 TFLOP at np 2520 | ~190 MB |
| Qwen3.5-0.8B | hidden 768, 12 x 64, ffn 3072, 12 layers, patch 16, merge 2, temporal 2 | ~345 MB | ~1.3 TFLOP at the serve cap (np 4096) | ~265 MB |
| GLM-OCR | hidden 1024, 16 x 64, ffn 4096, 24 layers, patch 14, merge 2, temporal 2 (patch dim 1176) | ~1.62 GB | ~6.6 TFLOP at 1 MP (np 5040); ~79 TFLOP at the 4.8 MP ceiling (np 24576, three quarters of it attention) | ~0.4 GB at 1 MP; ~2.0 GB at the ceiling |

All three fit f32 beside a small decoder at modest image sizes; GLM-OCR at its ceiling is the one that does not fit unplanned. E4B's tower shape was not looked up.

## VRAM beside the resident decoder (the thing the Qwen2.5-VL served run hit)

The tower loads lazily on the first image, after the resident decoder, and nothing reserves VRAM for it. A reserve already exists for a drafter:
`decoder.Options.ExtraResidentBytes` and `ExtraResidentKVPerPosition`, read by the resident build's weight and KV fit checks and by the context planner, set by serve
only for `--drafter` before `decoder.Load`. A tower can use it with no change in `cuda/`: serve computes a tower estimate (f32 weights from the config plus peak
scratch at the family's cap) and adds it into `ExtraResidentBytes` when a CUDA tower will attach (skipped under `-vision-device cpu`, int8, or no registered
tower). Because scratch scales with the image, the tower should also cap itself or decline an oversize image by name. A layer-streamed tower (host-pinned
weights uploaded per layer) is the alternative with no reserve.

## Gates and CUDA-specific planted defects (registered in the task doc)

Metal's gates port directly: the Gemma 4 tiny tower at grids {2,1 raster}, {3,2 shuffled}, {5,3 raster}, {5,3 shuffled} with attention block sizes 256, 6, 5, the
clamp-off control, and the real E2B on the four F2a images; the grid towers' `gvCase` list at four grids (including a two-image batch of two segments) with
norms randomised through the export's aliasing slices and five planted defects each (scale, RoPE, position grid or q/k norm, patch bias). CUDA adds:
per-branch GEMM tests at real shapes (reg, tiled, bias, bias-add), the bias dropped only in the tiled-fallback branch, padded keys not masked, all patches in
one segment, stale scratch across sizes (large, small, large), the Gemma 4 position x and y swapped, v-norm weighted instead of unweighted, and the scale
changed from 1.0. Controls that are not planted defects: two identical runs bit-identical, a ledger-leak test after close, a factory that returns an error
under allocation failure, and serve naming the CPU fallback with `-require-backend` refusing it.

## Build order

0. Serve plumbing: `planGridTower` and `chooseGemma4Tower` made backend-generic (the existing test row that pins CUDA to the CPU changes).
1. `cuda/tower_base.cu` and `cuda/tower_base.go`: the new kernels, a tower-ops struct with a launch helper, scratch and plan selection, and per-branch tests.
2. The Gemma 4 tower (the base's first user), registered as "cuda", with Metal's Gemma 4 gates and a served image turn.
3. The Qwen3.5+ and GLM-OCR towers on qwencuda's block loop, with G-S2b/c/d unchanged, on CUDA.
4. The VRAM reserve in serve (before the served gates, since it is what made the Qwen2.5-VL served run need `--kv-sessions 1 -ctx 4096`).
5. The speed record (a CUDA copy of the Mac's `run-s2-tower-speed.sh`, queued, not started).
6. Only if the record says so: a head-dim-64 fused attention with segment ranges, a batched-over-heads GEMM, CUDA Graph capture, layer streaming, MLP row-chunking.

Open questions to resolve while building: the f32 GEMM throughput on the RTX 2070 SUPER (3 to 4 TFLOP/s is an assumption), whether f16 attention operands hold
the 0.9999 bar, whether the Go driver binding alone gives thread affinity for a tower called from arbitrary HTTP goroutines, GLM-OCR's resident decoder size
for planning the 4.8 MP reserve, and the NVRTC version used for the new PTX.
