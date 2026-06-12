# Task (goinfer): GPU vision tower — the real fix for image-prefill latency

> **For:** Claude Code, in `~/mycode/goinfer` (GPU work → the RTX box; `-tags gpu`,
> cgo WebGPU). Follow-on from `docs/task-cpu-vision-prefill.md` (which established
> there is NO big CPU-only lever — int8 is a wash on AVX2). Increments ordered and
> independently shippable. Parity-gated: the GPU encoder forward must match the
> CPU/HF golden (cosine, like the existing `gpu/forward_parity_test.go`). The
> pure-Go default build is untouched — GPU is opt-in under `-tags gpu`.

## Problem & premise (measured)

A Gemma 3 image prefill is ~190 s on CPU (4096 patches × 27 SigLIP layers, f32;
`docs/task-cpu-vision-prefill.md`). The encoder is compute-bound matmul, which is
the GPU's home turf. **Measured on this box (RTX 2070 SUPER, `-tags gpu`)** at the
encoder's exact dims (M=4096):

| matmul (M=4096) | f32 naive (`MatmulBTResident`) | **int8 tiled (`MatmulW8A8Tiled`)** |
|---|---|---|
| proj K=N=1152 | 95 ms | **28 ms** |
| fc1 K=1152 N=4304 | 346 ms | **88 ms** |
| fc2 K=4304 N=1152 | 236 ms | **82 ms** |

The **tiled W8A8 kernel (`gpu/gemm.go`) is ~3× the naive f32 path** and is the one
to use. Per layer ≈ 4×28 + 88 + 82 ≈ **~280 ms**, ×27 ≈ **~8–12 s/image** (matmuls)
vs ~190 s on CPU — a **~15–20× win**. (The naive f32 GPU kernel would still be
~30 s — a 6× win — but the tiled int8 path is strictly better.)

**This is exactly where the int8 `qmat` work pays off:** int8 is a wash on AVX2
CPU but the *fast* path on GPU (the tiled W8A8 GEMM). The `qmat` abstraction
(`vision/qmat.go`) already holds per-row int8 weights + scales — the seam a GPU
backend uploads as `ResidentW8A8`.

## Architecture: how vision reaches the GPU (mirror the decoder)

`vision/` is the pure-Go root module; `gpu/` is the cgo `-tags gpu` submodule. The
decoder already solves this: `decoder/backend.go` defines a `Backend` interface +
`RegisterBackend(name, factory)`, and `gpu/backend.go` calls it from an `init()`
under `//go:build gpu`. Mirror it for vision:

- **`vision`**: a small `Backend` interface — `UploadW8A8(q []int8, scales []float32,
  N, K int) (handle, error)` + `MatmulW8A8(activations, handle, M) ([]float32, error)`
  — and a `RegisterBackend`/`SetBackend` seam. Default nil ⇒ the current CPU path,
  unchanged.
- **`gpu`**: register a vision backend (init under `-tags gpu`) wrapping
  `Context.UploadW8A8` + `Context.MatmulW8A8Tiled` (both already exist, `gpu/gpu.go`).
- **serve/agent**: a `--backend webgpu` (serve already has `--backend`; the agent
  hardcodes cpu — add a flag) wires the vision GPU backend the same way it wires
  the decoder's.

## Increment 1 — per-call matmul offload — ⚠️ MEASURED DEAD END (built + reverted)

Routing the encoder's projection/FFN matmuls to the GPU one call at a time (the
`vision.Backend` per-call seam, reusing the decoder's `QuantBackend.MatmulW8A8`)
was built, **proved bit-correct** (GPU-W8A8 == CPU cosine 1.0), and **measured: no
speedup at all** — real gemma-3-4b-it, cached forward **3m6s ≈ the CPU 171–191s**.

**Why it can't work.** A forward issues ~162 weight matmuls. Each GPU call pays
WebGPU's fixed **submit + synchronous-readback overhead (~1 s/call here)** — create
the activation buffer, dispatch, block-poll, copy back. 162 × ~1 s ≈ the whole
runtime; the isolated-kernel ~28 ms is noise next to the per-call sync. Offloading
attention too (the old "Inc 2") is the same trap with more round-trips. **The
per-call seam was reverted** (the int8 `qmat` weights + the f32-QKᵀ CPU win stay).

## ✅ RESULT — resident encoder DONE (commits 886c8fd, 5d7c572)

Built and measured on this box (RTX 2070 SUPER, 8 GB, `-tags gpu`):

| path | gemma-3-4b-it image prefill | parity vs CPU W8A8 |
|---|---|---|
| CPU (f32 QKᵀ) | ~171 s | — |
| **resident GPU encoder** | **18.8 s** (~9×; +676 ms one-time weight upload) | **cosine 1.000000** (0.999959 vs HF golden) |

The full forward runs on-device — patch-embed → 27 layers → final LN → one
readback — paying WebGPU's submit/sync ~27× (one Poll/layer to bound scratch)
instead of the per-op offload's ~162× (the dead end below). New WGSL kernels:
batched LayerNorm, gelu-tanh, bidirectional row softmax, addRows
(broadcast/elementwise bias+residual), per-head gather/scatter — composed with
the existing device matmul/quantize primitives. Wired into serve: `--backend
webgpu` attaches it via `vision.Encoder.EnableResident()` (gpu init →
`vision.RegisterResident`). Gotcha fixed: `addRows`/`gelu` grid-stride with the
dispatch capped at the 65535 workgroups/dim limit — the real model's
`np*inter/64 ≈ 275k` blocks fail `ComputePassEncoder.End()` validation otherwise
(the tiny ckpt slipped under it — small-ckpt parity ≠ real-model-safe).

**Not the ~1–2 s target.** The FFN/proj matmuls use the tiled W8A8 kernel, but
the attention QKᵀ / scores·V still use the **naive f32** `matmulF32Device` — that
is the remaining limiter. A tiled f32 (or int8) GEMM for the attention shapes
(M=N=4096, K=72) is the follow-up to close the gap toward the ~8–12 s matmul
estimate below. 18.8 s is what the naive attention kernel delivers today.

## Increment 1 (revised) — the RESIDENT encoder is the only path

Submit the whole forward as one (or few) command buffer(s): upload pixels once,
keep the [4096, hidden] activation **on-device** through all 27 layers, read back
last_hidden_state once. Pay WebGPU's overhead ~once, not 162×. This is exactly the
decoder `DecodeRunner` pattern — a new `VisionEncoder` runner in the gpu module
that holds the uploaded weights + scratch buffers and runs the SigLIP forward
on-device. Needs new WGSL kernels (the gpu module flags this as "remaining work,"
backend.go:42):

- **LayerNorm** (mean/var) — NEW (the module has RMSNorm, not standard LayerNorm).
- **Bidirectional attention** — the existing attention kernel is causal/cached for
  decode; the encoder is full bidirectional over 4096 patches.
- **gelu-tanh** — the module has silu-gated MLP; SigLIP is gelu-tanh, non-gated.
- **Resident orchestration** — patch-embed → per layer {LN, qkv, attn, out-proj,
  residual, LN, fc1, gelu, fc2, residual} → final LN, all on-device, the weight
  matmuls on the tiled W8A8 kernel (int8 `qmat`), no intermediate readback.

Build incrementally, parity-gating each stage (cosine vs the CPU encoder on the
tiny golden). Target ~1–2 s/image. This is a multi-stage GPU-kernel effort, not a
quick offload — the dead-end above is *why*.

## Non-goals

- A hand-fused single-kernel ViT (the resident-op approach in Inc 3 is enough).
- Audio / video. Qwen2.5-VL (that's multimodal P5).
- Beating cloud GPUs — a 2070 SUPER at ~1–2 s/image is the realistic target.

## Verification

- `go test -tags gpu ./gpu/...` green (existing parity tests + a new encoder gate).
- `TestSiglipEncoder_parity` GPU path cosine ≥ 0.999.
- Real gemma-3-4b-it caption via `serve --backend webgpu --vision …` matches the CPU
  caption; record encoder.Forward time in `docs/benchmarks.md`.
