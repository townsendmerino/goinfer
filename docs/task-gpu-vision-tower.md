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

## Increment 1 — matmul offload (per-call transfer): ~190 s → ~10 s

The encoder forward stays structured as it is; only the projection/FFN matmuls go
to the GPU.

- At load (when a GPU backend is set + `--vision-quant int8` implied for GPU),
  upload each `qmat`'s int8 weights as a resident `ResidentW8A8`; keep the handle.
- In `Encoder.Forward`, route `qmat.matmul` (the 4 attention projections + fc1/fc2)
  through the backend: quantize the activation rows to int8 (`linalg.QuantizeRowsInt8`,
  already used by W8A8), call the GPU tiled kernel, read back. LayerNorm, softmax,
  GELU, the patch-embed conv, and QKᵀ/scores·V stay on CPU for now.
- **Gate:** `TestSiglipEncoder_parity` under `-tags gpu` cosine ≥ 0.999 (W8A8
  tolerance) vs the HF golden; end-to-end `gemma3_vl` image caption unchanged.
  Re-time `encoder.Forward` on real gemma-3-4b-it; land it in `docs/benchmarks.md`.
- Expect ~8–12 s (matmuls) + CPU layernorm/attention + per-call transfer overhead.

## Increment 2 — offload QKᵀ / scores·V too

The bidirectional attention is two more big matmuls per head (M=4096). Route them
through the GPU (f32 `MatmulBT` or a W8A8 of the activations); softmax stays CPU
(cheap) or moves to a small kernel. Shaves the remaining CPU matmul time.

## Increment 3 — resident encoder (activations stay on-device): ~10 s → ~1–2 s

The big remaining cost is shuttling the [4096, hidden] activation CPU↔GPU between
every op (162 round-trips/image). Keep the activation resident on the GPU across
the whole forward — upload pixels once, run all 27 layers on-device, read back the
last_hidden_state once — exactly what the decoder's `DecodeRunner` does for decode.
This needs on-device LayerNorm / softmax / GELU kernels (the decoder has RMSNorm +
gated-MLP kernels to crib from; LayerNorm = mean/var is new). Biggest effort, the
path to "feels like the hosted app."

## Non-goals

- A hand-fused single-kernel ViT (the resident-op approach in Inc 3 is enough).
- Audio / video. Qwen2.5-VL (that's multimodal P5).
- Beating cloud GPUs — a 2070 SUPER at ~1–2 s/image is the realistic target.

## Verification

- `go test -tags gpu ./gpu/...` green (existing parity tests + a new encoder gate).
- `TestSiglipEncoder_parity` GPU path cosine ≥ 0.999.
- Real gemma-3-4b-it caption via `serve --backend webgpu --vision …` matches the CPU
  caption; record encoder.Forward time in `docs/benchmarks.md`.
