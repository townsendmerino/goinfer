# GPU (CUDA, cgo-free)

An additive, opt-in NVIDIA GPU backend that decodes **dense** models GPU-resident with
**no CUDA toolkit, no Python, no cgo** — it dlopens your existing NVIDIA driver and
JIT-compiles embedded PTX at runtime (`cuModuleLoadDataEx`). The default build stays
pure-Go CPU/WebGPU; CUDA is never the default.

## Build & run

```sh
# build with the cuda tag (the go.work stitches the ./cuda submodule)
go build -tags cuda -o goinfer-cuda ./cmd/serve      # or ./demo/chat

# select it at runtime
./goinfer-cuda --backend cuda --model qwen2.5-coder-1.5b-instruct-q4_k_m.gguf
```

`CGO_ENABLED=0` is honored throughout — the binary links only libc/libdl/libpthread; the
driver (`libcuda.so.1`) is dlopen'd at runtime. Verify with `ldd` (no `libnvrtc`, no
`libcudart`, no toolkit libs).

## Requirements

- An **NVIDIA GPU + driver** (already present if you have the card). **No CUDA toolkit.**
- Linux or Windows, x86-64.
- A **dense** checkpoint quantized to **int4/int8** (e.g. a q4_k_m GGUF). Qwen2 / Llama
  family.

## Scope (dense residency only)

The resident GPU decode path covers **dense Qwen2/Llama** decode (the `DecodeRunnerEligible`
archs), mixed int4/int8/f32 weights as the checkpoint stores them. It is **decode-only and
dense-only** by design.

Everything off that path routes to the existing staged/CPU path automatically — never a
crash:

- **No NVIDIA driver / dlopen fails** → declines, falls back to CPU, one-line stderr note.
- **Non-dense family** (MoE / MLA / Mamba / hybrid / vision) → not residency-eligible on
  CUDA (same as WebGPU); runs staged.
- **Backend not built in** (`--backend cuda` on a binary without `-tags cuda`) → falls back
  to CPU with a note telling you to rebuild with `-tags cuda`.

## Correctness

The CUDA decode is gated **token-equivalent to the CPU decode** on a real q4_k_m checkpoint
under goinfer's 3%-near-tie rule (`TestRealForwardParity`, `TestBackendResidentWired` — the
latter drives the exact production `decoder.Load(cuda) → BuildResident → Forward` path). The
gate hard-fails on any position that diverges by more than 3% of the logit range, so the
shipped backend cannot silently regress.

See [`task-cuda-cgofree-spike.md`](task-cuda-cgofree-spike.md) for the full evidence
(ldd/driver-only proof, the measured executor tax, the kernel tuning), and
[`task-cuda-b-ship-checklist.md`](task-cuda-b-ship-checklist.md) for the release plan.
