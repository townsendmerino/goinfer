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
archs) plus **routed sparse MoE** (Mixtral-class), mixed int4/int8/f32 weights as the
checkpoint stores them. It is **decode-only** by design.

Which features it implements is not prose — it is
`decoder.ResidentBackendFeatures["cuda"]`, and admission is a subset check against the
features each arch derives from its own flags. A backend that has not shipped a kernel
declines rather than dropping the feature silently (`decoder/features.go`).

MoE specifics: the router runs on-GPU and stays **f32** while the experts are int4. That is
deliberate — the router's output steers a *discrete* choice, so a quantization error near a
tie does not perturb the result slightly, it runs a different expert. The experts are
row-stacked into one buffer per projection and selected by indexing, which keeps the launch
geometry fixed no matter what the routing picks. The always-on **shared expert** (Qwen-MoE /
GLM / DeepSeek) is **not** built yet and declines at load.

Everything off that path routes to the existing staged/CPU path automatically — never a
crash:

- **No NVIDIA driver / dlopen fails** → declines, falls back to CPU, one-line stderr note.
- **MLA / Mamba / hybrid / vision, or a MoE with a shared expert** → declines; runs staged.
- **Backend not built in** (`--backend cuda` on a binary without `-tags cuda`) → falls back
  to CPU with a note telling you to rebuild with `-tags cuda`.

## Correctness

The CUDA decode is gated **token-equivalent to the CPU decode** on a real q4_k_m checkpoint
under goinfer's 3%-near-tie rule (`TestRealForwardParity`, `TestBackendResidentWired` — the
latter drives the exact production `decoder.Load(cuda) → BuildResident → Forward` path). The
gate hard-fails on any position that diverges by more than 3% of the logit range, so the
shipped backend cannot silently regress.

MoE is gated separately by `TestMoEResidentParity` against `testdata/mixtral-tiny`, whose
required feature set is exactly `[moe]` — so a failure there is the MoE block and not some
other axis leaking in. That gate carries a **measured** control table (six deliberately
broken dispatch states) because the 3% rule alone proved **insufficient** on it: with random
fixture weights the experts are near-interchangeable, so a wrong expert contributes a
similar-magnitude random vector and argmax cannot see it. Two real bugs — the wrong expert's
down-proj, and `silu(up)*gate` instead of `silu(gate)*up` — sail past the argmax rule and are
caught only by the logit-cosine floor, which is calibrated to sit between the correct run
(0.999906) and the tightest surviving control (0.997687). Read the table before touching
either threshold.

See [`task-cuda-cgofree-spike.md`](task-cuda-cgofree-spike.md) for the full evidence
(ldd/driver-only proof, the measured executor tax, the kernel tuning), and
[`task-cuda-b-ship-checklist.md`](task-cuda-b-ship-checklist.md) for the release plan.
