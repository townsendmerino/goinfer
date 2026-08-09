# GPU (CUDA, cgo-free)

An additive, opt-in NVIDIA GPU backend that decodes **dense** models GPU-resident with
**no CUDA toolkit, no Python, no cgo** — it dlopens your existing NVIDIA driver and
JIT-compiles embedded PTX at runtime (`cuModuleLoadDataEx`). The default build stays
pure-Go CPU/WebGPU; CUDA is never the default.

## Build & run

```sh
# build the cuda submodule entrypoint (since v0.10.0 the root cmd/serve builds no backend —
# a -tags cuda on it fails the build; see the README build section)
CGO_ENABLED=0 go build -tags cuda -o goinfer-cuda github.com/townsendmerino/goinfer/cuda/cmd/serve   # or …/cuda/cmd/chat

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

### Resident context capacity (`-ctx`)

The resident K/V caches are allocated once at load, so their capacity is fixed for the process. It
defaults to **4096 positions** and is raised with `-ctx N` (or per model, `--model name=path,ctx=N`):

```
cap = min(model context window, -ctx)        # -ctx unset (0) ⇒ 4096
```

**The default is deliberately unchanged by this knob's existence.** Raising the default would
multiply every resident model's KV footprint for callers who never asked; a caller who does not pass
`-ctx` allocates exactly what they always did, and a request past 4096 still fails cleanly and takes
the staged path.

The cap costs VRAM linearly and it is **not** small: **24.0 KB/position** on qwen2.5-coder-0.5b
(24 layers × 128 kvDim × K+V × f32) and **56.0 KB/position** on the 1.5B, so 32k positions is
0.79 GB and **1.88 GB** respectively. That is checked **at load**, immediately before the caches are
allocated and after the weights are on the device, so `free` means what is actually left for KV:

- **Configured cap that does not fit → hard startup failure**, naming the cost:
  `resident context 32768 positions needs 1.88 GB of KV (56.0 KB/position across 28 layers) but only
  0.37 GB is free …`. It is reported in the startup `decode path:` line and on `/health`, and
  `-require-backend` turns it into a **non-zero exit**. An operator who asked for a capability and
  cannot have it should not discover that as a latency mystery under load.
- **Default cap that does not fit → ordinary decline** to the staged path, as it always has. This
  path must not start failing to boot on deployments that never configured anything.

Measured against the formula: at `-ctx 8192` the 1.5B's VRAM rose exactly **+224 MiB** over the
default, and at `-ctx 32768` **+1570 MiB** (predicted +1568). Depth measurements at these caps:
`docs/benchmarks.md`.

### …but that guarantee is about correctness, not speed

A missing kernel never silently produces wrong logits. It **can** silently produce a much
slower server, and the difference cost a day. Two fast paths decline *per call* and fall back
correctly but with nothing announced: the resident decode runner (`withResidency`) and the
batched prefill (`Prefiller`). The sharp case *was* `--backend cuda --quant int8int8` on a dense
model: it builds a **full resident decode path** — `ResidentActive` is true, decode runs at
~0.7× int4, everything looks healthy — and then every prompt took the sequential per-token
prefill, because the batched GEMV was int4-only. Measured on a 300-token prompt (0.5B, RTX 2070
SUPER): **TTFT 1.73 s vs 0.19 s (9×), 4.56 vs 0.22 CPU-seconds (20×)**, with no compute
hotspot — the CPU spin-waiting through 300 sequential launches. That specific decline is now
**fixed**: `gemv_w8a8_batched.cu` batches int8/int8int8/int4mix as well as int4 (§C6), so only
native-f32 weights (and the non-dense family/MoE/MLA classes) still fall back. But the lesson —
and the visibility below — stand: a per-call decline that announces nothing is indistinguishable
from a slow machine, and the remaining fallbacks are exactly as silent as this one was.

The runtime now states both resolved paths, because a decline nothing announces is
indistinguishable from a slow machine:

- **serve prints the resolved `decode path` / `prefill path`** per model at load — the paths
  it got, not the ones requested.
- **`GET /health`** carries `decode_path`, `prefill_batched`, `prefill_path`. The same three
  fields ride on each `GET /v1/models` entry as a **vendor extension** (extra keys on a schema
  goinfer does not own — the Go/Python/JS clients ignore unknown keys, but a strictly-typed
  decoder elsewhere may not; `/health` is the surface with no compatibility contract).
- **`--require-backend`** turns either decline into a **startup failure**, so a batch client
  fails at second zero instead of discovering a 9× under load. Opt-in: refusing to start by
  default would break existing deployments that are legitimately on the fallback.

The prefill report shares `prefillStaticDecline` with `prefillCore` rather than restating its
conditions, so the startup line cannot drift from the decline it describes. The residency
decline keeps its **reason** (`Model.ResidentDecline`) — module not built in, no usable device,
ineligible arch — because those three are fixed by three different actions and are otherwise
indistinguishable from outside.

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
