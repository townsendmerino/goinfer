# Chapter 10 — Kernels and backends

*Everything so far has been architecture. This chapter is about the machine underneath: the
matrix multiplies that consume nearly all the time, the three backends they run on, and the
constraint that shapes all of it — no C.*

---

## The constraint

Every other serious local inference engine reaches C or C++ for the inner loops. Ollama, LM
Studio, and the Go bindings all ultimately call into llama.cpp, which means a native shared
library ships alongside the binary.

goinfer does not reach for C. The forward pass is Go, compiled by the Go toolchain, with no
cgo.

For a Go engineer, cgo's costs need no explanation: cross-compilation stops being trivial,
the build needs a C toolchain per target, the race detector gets less useful, goroutine
scheduling interacts badly with blocking C calls, and a single static binary stops being
single or static.

What you get by refusing cgo is the property the project is built around: `GOOS=windows
GOARCH=arm64 go build` produces a working inference engine. One file, no installer, no runtime
dependency. The demo binary can embed the model too, making the engine and its weights a single
downloadable artifact that boots in about half a second.

```
  refusing cgo BUYS                      refusing cgo COSTS

  cross-compile to any Go target         no intrinsics, no auto-vectorization
  one static binary, no installer        hand-written assembly per architecture
  race detector stays useful             every kernel is yours to write and tune
  no C toolchain in the build            ~60–70% of native CUDA (see the scope
  model embeddable in the binary            note at the end of this chapter)
```

What you pay is the rest of this chapter.

---

## What the kernels are

Strip away the architecture and inference is a small number of operations repeated enormously:

- **Matrix-vector multiply (GEMV)** — the decode workhorse. One position against a weight
  matrix. Memory-bound, as Chapter 5 explained.
- **Matrix-matrix multiply (GEMM)** — prefill's workhorse, and the attention kernels
  `MatmulQK` and `MatmulAV` from Chapter 8.
- **Softmax, normalization, elementwise activations** — cheap individually, not free in
  aggregate.

Nearly all of the time is in the first two. Everything in this book is ultimately a strategy
for doing fewer of them, doing them on smaller data, or doing them on better hardware.

---

## Making Go fast enough

Go gives you no intrinsics and no auto-vectorization worth the name. A straightforward Go loop
over float32 will not use the vector units, and on this workload not using the vector units is
a large factor.

The available techniques, roughly in order of how often they're reached for here:

**Hand-written assembly.** Go's assembler supports NEON on arm64 and AVX2/FMA on amd64, and
hand-written assembly is where the quantized kernels live. A `.s` file per architecture, called
from Go, with a pure-Go fallback for any architecture without a `.s` file. Hand-written assembly
is the same pattern the standard library uses for crypto and hashing.

**Layout.** Quantized weights get repacked so the assembly reads the weights sequentially.
Chapter 5's block structure — scale factor plus small integers — is arranged for the kernel's
access pattern rather than for readability. On a memory-bound workload, layout is often worth
more than instruction selection.

**Parallelism.** Go's actual advantage. Prefill splits across goroutines with no special
machinery, and Chapter 8's 3.28× from six workers is goroutine parallelism working. Getting the
same parallelism in C means a thread pool you wrote or a dependency you took.

---

## Three backends

**CPU** is the reference and the fallback. The CPU backend runs everywhere Go runs, the CPU
backend is where parity is established against HuggingFace, and the CPU backend is the only
backend guaranteed present.

**CUDA**, cgo-free, driven through the driver API rather than the CUDA runtime. Kernels are
shipped as pre-compiled PTX, and the driver compiles the PTX at load time. That detail is worth
understanding for Chapter 11's reasons: the CUDA backend is **driver-JIT**, so a driver upgrade
changes the *compiler*, not merely the runtime. When this repo's Linux box went through a distro
upgrade that moved the NVIDIA driver, the correct response was to mark every number measured
on it as stale pending re-anchor — and to re-verify the bit-identical and parity claims
separately, ahead of any throughput number.

**Metal**, also cgo-free, for Apple Silicon. Chapter 6's expert streaming and slot sweep
happened here.

The three backends are not equivalent, and the capability matrix in
`docs/capability-matrix.md` exists because "does model X run on backend Y" has a genuinely
complicated answer. Adding an architecture means implementing that architecture once per
backend, which is the standing tax of owning the forward pass rather than inheriting one.

---

## How it stands against the peers

Against current Ollama, this repo's own summary in `docs/benchmarks.md` is that goinfer is at
parity or slower almost everywhere. The cgo-free CUDA backend holds a real edge only on
tiny-model dense 4-bit decode — about 1.7× at 0.5B, where launch overhead dominates and Go's
cheaper dispatch shows — and reaches parity at 1.5B. It loses on long-context decode and
loses substantially on prefill, as Chapter 8 covered.

That is the trade stated plainly. A years-tuned CUDA kernel written by people who do only
that will beat a portable one. Owning the forward pass in Go costs throughput.

What it buys is peer-independent: no native dependency, cross-compilation to anything Go
targets, the model compiled into the binary, bit-identical decode, and a 26B-A4B model running
fully GPU-resident on an 8 GB card with every expert on the device.

Worth noting that the lane is no longer empty. `goccy/go-llama` runs llama.cpp in-process with
no cgo and no shared library, by compiling it to `wasm64-wasip1` and transpiling that to Go. It
inherits llama.cpp's kernels and model coverage outright. Its engine is single-threaded, GGUF-only,
and has no GPU backend — but the distinction between the two projects is narrower than "pure Go":
it is that goinfer implements the forward pass rather than inheriting one, which is what carries
multi-threaded CPU decode, the GPU backends, checkpoint formats beyond GGUF, and compiling the
model into the binary. No head-to-head numbers exist in either direction; neither project has
published any.

---

## What it costs

The summary figure for this chapter is the one the project is judged on: roughly 60–70% of
llama.cpp/Ollama-CUDA at equal 4-bit quantization on a GPU, in a static binary with no native
dependency that boots in about half a second.

That 60–70% needs its scope attached, because the figure is narrower and older than the figure
reads. `docs/positioning.md` records it as **dense-Qwen2/Llama residency decode only**, measured
**2026-06-08 against a 2025-era peer** — so it is not a current ratio, and it does not describe
the model families added since, because GPU residency requires dense attention and several
newer families are ineligible for residency entirely. Quoting 60–70% as a present-tense,
whole-engine number is the regime error this book keeps warning about; it is quoted here with
the label the source gives it.

Whether that trade is good depends entirely on what you are doing. If you are serving a model
at scale, the trade is bad — use vLLM. If you are shipping software that has to run a language
model on a machine you do not control, without an installer, on whatever architecture the user
has, the trade may be the only option that works.

Chapter 11 is about how you know any of these numbers are real.

---

*Sources: `docs/benchmarks.md` (lane statement, peer standings, re-anchor banner),
`docs/capability-matrix.md`, `docs/cuda-backend.md`, `docs/hardware-matrix.md`,
`docs/queue-performance.md`.*
