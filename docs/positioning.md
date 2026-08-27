# What goinfer is for — and what it isn't

The long form of the framing summarised on the [README](../README.md): the niche it aims at, the
things it deliberately does not try to be, and how it sits next to the native engines and the
other pure-Go options.

## What makes it different

Moved from the README (2026-08-27) when the front page was shortened; unchanged.

goinfer is a pure-Go, no-cgo decoder-only LLM runtime that loads open-weight checkpoints
and runs them **in-process**. What makes it different — you don't have to choose:

- **One cgo-free static binary.** Pure Go, no cgo → cross-compiles to a single file
  (macOS / Linux / Windows, Intel + ARM). No Python, no llama.cpp `.so`, no CUDA toolkit,
  no provider API. The runtime *and*, if you want, the model in one file you `scp` and run
  offline.
- **Fast when you want it — still cgo-free.** The default build is pure-Go CPU
  (SIMD-accelerated, NEON / AVX2). Opt into a GPU backend and it *stays* `CGO_ENABLED=0`:
  **native CUDA** (cgo-free, driver-only — no toolkit; **14.6 MB** of binary against a bundled
  toolkit's gigabytes, decoding qwen2.5-coder-1.5B at **220.8 tok/s** at short context — measured
  2026-08-26 on driver 595.91.07; numbers and the peer comparison
  below), **native Metal** on Apple Silicon,
  and a portable **WebGPU** backend (runs on *any* GPU and streams bigger-than-VRAM MoE weights).
  Its standing ~60–70%-of-native figure is narrower and older than it reads: **dense-Qwen2/Llama
  residency decode only**, measured 2026-06-08 against a 2025-era peer, so it is not a current
  ratio and does not describe the families added since — residency requires dense attention, and
  Granite, Nemotron, DeepSeek, GLM, Kimi and Gemma 4 are all ineligible for it
  (`docs/benchmarks.md` §B). Going fast never costs you the single binary.
- **~20 architectures, one binary.** All four attention / sequence-mixing families —
  softmax·GQA, gated-linear (DeltaNet), state-space (Mamba-2), latent-KV (MLA) — plus dense
  and sparse-MoE, across ~20 architectures (Gemma 3/4, Qwen 2.5/3, Llama, Mistral, Mixtral,
  Qwen-MoE, GLM-4.5/4.6, DeepSeek-V2/V3 + Kimi, Phi-3/4, Granite-4.0-H, Nemotron-H, GPT-2,
  Mellum2). From safetensors, GGUF, GPTQ, or AWQ; f32 / bf16 / f16 + int8 / int4.
- **Parity-gated against the reference implementation.** Every forward pass is parity-gated
  against the HuggingFace reference (argmax-exact + logit cosine). A shared feature taxonomy
  means a backend declares a feature only when it ships the kernel — so an architecture it
  can't fully run is declined at load and served on the CPU path, rather than run with a
  feature quietly dropped. And constrained decoding masks the logits so structured output
  always fits your JSON schema (below).

### What it's for — and what it isn't

goinfer targets **single-user local inference**: one process, one machine, batch-1 decode,
deployed by copying a file. That is the axis it optimizes — `go build` with **no toolchain of
any kind** (no CUDA toolkit, no C++ compiler, no CMake, no Python), cross-compiling like any
other Go program, and every GPU fast path is gated bit-identical against its own reference path,
with all backends parity-gated against the pure-Go CPU implementation — which is itself
parity-gated against HuggingFace. Bit-identity is a **within-machine, within-OS** property: the
Metal backend compiles its MSL at runtime with the OS's own shader toolchain, so identity holds
for a given machine and OS version, not between them. The parity *gates* are what is portable;
the bytes are not.

It is **not a serving engine.** There is no continuous batching and no paged attention: a
model serves one generation at a time behind a bounded queue. If your problem is saturating a
datacentre GPU with concurrent requests, vLLM and its ports are built for that and goinfer is
not.

It is also not a provider-orchestration library (e.g. `teilomillet/gollm`) that calls remote
LLM APIs. goinfer runs the weights itself, locally, in-process.

Built on [`aikit`](https://github.com/townsendmerino/aikit)'s embedding and tensor
primitives.

Full, generated support map (every supported `model_type`, how each family is
configured — coverage axis, MoE, RoPE, norm, loaders, modality):
[docs/capability-matrix.md](capability-matrix.md) (generated from the
registry; do not hand-edit).

The Go bindings for llama.cpp still ship a native library alongside the binary.
[`goccy/go-llama`](https://github.com/goccy/go-llama) reaches pure Go by another route —
llama.cpp compiled to `wasm64-wasip1` and transpiled to Go (`goccy/llamawasm2go`, llama.cpp base
b10223) — so it also runs with no cgo, no shared library and no wasm runtime, and it inherits
llama.cpp's kernels and model coverage. Its engine is single-threaded (the bundle contains no
`pthread_create`, and `ContextParams.NThreads` in `llama.go` documents the clamp to 1), GGUF-only,
and CPU-only. goinfer implements the forward pass itself, which is what lets it decode
multi-threaded, reach a GPU, and load safetensors, GPTQ and AWQ alongside GGUF. Measured numbers, every cell with provenance:
[docs/benchmarks.md](benchmarks.md).

![Mellum2 — a 12B coding MoE running GPU-resident on an 8 GB card, in pure Go](assets/mellum2-gpu.gif)

*Bigger than your VRAM: JetBrains **Mellum2** — a 12B sparse-MoE coding model — decoding
**GPU-resident on a consumer 8 GB card**. The int4 experts stream into VRAM through a
pure-Go WebGPU backend (no CUDA, no Python, no llama.cpp); a 12B that won't fit 8 GB at
int8 runs **fully resident** at int4, ~13–21 tok/s. It writes idiomatic Go. Prequant the
weights once to a `.giw` bundle and it reloads in ~13 s
([docs/mellum2-resident.md](mellum2-resident.md)).*

*And bigger still — **Gemma 4 26B-A4B** (a 26B MoE whose ~11.4 GB of int4 experts **do not
fit 8 GB even at 4-bit**) decodes coherently on the same card, running **fully GPU-resident** —
every expert executes on the GPU, streamed from host RAM into a VRAM cache over the cgo-free CUDA
backend. Rate depends on how much of that cache fits (all rates below are on the pre-2026-08-25
driver stack and are NOT re-anchored — the 26B host↔VRAM leg has its own procedure, which the
2026-08-26 greedy sweep does not run): **11.3 tok/s at 16 slots/layer**, and
**16.98 tok/s** was measured once at 38 — a configuration not currently reproducible on this card
(see the note below, and read the slot count as part of the claim).
(Current Ollama also runs this 26B on 8 GB, but by offloading 58% to the CPU, at ~24.5 tok/s;
goinfer's distinction is all-experts-on-GPU, not that peers can't run it —
[docs/task-moe-streaming.md](task-moe-streaming.md).)*

> **Running a model larger than your card is opt-in.** Gemma-4 residency is on by default, but
> host→VRAM expert streaming is not — without it the 26B's experts must fit VRAM, and on an 8 GB
> card they do not, so the runtime declines to CPU (and says so). Reproducing the number above
> needs two environment variables:
>
> ```bash
> cd cuda && CGO_ENABLED=0 \
>   GOINFER_MOE_CACHE_EXPERTS=1 \ # stream routed experts host→VRAM (they exceed 8 GB)
>   GOINFER_MOE_CACHE_SLOTS=128 \ # per-layer cache depth. NOT optional: the default is the
>                                 # minimum that works and re-fetches every expert every token.
>                                 # Set it high — the runtime caps it to what fits (33 here).
>   go run -tags cuda ./cmd/serve --backend cuda --quant int4 \
>     --model ~/models/gemma-4-26b-a4b-it
> ```
>
> **Sizing it.** A slot count is only safe *relative to free VRAM*, so there is no universally
> correct number: a display attached, another process resident, or a longer `--ctx` all leave you
> less than a bare card. **You do not have to pick one.** The runtime measures free VRAM and caps
> the slot count, so setting it high (or leaving it at 128, "all experts") gets you the largest cache
> that actually fits — on an 8 GB card with nothing else on the GPU that is 33, and it is chosen for
> you.
>
> **If you are on an older build, this section used to say the cap could not be trusted.** That was
> accurate. Between v0.11.0 and the fix, the cap summed requested bytes while the CUDA driver charges
> each allocation whole 2 MiB quanta, so on an 8 GB card it granted 34 slots — a value that allocates
> successfully and then cannot launch, giving `CUDA_ERROR_OUT_OF_MEMORY` at the first forward and
> zero tokens. The recommended workaround was to set the count manually to 30.
>
> The cap now accounts for both costs it was missing:
>
> - **allocation granularity** — four buffers per MoE layer, each rounded up to its own 2 MiB quantum,
>   so the requirement is a step function of the slot count and is searched rather than divided. At 34
>   slots all four buffers tip a quantum at once, putting the requirement 203,816,960 B over free.
> - **the deferred first-launch reservation** — the on-GPU router kernel declares per-thread scratch,
>   and the driver backs it for the device's occupancy the first time that kernel runs. Measured on an
>   RTX 2070 SUPER: it *demands* 289,013,760 B at that launch and *retains* 138,412,032 B, and none of
>   it is visible to the free-VRAM reading the cap is computed from.
>
> **How to tell whether your build has the fix:** the cap logs the value it chose
> (`C′ cache: … capping to N`). On an 8 GB card with a bare GPU, **33 has the fix and 34 does not**.
> If you still see `CUDA_ERROR_OUT_OF_MEMORY` at the first forward, the launch error now names the
> kernel and both the requested and effective slot counts; lower `GOINFER_MOE_CACHE_SLOTS` below the
> effective value.
>
> What the slot count buys, measured on this card — the flag is not a tuning knob, it is the
> difference between the feature working and not:
>
> | slots/layer | LRU hit rate | decode | note |
> |---|---|---|---|
> | 8 (default) | **0%** | ~5 tok/s | `top_k` is 8, so each token's routed set exactly fills the cache and nothing survives to the next — the cache is **inert** |
> | 16 | 57.3% | 11.33 tok/s | eviction cycling normally |
> | 30 | *not yet measured* | *not yet measured* | highest value confirmed to run here |
> | 38 | 81.6% | **16.98 tok/s** | the published figure — see below |
>
> **About the 16.98 tok/s quoted above.** It was measured, on this card, at 38 slots. It is not
> currently reproducible here: that run had materially more free VRAM than the gates now see, and
> at the free VRAM these tests observe the cap lands at 34, which fails. The number was real and
> the configuration was narrow — it ran with roughly the forward's own demand left over. Treat it
> as the ceiling this approach reached on one occasion, not as a target to configure toward.
