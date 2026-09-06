# goinfer

**Run open-weight LLMs in pure Go — one cgo-free static binary, portable by default and
native-GPU-fast when you want it.** 27 model families, HuggingFace-parity-gated, with
schema-constrained structured output. No Python, no llama.cpp, no CUDA toolkit.

![goinfer chat — an entire LLM in one file](docs/assets/demo.gif)

*An entire 1.5B LLM in one file — instant boot (~0.4 s), <100 MB heap, runs offline. Writes correct
generic Go and **cannot** emit invalid JSON. No cgo, no Python, no model download.*

<sub>Recorded on an Apple M1 Pro (the visible `linux-amd64` filename is a leftover from the tape's
usual render target — a `darwin-arm64` binary is what actually ran); a desktop x86 CPU measures
roughly half the on-screen tok/s on the identical harness — see
`docs/measurements/demo-chat-macbook-2026-08-22.md`.</sub>

## Install

Two ways in. **Download a binary** — nothing to build, no Go toolchain:

```bash
# macOS arm64; swap the suffix for your platform
curl -fsSL -o goinfer-serve https://github.com/townsendmerino/goinfer/releases/latest/download/goinfer-serve-darwin-arm64
chmod +x goinfer-serve
```

Or **build from source** with Go 1.27+:

```bash
# <!-- smoke --> installs as `serve` (the directory name); rename it if you want
go install github.com/townsendmerino/goinfer/cmd/serve@latest
```

> That builds the **CPU** server. For the GPU on your machine, build the backend's own
> entrypoint — `-tags metal` on `cmd/serve` does *not* work and fails the build saying so:
>
> ```bash
> go install -tags metal github.com/townsendmerino/goinfer/metal/cmd/serve@latest   # macOS
> go install -tags cuda  github.com/townsendmerino/goinfer/cuda/cmd/serve@latest    # Linux + NVIDIA
> ```
>
> The downloaded `goinfer-serve` assets already have this built in — Metal on macOS, CUDA on
> Linux. `goinfer-serve --version` prints which backends a given binary carries.

**Using it as a library?** `go get github.com/townsendmerino/goinfer` fetches the module but is
**not enough to build against** — the packages live in their own import paths, and you will get
`missing go.sum entry for module providing package …`. Get the packages you import:

```bash
# <!-- smoke --> from inside your own module (`go mod init …` first)
go get github.com/townsendmerino/goinfer/decoder@latest github.com/townsendmerino/goinfer/tokenizer@latest
```

See [`examples/embed/main.go`](examples/embed/main.go) for a complete 40-line program.

## Download and run

Binaries on the [latest release](https://github.com/townsendmerino/goinfer/releases/latest)
(macOS / Linux / Windows, Intel + ARM). Sizes are the darwin-arm64 assets of v0.16.0:

| asset | size | what it is |
|---|---|---|
| `goinfer-serve-<os>-<arch>` | ~16 MB | the **server** — OpenAI + Anthropic APIs, web UI, GPU built in |
| `goinfer-chat-<os>-<arch>` | 8.3 MB | the single-shot runtime; point it at your own GGUF |
| `goinfer-chat-0.5b-<os>-<arch>` | 652 MB | runtime **and** model in one file — no download, no install |
| `goinfer-chat-1.5b-<os>-<arch>` | 1.81 GB | same, with the 1.5B coder model |

```bash
# model included — nothing else to fetch
./goinfer-chat-1.5b-darwin-arm64

# or bring your own GGUF
./goinfer-chat-darwin-arm64 --model ~/models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf
```

Don't have one yet? The runtime can fetch a GGUF straight from HuggingFace — no extra tool
to install, and no `huggingface-cli`:

```bash
# see what a repo publishes
./goinfer-chat-darwin-arm64 pull Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF

# fetch one quant (case-insensitive; verified against the sha256 HuggingFace declares)
./goinfer-chat-darwin-arm64 pull Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF:q4_k_m

# or the models goinfer itself vets and pins
./goinfer-chat-darwin-arm64 pull demo:1.5b
```

Interrupted transfers resume where they stopped. `goinfer-serve pull …` is the same command.

Or skip the separate step entirely — `--model` takes the same reference and fetches it on first
use, so one command goes from nothing to a running endpoint:

```bash
goinfer-serve -model hf:Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF:q4_k_m
goinfer-chat  --model demo:0.5b
```

A plain path still means exactly what it always did; only the `hf:`/`demo:` prefixes are new.

It lands in your user cache dir and prints the exact `--model` command to run it. Anonymous
only: a gated repo is detected before the transfer starts and named, rather than failing after
a multi-gigabyte download — community GGUF re-uploads are usually ungated and work directly.

Prefer a browser? `serve -web` adds a local UI at `http://127.0.0.1:8080` — chat with the loaded
model, browse a HuggingFace repo, and pull a checkpoint with live progress:

```bash
goinfer-serve -web -model ~/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf
```

One embedded HTML file, no external assets, so it works offline like everything else here. Off by
default, and `-web` alone is enough to start with no model at all — which is how you use it to go
and fetch your first one.

### Bake any model into its own single file

The two pre-built tiers above are just this pipeline run for two models we picked. From a source
checkout you can run it for **any** supported checkpoint, for any OS/arch — something no other
local runner will do for a model that isn't on its curated list:

```bash
go run ./demo/chat pull bartowski/google_gemma-3-4b-it-GGUF:Q4_K_M -embed darwin/arm64 linux/amd64
# → demo/chat/dist/goinfer-chat-google_gemma-3-4b-it-{darwin-arm64,linux-amd64}
```

Out comes a static, cgo-free binary with the weights inside it: no runtime, no download, no
install. Air-gapped machines, workshops, handing a demo to a colleague. The binary is model-sized,
and the model's licence travels with it — if you redistribute one, that licence is yours to satisfy.

From source, against any supported checkpoint (the chat template is applied automatically):

```bash
go run ./demo/chat --model ~/models/gemma-4-E2B_q4_0-it.gguf
```

## Running a model bigger than your RAM

A 30B-class MoE does not fit in 16 GB, and loading it anyway will drive your machine into swap
before anything says so. `-stream-weights` is the answer, and it is not a fallback mode — it is
how these models are meant to run here:

```bash
# <!-- smoke-help --> pages weights on demand out of an mmap'd .giw
goinfer-serve -stream-weights -weight-cache 6GiB -model ~/models/qwen3.5-35b-a3b-q4_k_m.gguf
```

**Rule of thumb:** if the checkpoint file is larger than about half your physical RAM, use
`-stream-weights`. Resident memory is then capped near `-weight-cache` rather than the model
size — a 35B-A3B runs in ~16–20 GB of machine instead of the ~21 GB the weights alone would
need, because only the experts a token actually routes to are resident.

Measured on an M1 Pro / 16 GB with a 21 GB 35B-A3B: without the flag, **+7.8 GB of swap in five
seconds**; with it, RSS peaked at **8.95 GB** and fell back to 2.7 GB, with zero swapouts
([`docs/measurements/cold-user-2026-09-06.md`](docs/measurements/cold-user-2026-09-06.md),
scenario D).

**This is `goinfer-serve`'s job, not `goinfer-chat`'s.** The single-shot chat runtime holds all
weights resident by design; it has no `-stream-weights`. If your model is bigger than your RAM,
reach for the server.

## A Go struct the model cannot violate

Derive a JSON Schema from a Go struct, constrain generation to it, and
`json.Unmarshal` the result — the model **physically cannot** emit JSON that
doesn't fit the struct. The constraint is a logit mask over goinfer's incremental
byte-level grammar: at every step, tokens that would break the schema are set to
−∞, so an invalid token is *unreachable* (not retried — impossible).

```go
type Person struct {
    Name string   `json:"name"`
    Age  int      `json:"age"`
    Tags []string `json:"tags"`
}

g, _ := constrain.GrammarFromStruct(Person{})       // struct → JSON Schema → grammar
sp.LogitProcessor = constrain.NewMasker(g, toks, eos).StopWhenComplete().Process

out := generate(sp)                                  // constrained decode
var p Person
_ = json.Unmarshal(out, &p)                          // shape guaranteed, not magnitude
```

Works from any JSON Schema too (`constrain.JSONSchema(bytes)`), or from the demo:
`go run ./demo/chat --model … --schema person.schema.json`. Supported subset:
objects (required + optional, `additionalProperties:false`), arrays
(`items`/`minItems`/`maxItems`), `string`/`number`/`integer`/`boolean`/`null`,
`enum`/`const`, and arbitrary nesting. A property-based test asserts that every
constrained generation validates against its schema.

## What it is, and isn't

goinfer targets **single-user local inference**: one process, one machine, batch-1 decode,
deployed by copying a file. It builds with no toolchain of any kind — no CUDA toolkit, no C++
compiler, no CMake, no Python — and cross-compiles like any other Go program.

It is **not a serving engine**: no continuous batching, no paged attention, one generation at a
time behind a bounded queue. If you need to saturate a datacentre GPU with concurrent requests,
vLLM is built for that and goinfer is not. It is also not a provider-orchestration library — it
runs the weights itself, in-process. Longer form: [docs/positioning.md](docs/positioning.md).

## What it runs

- **27 model families** (counted from the generated `docs/capability-matrix.md`, which the
  `decoder` registry produces) — Gemma 3/4, Qwen 2.5/3, Llama, Mistral, Mixtral, Phi-3, DeepSeek/MLA,
  GLM, Kimi, Granite, Nemotron, Mellum and more. Full generated map:
  [docs/capability-matrix.md](docs/capability-matrix.md).
- **All four sequence-mixing families** — softmax·GQA, gated-linear (DeltaNet), state-space
  (Mamba-2), latent-KV (MLA) — plus dense and sparse-MoE.
- **Loaders** — GGUF, safetensors, GPTQ, AWQ, and prequantized
  [`.giw` bundles](docs/giw-bundles.md).
- **Quantization** — f32, int8, int8int8, int4 (W4A8), with a HuggingFace logit-parity gate per
  family. **What a given parity run proves is scoped to the fixtures that machine has**, and a
  missing fixture skips silently rather than failing — a run reading `28 ran / 20 skipped / 0
  failed` is a pass. Measured on a MacBook 2026-08-31, all eleven GGUF-quant gates skipped for want
  of a local checkpoint while int4 and one of three int8×int8 goldens ran. Quote a run's counts, not
  the word "green": `docs/parity-coverage-policy.md` §"Scoped: a goldens green names the
  quantizations that actually RAN".
- **GPU** — WebGPU everywhere, plus cgo-free CUDA and Metal for dense and MoE models; anything
  unsupported declines at load and falls back to CPU rather than dropping a feature silently.
  See [docs/cuda-backend.md](docs/cuda-backend.md) and
  [docs/gpu-residency-coverage.md](docs/gpu-residency-coverage.md).
- **Tensor-core prompt prefill on CUDA** (new, 2026-09-05) — a fused FlashAttention-style
  attention kernel and a tensor-core int4 GEMM, on by default for prompts of **512 tokens or
  more**. End-to-end prefill is **3.9× faster** on a 1.5B int4 at a 3900-token prompt, and the
  overhead-free gap to Ollama at depth narrows from 12.1× to **3.2×** (1.5B) and 14.5× to **1.9×**
  (0.5B). Shorter prompts keep the exact path, because that is where a fidelity gate against an
  f32 reference says the fast kernels do not earn their place; at depth the same gate finds them
  **closer to that reference than the path they replaced**. `GOINFER_CUDA_FAST_PREFILL=0` restores
  the previous behaviour in full. Details:
  [docs/measurements/prefill-l2l3-phase3-2026-09-05.md](docs/measurements/prefill-l2l3-phase3-2026-09-05.md).
- **Serving** — OpenAI-compatible and Anthropic Messages endpoints, multi-model, vision,
  embeddings: [docs/server.md](docs/server.md).

## Docs

**New to how any of this works?** [**An inference primer for Go engineers**](https://townsendmerino.github.io/goinfer/)
— eleven chapters on how a language model actually runs, written for someone who knows Go and does
not know machine learning. Each chapter ends in a measured number from this repo. Source in
[docs/book/](docs/book/); chapter 11, on how measurements in this tree have gone wrong, is the one
to read if you only read one.

| page | what's in it |
|---|---|
| [docs/README.md](docs/README.md) | **the map of the docs** — what each kind of page is, and which ones are current claims |
| [docs/book/](docs/book/) · [read online](https://townsendmerino.github.io/goinfer/) | the inference primer — concepts from zero, tied to measured numbers |
| [docs/how-inference-works.md](docs/how-inference-works.md) | the same ground in ten minutes, anchored to specific source lines |
| [docs/server.md](docs/server.md) | the HTTP surface: OpenAI, Anthropic, multi-model, vision, embeddings, admin |
| [docs/benchmarks.md](docs/benchmarks.md) | every measured number, each with machine, checkpoint, quant and date |
| [docs/capability-matrix.md](docs/capability-matrix.md) | generated per-architecture support map |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | modules, packages, and how the pieces fit |
| [docs/giw-bundles.md](docs/giw-bundles.md) | prequantized `.giw` bundles and `cmd/prequant` |
| [docs/positioning.md](docs/positioning.md) | what goinfer is for, and what it is not |
| [docs/api-tiers.md](docs/api-tiers.md) | which surfaces v1.0 will semver-bind |

Demos: `demo/chat` (single-binary local chat), `demo/agent` (fully-local stdlib RAG coding
agent), `demo/gemma` (minimal CLI: tokenizer → decoder → streamed tokens).

Built on [`aikit`](https://github.com/townsendmerino/aikit)'s embedding and tensor primitives.

## Status

Pre-1.0; the forward-pass / quantization contract is parity-gated and stable, the
loader and architecture-descriptor surface is still moving as new model families
land. See `CHANGELOG.md`.

**Which surfaces v1.0 will semver-bind is already decided** — see
[`docs/api-tiers.md`](docs/api-tiers.md) (signed off 2026-08-18). The Hard tier is
what the demos and `serve` use: load a model, tokenize, render a chat prompt,
generate, optionally constrain. The backend/residency seam, family descriptors,
drafters, multimodal and serialization plumbing are named Experimental and stay
outside the promise. The split takes effect at the v1.0 tag, not before.

## License

MIT — see [LICENSE](LICENSE).
