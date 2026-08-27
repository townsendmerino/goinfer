# goinfer

**Run open-weight LLMs in pure Go — one cgo-free static binary, portable by default and
native-GPU-fast when you want it.** ~20 model architectures, HuggingFace-parity-gated, with
schema-constrained structured output. No Python, no llama.cpp, no CUDA toolkit.

![goinfer chat — an entire LLM in one file](docs/assets/demo.gif)

*An entire LLM in one file — instant boot (~0.4 s), <100 MB heap, runs offline. Writes correct
generic Go and **cannot** emit invalid JSON. No cgo, no Python, no model download.*

<sub>Recorded on an Apple M1 Pro (the visible `linux-amd64` filename is a leftover from the tape's
usual render target — a `darwin-arm64` binary is what actually ran); a desktop x86 CPU measures
roughly half the on-screen tok/s on the identical harness — see
`docs/measurements/demo-chat-macbook-2026-08-22.md`.</sub>

## Download and run

Two kinds of binary on the [latest release](https://github.com/townsendmerino/goinfer/releases/latest)
(macOS / Linux / Windows, Intel + ARM):

| asset | size | what it is |
|---|---|---|
| `goinfer-chat-<os>-<arch>` | ~5 MB | the runtime; point it at your own GGUF |
| `goinfer-chat-0.5b-<os>-<arch>` | ~615 MB | runtime **and** model in one file — no download, no install |

```bash
# model included — nothing else to fetch
./goinfer-chat-0.5b-darwin-arm64

# or bring your own GGUF
./goinfer-chat-darwin-arm64 --model ~/models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf
```

From source, against any supported checkpoint (the chat template is applied automatically):

```bash
go run ./demo/chat --model ~/models/gemma-4-E2B_q4_0-it.gguf
```

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
_ = json.Unmarshal(out, &p)                          // always succeeds
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

- **~20 architectures** — Gemma 3/4, Qwen 2.5/3, Llama, Mistral, Mixtral, Phi-3, DeepSeek/MLA,
  GLM, Kimi, Granite, Nemotron, Mellum and more. Full generated map:
  [docs/capability-matrix.md](docs/capability-matrix.md).
- **All four sequence-mixing families** — softmax·GQA, gated-linear (DeltaNet), state-space
  (Mamba-2), latent-KV (MLA) — plus dense and sparse-MoE.
- **Loaders** — GGUF, safetensors, GPTQ, AWQ, and prequantized
  [`.giw` bundles](docs/giw-bundles.md).
- **Quantization** — f32, int8, int8int8, int4 (W4A8), with a HuggingFace logit-parity gate per
  family.
- **GPU** — WebGPU everywhere, plus cgo-free CUDA and Metal for dense and MoE models; anything
  unsupported declines at load and falls back to CPU rather than dropping a feature silently.
  See [docs/cuda-backend.md](docs/cuda-backend.md) and
  [docs/gpu-residency-coverage.md](docs/gpu-residency-coverage.md).
- **Serving** — OpenAI-compatible and Anthropic Messages endpoints, multi-model, vision,
  embeddings: [docs/server.md](docs/server.md).

## Docs

| page | what's in it |
|---|---|
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

## Install

```bash
go get github.com/townsendmerino/goinfer
```

## License

MIT — see [LICENSE](LICENSE).
