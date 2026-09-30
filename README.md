# goinfer

**Run open-weight LLMs in pure Go — one cgo-free static binary, portable by default and
native-GPU-fast when you want it.** 39 model families, HuggingFace-parity-gated, with
schema-constrained structured output. No Python, no llama.cpp, no CUDA toolkit.

**[goinfer.dev](https://goinfer.dev)** — the [models](https://goinfer.dev/models/) it runs (one page per family),
[downloads](https://goinfer.dev/download/), [docs](https://goinfer.dev/docs/), and the
[inference primer](https://goinfer.dev/book/). This page is install, first run, and what it is and is not.

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
> go install github.com/townsendmerino/goinfer/metal/cmd/serve@latest              # macOS
> go install -tags cuda  github.com/townsendmerino/goinfer/cuda/cmd/serve@latest    # Linux + NVIDIA
> ```
>
> The downloaded `goinfer-serve` assets already have this built in — Metal on macOS, CUDA on
> Linux. `goinfer-serve --version` prints which backends a given binary carries.

**Using it as a library?** `go get github.com/townsendmerino/goinfer` — **the bare module, no
package path** — fetches only enough to record the requirement, not enough to build against: a
program that then imports `decoder` (or any other package here) fails with `missing go.sum entry
for module providing package …`. Naming the packages you actually import is what fixes it:

```bash
# <!-- smoke --> from inside your own module (`go mod init …` first)
go get github.com/townsendmerino/goinfer/decoder@latest github.com/townsendmerino/goinfer/tokenizer@latest
```

That command resolves enough of the module's own dependency graph that further same-module
imports (`chat`, `constrain`, …) build with it too — you do not need a separate `go get` per
package, only per module boundary crossed (verified: a program importing `decoder` + `tokenizer` +
`chat` off exactly this command built clean, with `chat` never named).

See [`examples/embed/main.go`](examples/embed/main.go) for a complete 40-line program.
`decoder.Model.Generate` is a raw completion primitive — it has no notion of chat turns —
so the example resolves the checkpoint's own template with `chat.Detect` before encoding;
skip that step and an instruct model degenerates into repeating itself.

## Download and run

Binaries are on the [latest release](https://github.com/townsendmerino/goinfer/releases/latest) and the
[download page](https://goinfer.dev/download/) (macOS / Linux / Windows, Intel + ARM; sizes and sha256 are listed there):

| asset | what it is |
|---|---|
| `goinfer-serve-<os>-<arch>` | the **server** — OpenAI + Anthropic APIs, web UI, GPU built in |
| `goinfer-chat-<os>-<arch>` | the single-shot runtime; point it at your own GGUF |
| `goinfer-chat-0.5b-<os>-<arch>` | runtime **and** model in one file — no download, no install |
| `goinfer-chat-1.5b-<os>-<arch>` | same, with the 1.5B coder model |

```bash
# model included — nothing else to fetch
./goinfer-chat-1.5b-darwin-arm64

# or bring your own GGUF
./goinfer-chat-darwin-arm64 --model ~/models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf
```

**Against Ollama, honestly.** From nothing to an answer on an M1 Pro / 16 GB: **25 s vs 33 s**, from an 8 MB binary
with no daemon ([cold-user run](docs/measurements/cold-user-2026-09-06.md), scenario E; a cold start only). Steady-state
decode is mixed and machine-dependent: on CUDA, matched quant, interleaved, goinfer **192.8 tok/s** vs Ollama
**183.6 tok/s** (~5% ahead, short context —
[record](docs/measurements/cold-user-2026-09-06-nobara-pc.md)), and level or ahead at 10 of 12 cells from 128 to 8,000
tokens with none behind and two void, after flash-decode ([peer sweep](docs/measurements/peer-claim-2026-09-25.md)); on Apple Metal
(v0.17.1, Qwen2.5-Coder-1.5B q4_K_M) **~13–18% behind**
([record](docs/measurements/cold-user-2026-09-07-macbook-arm64.md)). Every figure names its machine, checkpoint,
quant and date in [`docs/benchmarks.md`](docs/benchmarks.md); measure it yourself with `scripts/bench_peer.py` — same
weights both sides, decode-only, interleaved, server restarted per cell.

Don't have a model yet? The runtime can fetch a GGUF straight from HuggingFace — no extra tool
to install, and no `huggingface-cli`:

```bash
# <!-- smoke-model --> see what a repo publishes
./goinfer-chat-darwin-arm64 pull Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF

# <!-- smoke-model --> fetch one quant (case-insensitive; verified against the sha256 HuggingFace declares)
./goinfer-chat-darwin-arm64 pull Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF:q4_k_m

# <!-- smoke-model --> or the models goinfer itself vets and pins
./goinfer-chat-darwin-arm64 pull demo:1.5b
```

Interrupted transfers resume where they stopped; `goinfer-serve pull …` is the same command. `--model` takes the same
reference and fetches it on first use, so one command goes from nothing to a running endpoint:

```bash
goinfer-serve -model hf:Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF:q4_k_m
goinfer-chat  --model demo:0.5b
```

Downloads land in your user cache dir and print the exact `--model` command to run them. Anonymous only: a gated repo
is detected before the transfer starts and named, rather than failing after a multi-gigabyte download. A plain path
still means exactly what it always did; only the `hf:`/`demo:` prefixes are new.

Prefer a browser? `serve -web` adds a local UI at `http://127.0.0.1:8080` — chat, browse a HuggingFace repo, and pull a
checkpoint with live progress. One embedded HTML file, no external assets, off by default, and `-web` alone is enough
to start with no model at all:

```bash
goinfer-serve -web -model ~/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf
```

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

## Which model to download

[**goinfer.dev/models**](https://goinfer.dev/models/) has a page per family and the checkpoints this project has
actually run — size, quantization, what each is good for, what it costs to run, and the sha256 the download is
verified against. The same list is in your terminal:

```bash
# <!-- smoke-help --> lists known-good checkpoints; every row traces to a parity-gated family
goinfer-chat models
# <!-- smoke-model -->
goinfer-chat pull qwen2.5-coder-0.5b        # short name, no repo path to look up
```

Both derive from [`docs/capability-matrix.json`](docs/capability-matrix.json), so nothing is listed that the parity
gates do not back. **goinfer hosts no weights** — downloads come from Hugging Face, and any other GGUF works too via
the explicit `owner/repo:quant` form.

## Which quantization to use

`--quant int4` is the default and the one to reach for; `--quant int8int8` when accuracy matters
more than RAM, and on Apple Silicon when you want the *smaller* resident footprint (int4 is faster
there but larger — the NEON repack keeps a second copy of the weights).

goinfer **reads** 15 GGUF types and **computes** in five precisions, which are different
questions: loading a Q2_K file at `--quant int8int8` gives you Q2_K quality computed carefully,
not int8 quality. [`docs/quantization.md`](docs/quantization.md) states which quants the project
stands behind, which it has measured and refused, and — importantly — which read paths have no
quality evidence at all.

## Bigger than your RAM or your GPU

A 20-35B-class MoE does not fit in 16 GB of RAM or on an 8 GB GPU, and loading it anyway drives the machine into swap
(or the CUDA allocator into an OOM) before anything says so. Two flags cover it, and both examples use a checkpoint this
project actually validates at that size:

```bash
# <!-- smoke-model --> a 20-35B-class MoE, real and resolvable — goinfer-chat models for the full entry
goinfer-chat pull gpt-oss-20b
```

```bash
# <!-- smoke-help --> the flag to reach for whenever the checkpoint file is larger than about half your physical RAM
goinfer-serve -stream-weights -weight-cache 6 -model ~/models/gpt-oss-20b-MXFP4.gguf
```

**RAM:** `-stream-weights` caps resident memory near `-weight-cache`, because only the experts a token routes to are
resident (a 21 GB 35B-A3B on an M1 Pro / 16 GB: without it **+7.8 GB of swap in five seconds**, with it RSS peaked at
**8.95 GB** with zero swapouts). This is `goinfer-serve`'s job; the single-shot `goinfer-chat` holds all weights resident.
**Caution:** five families (gemma4, laguna, granite, nemotron, llama4) still build resident before their one-time
transcode — watch the first `-stream-weights` run of one, don't walk away from it.

**GPU:** on cuda/metal, GPU means fully resident. A MoE bigger than your card has `-moe-cache-experts`: the non-expert
core stays resident and the experts a token routes to stream host→VRAM on demand — **39.3 tok/s** at ctx 2048 (2026-09-29, median of three runs) for
`gemma-4-26b-a4b` (26B-A4B, 128 experts top-8) on an 8 GB RTX 2070 SUPER with the DMA overlap
([`docs/benchmarks.md`](docs/benchmarks.md) §B4/§B4.1, [peer sweep](docs/measurements/peer-sweep-2026-09-29.md) cell c),
capacity-bound (PCIe host→VRAM streaming), not a kernel or MoE deficiency. A dense model bigger than your card has no
partial-GPU story here.

```bash
# <!-- smoke-model --> the checkpoint this project has the most measurements on at this size
goinfer-chat pull gemma-4-26b-a4b
```

```bash
# <!-- smoke-help --> off by default; a model that does not fit then declines to the CPU path and says why
goinfer-serve -backend cuda -moe-cache-experts -model ~/models/gemma-4-26B_q4_0-it.gguf
```

The evidence, the swap-tripwire history and the `gpt-oss-20b` caveats:
[`docs/bigger-than-memory.md`](docs/bigger-than-memory.md).

## Small devices

Because the whole build is `CGO_ENABLED=0`, a 64-bit ARM Linux board is an ordinary cross-compile and a copy — nothing
on the board needs installing:

```bash
GOOS=linux GOARCH=arm64 CGO_ENABLED=0 go build -o goinfer-serve ./cmd/serve
scp goinfer-serve pi@raspberrypi.local:
```

A 512 MB board (Pi Zero 2 W) is the low end, running one 270M–0.5B model with a short context; the fast arm64 kernels
need the DotProd extension (Pi 5 and newer). **No Pi figure is published yet** — treat "a small model decodes at a usable
rate on a Pi 5" as an expectation, not a claim. Microcontrollers and TinyGo are out of scope. Details:
[`docs/small-devices.md`](docs/small-devices.md).

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

**How sure was it?** `.CaptureConfidence(...)` on the masker gives each enum, boolean and integer field the model's
probability over what the schema allowed at the position that decided it (server: `"goinfer_confidence": true`
beside a `json_schema` `response_format`). Good for routing ("ask a person below 0.7"); **not** the probability the
value is right, and not calibrated ([what it is and is not](docs/server.md);
[example](examples/confidence/main.go)).

**Decisions.** `POST /v1/systemone` answers TypeSafe's decisions wire shape by label scoring on the served model — one
prefill per question, no decode — so clients written for TypeSafe, such as jevx and its SDKs, work against it
([recipe](docs/integrations/typesafe-jevx.md)). Not included: TypeSafe's hosted model or a trained decision head.
Answer quality depends on the model; goinfer's own measurement is pending ([details](docs/server.md)).


## What it is, and isn't

goinfer targets **single-user local inference**: one process, one machine, batch-1 decode,
deployed by copying a file. It builds with no toolchain of any kind — no CUDA toolkit, no C++
compiler, no CMake, no Python — and cross-compiles like any other Go program.

It is **not a serving engine**: no continuous batching, no paged attention, one generation at a
time behind a bounded queue. If you need to saturate a datacentre GPU with concurrent requests,
vLLM is built for that and goinfer is not. It is also not a provider-orchestration library — it
runs the weights itself, in-process. Longer form: [docs/positioning.md](docs/positioning.md).

## What it runs

- **39 model families** — Gemma 1/2/3/4 (and CodeGemma), Qwen 2.5/3, Llama, Mistral, Mixtral, Phi-3, DeepSeek/MLA, GLM, Kimi, Granite,
  Nemotron, Mellum and more; one page each at [goinfer.dev/models](https://goinfer.dev/models/), generated from the
  `decoder` registry ([capability-matrix.md](docs/capability-matrix.md)).
- **All four sequence-mixing families** — softmax·GQA, gated-linear (DeltaNet), state-space (Mamba-2), latent-KV
  (MLA) — plus dense and sparse-MoE.
- **Loaders** — GGUF, safetensors, GPTQ, AWQ, and prequantized [`.giw` bundles](docs/giw-bundles.md).
- **Quantization** — f32, int8, int8int8, int4 (W4A8), with a
  [HuggingFace logit-parity gate per family](docs/what-parity-gated-means.md). **What a given parity run proves is
  scoped to the fixtures that machine has**, and a missing fixture skips silently rather than failing — quote a run's
  counts (`28 ran / 20 skipped / 0 failed`), not the word "green": `docs/parity-coverage-policy.md` §"Scoped: a
  goldens green names the quantizations that actually RAN".
- **GPU** — WebGPU everywhere, plus cgo-free CUDA and Metal for dense and MoE models; anything unsupported declines at
  load and falls back to CPU rather than dropping a feature silently. See [docs/cuda-backend.md](docs/cuda-backend.md)
  and [docs/gpu-residency-coverage.md](docs/gpu-residency-coverage.md). On CUDA, prompts of 512 tokens or more use a
  fused FlashAttention-style kernel and a tensor-core int4 GEMM (**3.9×** end-to-end prefill at a 3900-token prompt on a
  1.5B int4; `GOINFER_CUDA_FAST_PREFILL=0` restores the previous path):
  [docs/measurements/prefill-l2l3-phase3-2026-09-05.md](docs/measurements/prefill-l2l3-phase3-2026-09-05.md).
- **Serving** — OpenAI-compatible and Anthropic Messages endpoints, multi-model, vision,
  embeddings: [docs/server.md](docs/server.md). Pointing a real agent (Claude Code, opencode) at
  it: [docs/integrations/](docs/integrations/) — `serve check`'s harness-scale tools row and
  `goinfer-chat models`' `tools:` line say which checkpoints actually hold up under a real
  agent's tool schema, measured, before you find out the way a cold-user run did. On the Qwen
  families (Qwen2 through Qwen3.8), Nemotron-3-Nano, Mellum2 and Granite 4.2, a tool call **cannot be malformed
  or name a tool you did not send**, with any number of tools — and a turn that answers in prose
  instead runs exactly as fast as without the constraint:
  [docs/tool-call-coverage.md](docs/tool-call-coverage.md).

## Docs

The site is the readable front door: [**goinfer.dev**](https://goinfer.dev) — [models](https://goinfer.dev/models/),
[download](https://goinfer.dev/download/), [docs](https://goinfer.dev/docs/), and
[**an inference primer for Go engineers**](https://goinfer.dev/book/): eleven chapters on how a language model
actually runs, for someone who knows Go and not machine learning, each ending in a measured number from this repo.
Chapter 11, on how measurements in this tree have gone wrong, is the one to read if you only read one. Source in
[docs/book/](docs/book/). The site is rebuilt when a release is cut, so the repo is the source of truth between
releases.

| page | what's in it |
|---|---|
| [docs/README.md](docs/README.md) | **the map of the docs** — what each kind of page is, and which ones are current claims |
| [docs/how-inference-works.md](docs/how-inference-works.md) | the same ground as the book in ten minutes, anchored to specific source lines |
| [docs/server.md](docs/server.md) | the HTTP surface: OpenAI, Anthropic, multi-model, vision, embeddings, admin |
| [docs/benchmarks.md](docs/benchmarks.md) | every measured number, each with machine, checkpoint, quant and date |
| [docs/capability-matrix.md](docs/capability-matrix.md) | generated per-architecture support map |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | modules, packages, and how the pieces fit |
| [docs/bigger-than-memory.md](docs/bigger-than-memory.md) · [docs/small-devices.md](docs/small-devices.md) | running past your RAM or GPU; running on a Raspberry Pi |
| [docs/giw-bundles.md](docs/giw-bundles.md) | prequantized `.giw` bundles and `cmd/prequant` |
| [docs/positioning.md](docs/positioning.md) | what goinfer is for, and what it is not |
| [docs/api-tiers.md](docs/api-tiers.md) | which surfaces v1.0 will semver-bind |

Demos: `demo/chat` (single-binary local chat), `demo/agent` (fully-local stdlib RAG coding
agent).

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
