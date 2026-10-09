# goinfer

[![Go Reference](https://pkg.go.dev/badge/github.com/townsendmerino/goinfer.svg)](https://pkg.go.dev/github.com/townsendmerino/goinfer)
[![codecov](https://codecov.io/gh/townsendmerino/goinfer/graph/badge.svg)](https://codecov.io/gh/townsendmerino/goinfer)

**Run an open-weight LLM inside your Go program.** Pure Go, no cgo: `go get` it, import it,
cross-compile it like anything else. No Python, no llama.cpp, no C toolchain, no daemon.

- **Output your types guarantee** — constrain generation to a Go struct or a JSON Schema; an
  invalid token is unreachable, not retried.
- **One static binary** — and, if you want, the model baked into it.
- **41 model families**, each behind a HuggingFace logit-parity gate.
- **CPU, CUDA and Metal**, cgo-free; **WebGPU** as an opt-in cgo build.

Also ships as a ready-made server (`goinfer-serve`: OpenAI and Anthropic APIs, web UI) and a
single-shot chat binary (`goinfer-chat`).

**[goinfer.dev](https://goinfer.dev)** — [models](https://goinfer.dev/models/) ·
[download](https://goinfer.dev/download/) · [docs](https://goinfer.dev/docs/) ·
[an inference primer for Go engineers](https://goinfer.dev/book/)

[![goinfer chat — an entire LLM in one file](docs/assets/demo.gif)](docs/assets/demo.gif)

*A 1.5B model and its runtime in one file: boots in about 0.4 s, under 100 MB of heap, runs
offline. Recorded on an Apple M1 Pro (the `linux-amd64` filename on screen is left over from the
tape's usual render target; a `darwin-arm64` binary is what ran). Details and the x86 figures are in
[`docs/measurements/demo-chat-macbook-2026-08-22.md`](docs/measurements/demo-chat-macbook-2026-08-22.md).*

## A Go struct the model cannot violate

Derive a JSON Schema from a struct, constrain generation to it, and `json.Unmarshal` the result.
The constraint is a logit mask over an incremental byte-level grammar: at every step, tokens that
would break the schema are set to −∞.

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

It works from any JSON Schema too (`constrain.JSONSchema(bytes)`), and from the command line:
`go run ./demo/chat --model … --schema person.schema.json`.

Supported subset: objects (required and optional, `additionalProperties:false`), arrays
(`items`/`minItems`/`maxItems`), `string`/`number`/`integer`/`boolean`/`null`, `enum`/`const`,
and arbitrary nesting. A property-based test asserts that every constrained generation validates
against its schema.

**Tool calls get the same treatment.** On the Qwen families (Qwen2 through Qwen3.8),
Nemotron-3-Nano, Mellum2 and Granite 4.2, a tool call cannot be malformed or name a tool you did
not send, with any number of tools, and a turn that answers in prose runs as fast as it would
unconstrained: [docs/tool-call-coverage.md](docs/tool-call-coverage.md).

**How sure was it?** `.CaptureConfidence(...)` on the masker reports, for each enum, boolean and
integer field, the model's probability over what the schema allowed at the deciding position
(server: `"goinfer_confidence": true` beside a `json_schema` `response_format`). Useful for
routing ("ask a person below 0.7"). It is not the probability that the value is right, and it is
not calibrated — see [docs/server.md](docs/server.md) and
[examples/confidence](examples/confidence/main.go).

**A scanned invoice in, a Go struct out.** GLM-OCR (a 0.9B document model) reads the image; the struct is both the
prompt (`constrain.TemplateFromStruct`, the JSON template the model is trained to fill) and the guarantee
(`constrain.GrammarFromStruct`). One line: `goinfer-chat --model ~/models/glm-ocr --image invoice.png --schema
invoice.schema.json`; in Go: [examples/invoice](examples/invoice/main.go); over HTTP, an `image_url` part plus
`response_format` `json_schema` ([docs/server.md](docs/server.md)). Field accuracy on rendered test invoices, not real scans:
[docs/measurements/glm-ocr-o5-2026-10/](docs/measurements/glm-ocr-o5-2026-10/).

## Use it as a library

```bash
# <!-- smoke --> from inside your own module (`go mod init …` first)
go get github.com/townsendmerino/goinfer/decoder@latest github.com/townsendmerino/goinfer/tokenizer@latest
```

Name the packages you import, as above. A bare `go get github.com/townsendmerino/goinfer` records
the requirement but does not fetch enough to build against, and the next build fails with
`missing go.sum entry for module providing package …`. One command like the one above is enough
for every other package in the same module (`chat`, `constrain`, …).

[`examples/embed/main.go`](examples/embed/main.go) is a complete 40-line program.
`decoder.Model.Generate` is a raw completion primitive with no notion of chat turns, so the
example resolves the checkpoint's own template with `chat.Detect` before encoding. Skip that step
and an instruct model will repeat itself.

Requires Go 1.27+. A walkthrough of this example, the struct and schema constraints, and
per-field confidence, with every code block taken from a tested example:
[docs/use-from-go.md](docs/use-from-go.md). Which surfaces v1.0 will semver-bind is already
decided: [docs/api-tiers.md](docs/api-tiers.md).

## Or run it as a binary

Nothing to build and no Go toolchain needed. Binaries for macOS, Linux and Windows (Intel and
ARM) are on the [latest release](https://github.com/townsendmerino/goinfer/releases/latest) and
the [download page](https://goinfer.dev/download/), with sizes and sha256.

| asset | what it is |
| --- | --- |
| `goinfer-serve-<os>-<arch>` | the **server** — OpenAI + Anthropic APIs, web UI, GPU built in |
| `goinfer-chat-<os>-<arch>` | the single-shot runtime; point it at your own GGUF |
| `goinfer-chat-0.5b-<os>-<arch>` | runtime **and** model in one file — no download, no install |
| `goinfer-chat-1.5b-<os>-<arch>` | same, with the 1.5B coder model |

```bash
# macOS arm64; swap the suffix for your platform
curl -fsSL -o goinfer-serve https://github.com/townsendmerino/goinfer/releases/latest/download/goinfer-serve-darwin-arm64
chmod +x goinfer-serve
```

```bash
# model included — nothing else to fetch
./goinfer-chat-1.5b-darwin-arm64

# or bring your own GGUF
./goinfer-chat-darwin-arm64 --model ~/models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf
```

### Getting a model

The runtime fetches GGUFs straight from HuggingFace; there is no extra tool to install.

```bash
# <!-- smoke-model --> see what a repo publishes
./goinfer-chat-darwin-arm64 pull Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF

# <!-- smoke-model --> fetch one quant (case-insensitive; verified against the sha256 HuggingFace declares)
./goinfer-chat-darwin-arm64 pull Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF:q4_k_m

# <!-- smoke-model --> or the models goinfer itself vets and pins
./goinfer-chat-darwin-arm64 pull demo:1.5b
```

`--model` takes the same references and fetches on first use, so one command goes from nothing to
a running endpoint:

```bash
goinfer-serve -model hf:Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF:q4_k_m
goinfer-chat  --model demo:0.5b
```

Interrupted transfers resume. Downloads land in your user cache dir and print the exact `--model`
command to run them; `goinfer-chat cache` lists what is there. A family that ships no GGUF (safetensors only) is fetched whole with
`:safetensors` — `pull HuggingFaceTB/SmolLM3-3B:safetensors`, or `--model hf:…:safetensors` — as one verified set
of config, tokenizer and weight shards; nothing lands until every file has checked out. Access is anonymous only; a gated repo is detected and named before the
transfer starts. A plain path still means a plain path — only the `hf:` and `demo:` prefixes are
special.

To see the checkpoints this project has actually run:

```bash
# <!-- smoke-help --> lists known-good checkpoints; every row traces to a parity-gated family
goinfer-chat models
# <!-- smoke-model -->
goinfer-chat pull qwen2.5-coder-0.5b        # short name, no repo path to look up
```

The same list, with a page per family, is at [goinfer.dev/models](https://goinfer.dev/models/).
Both derive from [`docs/capability-matrix.json`](docs/capability-matrix.json). goinfer hosts no
weights; any other GGUF works through the explicit `owner/repo:quant` form.

### Server and web UI

```bash
goinfer-serve -web -model ~/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf
```

`-web` adds a local UI at `http://127.0.0.1:8080` for chat and for browsing and pulling models.
It is embedded in the binary, uses no external assets, is off by default, and starts fine with no
model at all. The HTTP surface (OpenAI, Anthropic, multi-model, vision, embeddings, admin) is in
[docs/server.md](docs/server.md). Pointing a real agent (Claude Code, opencode) at it:
[docs/integrations/](docs/integrations). A coding agent's first request is about 11,000 tokens, more than the default GPU context
on Metal (4096) or on a CUDA model too big to hold 16384 positions in each of its 4 KV slots (the server then picks less),
so start the server for an agent with `-ctx 16384` as those recipes do; a longer prompt gets a 400 that names `-ctx`. `serve check` and the `tools:` line of
`goinfer-chat models` report which checkpoints hold up under a real agent's tool schema. The one measured to: Qwen2.5-7B-Instruct `q4_k_m` on a CUDA GPU with 8 GB or more. Qwen2.5-Coder-7B-Instruct on the same card did not (opencode made no edits), so run `serve check` before trusting a checkpoint for an agent ([measured table](docs/integrations/opencode.md#picking-a-model-honestly)).

`POST /v1/systemone` answers TypeSafe's decisions wire shape, so clients such as jevx work against
it ([recipe](docs/integrations/typesafe-jevx.md)). By default it scores the option labels with the
served model: one prefill per question, no decode. Answer quality then depends on the model, and
it is well short of a trained head: on Qwen3.5-9B Q4_K_M, calibrated, top-1 0.42 against the
trained JEV-9B's published 0.92 on an out-of-distribution sample
([record](docs/measurements/decisions-d6a-2026-09-28.md)). A trained decision head in the JEV
layout can be loaded with `--model name=DIR,head=DIR`
([docs/server.md](docs/server.md)). TypeSafe's hosted model is not included.

### Building from source

```bash
# <!-- smoke --> installs as `serve` (the directory name); rename it if you want
go install github.com/townsendmerino/goinfer/cmd/serve@latest
```

That is the CPU server. Each GPU backend has its own entrypoint (`-tags metal` on `cmd/serve`
fails the build and says so):

```bash
go install github.com/townsendmerino/goinfer/metal/cmd/serve@latest              # macOS
go install -tags cuda  github.com/townsendmerino/goinfer/cuda/cmd/serve@latest    # Linux + NVIDIA
```

The released `goinfer-serve` and `goinfer-chat` assets include Metal on macOS and CUDA on Linux,
and use them by default: `--backend auto` picks CUDA when a device answers, else Metal on Apple
silicon for an int4 model, else the CPU, and prints one line saying which. `--backend cpu` names
the CPU. `goinfer-serve --version` prints which backends a binary carries.

## Ship a model as one file

The two model-included downloads are this pipeline run for two models. From a source checkout you
can run it for any supported checkpoint and any OS/arch:

```bash
go run ./demo/chat pull bartowski/google_gemma-3-4b-it-GGUF:Q4_K_M -embed darwin/arm64 linux/amd64
# → demo/chat/dist/goinfer-chat-google_gemma-3-4b-it-{darwin-arm64,linux-amd64}
```

Out comes a static, cgo-free binary with the weights inside: no runtime, no download, no install.
Useful for air-gapped machines, workshops, and handing a demo to a colleague. The binary is
model-sized, and the model's licence travels with it — if you redistribute one, that licence is
yours to satisfy.

## Small devices

Because the build is `CGO_ENABLED=0`, a 64-bit ARM Linux board is an ordinary cross-compile and a
copy:

```bash
GOOS=linux GOARCH=arm64 CGO_ENABLED=0 go build -o goinfer-serve ./cmd/serve
scp goinfer-serve pi@raspberrypi.local:
```

A 512 MB board (Pi Zero 2 W) is the low end, with one 270M–0.5B model and a short context; the
fast arm64 kernels need the DotProd extension (Pi 5 and newer). No Pi figure is published yet.
Microcontrollers and TinyGo are out of scope. Details: [docs/small-devices.md](docs/small-devices.md).

## Which quantization

`--quant int4` is the default. Use `--quant int8int8` when accuracy matters more than RAM, and on
Apple Silicon when you want the smaller resident footprint (int4 is faster there but larger,
because the NEON repack keeps a second copy of the weights).

goinfer reads 15 GGUF types and computes in five precisions, and those are separate questions:
a Q2_K file loaded at `--quant int8int8` gives Q2_K quality computed carefully, not int8 quality.
[docs/quantization.md](docs/quantization.md) lists which quants the project stands behind, which
it has measured and refused, and which read paths have no quality evidence yet.

## Bigger than your RAM or your GPU

A 20–35B-class MoE does not fit in 16 GB of RAM or on an 8 GB GPU. Two flags cover it.

**RAM — `-stream-weights`.** Only the experts a token routes to stay resident, so memory is capped
near `-weight-cache`. A 21 GB 35B-A3B on an M1 Pro / 16 GB: without the flag, +7.8 GB of swap in
five seconds; with it, RSS peaked at 8.95 GB with zero swapouts.

```bash
# <!-- smoke-model --> a 20-35B-class MoE, real and resolvable — goinfer-chat models for the full entry
goinfer-chat pull gpt-oss-20b
```

```bash
# <!-- smoke-help --> the flag to reach for whenever the checkpoint file is larger than about half your physical RAM
goinfer-serve -stream-weights -weight-cache 6 -model ~/models/gpt-oss-20b-MXFP4.gguf
```

This applies to `goinfer-serve`; `goinfer-chat` holds all weights resident.

**GPU — `-moe-cache-experts`.** The non-expert core stays in VRAM and experts stream host→VRAM on
demand: 39.3 tok/s at ctx 2048 for `gemma-4-26b-a4b` on an 8 GB RTX 2070 SUPER (2026-09-29, median
of three; [docs/benchmarks.md](docs/benchmarks.md) §B4/§B4.1,
[peer sweep](docs/measurements/peer-sweep-2026-09-29.md) cell c). That figure is bound by PCIe
transfer. A dense model bigger than your card has no partial-GPU path here.

```bash
# <!-- smoke-model --> the checkpoint this project has the most measurements on at this size
goinfer-chat pull gemma-4-26b-a4b
```

```bash
# <!-- smoke-help --> off by default; a model that does not fit then declines to the CPU path and says why
goinfer-serve -backend cuda -moe-cache-experts -model ~/models/gemma-4-26B_q4_0-it.gguf
```

Evidence and caveats: [docs/bigger-than-memory.md](docs/bigger-than-memory.md).

## Performance next to Ollama

| measure | result | record |
| --- | --- | --- |
| Nothing → first answer, M1 Pro / 16 GB | 25 s against Ollama's 33 s, from an 8 MB binary with no daemon | [cold-user run](docs/measurements/cold-user-2026-09-06.md), scenario E |
| Greedy decode, CUDA (RTX 2070 SUPER), 0.5B / 1.5B / 7B, depth 128 to 8,000 | ahead of Ollama in all 12 cells, 1.05–1.47×; the margin narrows with depth on the 7B | [peer sweep 2026-09-29](docs/measurements/peer-sweep-2026-09-29.md) |
| Greedy decode, Apple Metal (M1 Pro), 1.5B / 7B, depth 128 to 3,900 | 1.03–1.06× Ollama on the 1.5B, 1.06–1.19× on the 7B (a same-session A/B, not the pre-registered sweep) | [R18b record](docs/measurements/metal-decode-gemv-r18b-2026-09-26.md) |

Where it is behind: Phi-3 mini decode on CUDA (0.72–0.90×), and CPU decode on a Mac against
llama.cpp (0.84× on the 0.5B). Long-prompt prefill on Metal, 1.96× behind Ollama on 2026-09-25, measured
level at a 3,900-token prompt on 2026-09-30 (0.98×, 1.5B,
[peer sweep](docs/measurements/peer-sweep-2026-09-29.md) cell h). The Metal decode figures are for the resident GPU path. `--embed-int4` defaults off on
Metal, because that path does not take an int4 embedding table yet
([docs/quantization.md](docs/quantization.md)).

Steady-state results are mixed and machine-dependent. Every figure names its machine, checkpoint,
quant and date in [docs/benchmarks.md](docs/benchmarks.md). To measure on your own hardware:
`scripts/bench_peer.py` — same weights both sides, decode-only, interleaved, server restarted per
cell.

## What it has been run on

goinfer is built, measured and checked on three machines: a MacBook Pro (M1 Pro, 16 GB, Metal), a desktop with an RTX 2070 SUPER (8 GB, CUDA), and the Ryzen 7 3700X in that same desktop (CPU only).
Every speed in this README came from one of them and does not carry over to other hardware. Other machines, other GPU generations and other operating systems have mostly not been run by us:
[the generated hardware matrix](docs/hardware-matrix.md#verified-on-what-has-actually-run) lists which hardware-selected paths have executed, on what, and how (real hardware, a CI runner, a build tag that forces a
narrower path, or emulation), and which have never executed at all.

What stands between you and a wrong answer on a machine we have not seen is a check goinfer runs at start: it runs the compute kernels it is about to use against a reference on a small fixed input, and if one disagrees
it steps down to a slower path, or declines that backend, rather than giving wrong numbers. Today that covers the CPU kernels, CUDA, WebGPU on a real GPU (not on a software renderer), and Metal; each GPU backend's margins were
measured on one device so far ([docs/server.md](docs/server.md)). `goinfer-serve check --hardware` prints what it found on your machine and sends nothing anywhere. If something is wrong, paste it into
[a bug report](https://github.com/townsendmerino/goinfer/issues/new?template=bug.yml); that is the most useful thing you can send.

## What it is, and isn't

goinfer targets **single-user local inference**: one process, one machine,
deployed by copying a file.

It is **not a serving engine**: no continuous batching and no paged attention. A server can run a
few generations of one model at once (`--max-concurrent`, default 4, each on its own KV), and on
Metal and CUDA a dense resident model batches their decode steps, but requests beyond that wait in
a bounded queue ([docs/server.md](docs/server.md)). To saturate a datacentre GPU with concurrent
requests, use vLLM. It is also not a provider-orchestration library; it runs the weights itself,
in-process. Longer form: [docs/positioning.md](docs/positioning.md).

## What it runs

- **41 model families** — Gemma 1/2/3/4 (and CodeGemma), Qwen 2.5/3, Llama, Mistral, Mixtral, Phi-3, DeepSeek/MLA,
  GLM, Kimi, Granite, Nemotron, Mellum and more; one page each at
  [goinfer.dev/models](https://goinfer.dev/models/), generated from the `decoder` registry
  ([capability-matrix.md](docs/capability-matrix.md)).
- **All four sequence-mixing families** — softmax·GQA, gated-linear (DeltaNet), state-space
  (Mamba-2), latent-KV (MLA) — plus dense and sparse-MoE.
- **Loaders** — GGUF, safetensors, GPTQ, AWQ, and prequantized [`.giw` bundles](docs/giw-bundles.md).
- **Quantization** — f32, int8, int8int8, int4 (W4A8) and int4mix, with a
  [HuggingFace logit-parity gate per family](docs/what-parity-gated-means.md). A parity run proves
  what its fixtures cover, and a missing fixture skips rather than fails, so quote a run's counts
  (`28 ran / 20 skipped / 0 failed`): see `docs/parity-coverage-policy.md`.
- **GPU** — CUDA and Metal are cgo-free: at runtime they open the vendor's own driver API
  (`libcuda.so.1`, Metal.framework) and run kernels shipped in the binary, so there is no CUDA
  toolkit to build against, and without the driver the load declines to the CPU path. WebGPU
  (opt-in, `-tags gpu`) is the one cgo build: it links the prebuilt wgpu-native library, which
  drives the system's Vulkan, Metal or DX12 driver, so it needs a C toolchain and is not in the
  release binaries. Nothing that performs inference is loaded from outside the binary: no
  llama.cpp, no libllama, no Python runtime. The library's `go.mod` has two direct requirements,
  `github.com/townsendmerino/aikit` and `golang.org/x/text`; each GPU backend is its own module.
  Dense and MoE models; anything unsupported declines at load and falls back to CPU. See [docs/cuda-backend.md](docs/cuda-backend.md)
  and [docs/gpu-residency-coverage.md](docs/gpu-residency-coverage.md). On CUDA, prompts of 512
  tokens or more use a fused FlashAttention-style kernel and a tensor-core int4 GEMM (3.9×
  end-to-end prefill at a 3900-token prompt on a 1.5B int4;
  [record](docs/measurements/prefill-l2l3-phase3-2026-09-05.md);
  `GOINFER_CUDA_FAST_PREFILL=0` restores the previous path).
- **Serving** — OpenAI-compatible and Anthropic Messages endpoints, multi-model, vision,
  embeddings: [docs/server.md](docs/server.md).

## Docs

[goinfer.dev](https://goinfer.dev) is the readable front door and is rebuilt at each release; the
repo is the source of truth in between. The [primer](https://goinfer.dev/book/) is eleven chapters
on how a language model runs, written for someone who knows Go and not machine learning, each
ending in a measured number from this repo. Chapter 11, on how measurements here have gone wrong,
is the one to read if you read only one. Source: [docs/book/](docs/book).

| page | what's in it |
| --- | --- |
| [docs/README.md](docs/README.md) | **the map of the docs** — what each kind of page is, and which are current claims |
| [docs/how-inference-works.md](docs/how-inference-works.md) | the book's ground in ten minutes, anchored to source lines |
| [docs/server.md](docs/server.md) | the HTTP surface |
| [docs/benchmarks.md](docs/benchmarks.md) | every measured number, with machine, checkpoint, quant and date |
| [docs/capability-matrix.md](docs/capability-matrix.md) | generated per-architecture support map |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | modules, packages, and how the pieces fit |
| [docs/bigger-than-memory.md](docs/bigger-than-memory.md) · [docs/small-devices.md](docs/small-devices.md) | running past your RAM or GPU; running on a Raspberry Pi |
| [docs/giw-bundles.md](docs/giw-bundles.md) | prequantized `.giw` bundles and `cmd/prequant` |
| [docs/positioning.md](docs/positioning.md) | what goinfer is for, and what it is not |
| [docs/api-tiers.md](docs/api-tiers.md) | which surfaces v1.0 will semver-bind |

Demos: `demo/chat` (single-binary local chat), `demo/agent` (fully-local stdlib RAG coding agent).

Built on [`aikit`](https://github.com/townsendmerino/aikit)'s embedding and tensor primitives.

## Status

Pre-1.0. The forward-pass and quantization contract is parity-gated and stable; the loader and
architecture-descriptor surface is still moving as new model families land. See `CHANGELOG.md`.

The Hard tier that v1.0 will bind is what the demos and `serve` use: load a model, tokenize,
render a chat prompt, generate, optionally constrain. The backend/residency seam, family
descriptors, drafters, multimodal and serialization plumbing are Experimental and stay outside
the promise. The split takes effect at the v1.0 tag ([docs/api-tiers.md](docs/api-tiers.md),
signed off 2026-08-18).

## License

MIT — see [LICENSE](LICENSE).
