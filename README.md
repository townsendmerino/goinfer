# goinfer

**Run open-weight LLMs in pure Go — one cgo-free static binary, portable by default and
native-GPU-fast when you want it.** 35 model families, HuggingFace-parity-gated, with
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

**From nothing to an answer in 25 seconds** — download, pull a model, get a reply. Measured on an
Apple M1 Pro / 16 GB against Ollama 0.32.5 doing the same thing on the same box: **25 s vs 33 s**,
from an 8 MB binary with no daemon to install and nothing left running afterwards
([`docs/measurements/cold-user-2026-09-06.md`](docs/measurements/cold-user-2026-09-06.md),
scenario E). That leg is a cold start only; steady-state decode on that machine is a separate
measurement. On **v0.17.1** — a Mac asset confirmed carrying Metal (`backends: cpu metal`,
`decode path: metal-resident`, both engines confirmed offloaded to GPU) — Qwen2.5-Coder-1.5B,
q4_K_M, interleaved: **goinfer ~13–18% behind Ollama**
([`docs/measurements/cold-user-2026-09-07-macbook-arm64.md`](docs/measurements/cold-user-2026-09-07-macbook-arm64.md),
scenario E; `docs/benchmarks.md` §B3 has the numbers and this run's caveats — 2 interleaved runs,
not the section's own best-of-3 protocol). This replaces an earlier v0.16.0-asset reading here,
which R2 found was measuring a Mac binary with no Metal backend linked in, not the engine.

On a Linux box with a GPU, cold start is network-bound (**56.5 s**, dominated by a **1.71 GiB**
binary download at **~31.5 MB/s** — not a fixed number, a function of your connection) and
steady-state CUDA decode, matched quant, interleaved, client-side tok/s from first token to
last: goinfer **192.8 tok/s** vs Ollama **183.6 tok/s**, ~5% ahead
([`docs/measurements/cold-user-2026-09-06-nobara-pc.md`](docs/measurements/cold-user-2026-09-06-nobara-pc.md),
scenario E) — consistent with `docs/benchmarks.md` §B8's own formally-provenanced anchor table,
whose shallow-KV-depth cells (this was a short completion, effectively depth ≈128) show goinfer
ahead of Ollama on the same quant class; §B8's deeper cells show Ollama pulling ahead as context
grows, which this short run does not contradict. Measure it yourself rather than trust either
number: `scripts/bench_peer.py` is the committed harness both of the above used underneath —
same weights both sides, decode-only, interleaved, server-restarted per cell, provenance
stamped into the output file.

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
# <!-- smoke-model --> see what a repo publishes
./goinfer-chat-darwin-arm64 pull Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF

# <!-- smoke-model --> fetch one quant (case-insensitive; verified against the sha256 HuggingFace declares)
./goinfer-chat-darwin-arm64 pull Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF:q4_k_m

# <!-- smoke-model --> or the models goinfer itself vets and pins
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

## Which model to download

`goinfer-chat models` lists the checkpoints this project has actually run — size, quantization,
what each is good for, and what it costs to run:

```bash
# <!-- smoke-help --> lists known-good checkpoints; every row traces to a parity-gated family
goinfer-chat models
# <!-- smoke-model -->
goinfer-chat pull qwen2.5-coder-0.5b        # short name, no repo path to look up
```

Each entry derives from a row in [`docs/capability-matrix.json`](docs/capability-matrix.json), so
nothing is listed that the parity gates do not back, and each carries a sha256 the download is
verified against. **goinfer hosts no weights** — downloads come from Hugging Face, and any other
GGUF works too via the explicit `owner/repo:quant` form.

## Which quantization to use

`--quant int4` is the default and the one to reach for; `--quant int8int8` when accuracy matters
more than RAM, and on Apple Silicon when you want the *smaller* resident footprint (int4 is faster
there but larger — the NEON repack keeps a second copy of the weights).

goinfer **reads** 15 GGUF types and **computes** in five precisions, which are different
questions: loading a Q2_K file at `--quant int8int8` gives you Q2_K quality computed carefully,
not int8 quality. [`docs/quantization.md`](docs/quantization.md) states which quants the project
stands behind, which it has measured and refused, and — importantly — which read paths have no
quality evidence at all.

## Running a model bigger than your RAM — or your GPU

A 20-35B-class MoE does not fit in 16 GB of RAM, or on an 8 GB GPU, and loading it anyway will
drive your machine into swap (or your CUDA allocator into an OOM) before anything says so.
The RAM-overflow and GPU-overflow examples below both use the checkpoint this project actually
validates at that size, rather than a size class with nothing behind it to download:

```bash
# <!-- smoke-model --> a 20-35B-class MoE, real and resolvable — goinfer-chat models for the full entry
goinfer-chat pull gpt-oss-20b
```

```bash
# <!-- smoke-help --> the flag to reach for whenever the checkpoint file is larger than about half your physical RAM
goinfer-serve -stream-weights -weight-cache 6GiB -model ~/models/gpt-oss-20b-MXFP4.gguf
```

Resident memory is then capped near `-weight-cache` rather than the model size, because only the
experts a token actually routes to are resident. Measured on an M1 Pro / 16 GB with a 21 GB
35B-A3B (a different checkpoint at the same size class — the mechanism is the same either way):
without the flag, **+7.8 GB of swap in five seconds**; with it, RSS peaked at **8.95 GB** and fell
back to 2.7 GB, with zero swapouts
([`docs/measurements/cold-user-2026-09-06.md`](docs/measurements/cold-user-2026-09-06.md),
scenario D).

**This is `goinfer-serve`'s job, not `goinfer-chat`'s.** The single-shot chat runtime holds all
weights resident by design; it has no `-stream-weights`. If your model is bigger than your RAM,
reach for the server.

**On cuda/metal, GPU means fully resident, full stop.** Neither backend has a partial/"staged"
GPU path (R9, [`docs/measurements/cold-user-2026-09-06-nobara-pc.md`](docs/measurements/cold-user-2026-09-06-nobara-pc.md)):
a model or architecture that does not build the resident runner declines straight to CPU, at
whatever quant you asked for. A dense model bigger than your card has no partial-GPU story here —
only `-stream-weights` (above, RAM-side) or the CPU. **A MoE does**, and that is
`-moe-cache-experts`: the non-expert core stays resident while a slot cache of the experts a
token actually routes to streams host→VRAM per token on demand, so the whole model never needs
to fit VRAM. Measured on this project's own most-benchmarked checkpoint at this size,
`gemma-4-26b-a4b` (26B-A4B, 128 experts top-8; [`docs/benchmarks.md`](docs/benchmarks.md) §B4/§B4.1),
on an RTX 2070 SUPER 8 GB: **16.12 tok/s** at 30 cached expert slots — capacity-bound (PCIe
host→VRAM streaming), not a kernel or MoE deficiency:

```bash
# <!-- smoke-model --> the checkpoint this project has the most measurements on at this size
goinfer-chat pull gemma-4-26b-a4b
```

```bash
# <!-- smoke-help --> off by default; a model that does not fit then declines to the CPU path and says why
goinfer-serve -backend cuda -moe-cache-experts -model ~/models/gemma-4-26B_q4_0-it.gguf
```

`gpt-oss-20b` (`goinfer-chat pull gpt-oss-20b`; 12 GB at native MXFP4) is a second real option at
this size class — its own CUDA resident gate is measured on an 8 GB card too, see
`docs/capability-matrix.md`'s gpt-oss row — with `-moe-cache-experts` results not yet as
thoroughly measured as gemma-4-26b-a4b's.

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

- **35 model families** (counted from the generated `docs/capability-matrix.md`, which the
  `decoder` registry produces) — Gemma 3/4, Qwen 2.5/3, Llama, Mistral, Mixtral, Phi-3, DeepSeek/MLA,
  GLM, Kimi, Granite, Nemotron, Mellum and more. Full generated map:
  [docs/capability-matrix.md](docs/capability-matrix.md).
- **All four sequence-mixing families** — softmax·GQA, gated-linear (DeltaNet), state-space
  (Mamba-2), latent-KV (MLA) — plus dense and sparse-MoE.
- **Loaders** — GGUF, safetensors, GPTQ, AWQ, and prequantized
  [`.giw` bundles](docs/giw-bundles.md).
- **Quantization** — f32, int8, int8int8, int4 (W4A8), with a
  [HuggingFace logit-parity gate per family](docs/what-parity-gated-means.md). **What a given
  parity run proves is scoped to the fixtures that machine has**, and a
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
  embeddings: [docs/server.md](docs/server.md). Pointing a real agent (Claude Code, opencode) at
  it: [docs/integrations/](docs/integrations/) — `serve check`'s harness-scale tools row and
  `goinfer-chat models`' `tools:` line say which checkpoints actually hold up under a real
  agent's tool schema, measured, before you find out the way a cold-user run did.

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
