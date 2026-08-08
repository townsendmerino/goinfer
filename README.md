# goinfer

![goinfer chat — an entire LLM in one file](docs/assets/demo.gif)

*An entire 1.5B LLM in one file — instant boot (~0.4s), <100 MB heap, runs offline. Writes correct generic Go and **cannot** emit invalid JSON. No cgo, no Python, no model download.*

**Run open-weight LLMs in pure Go — one cgo-free static binary, portable by default and
native-GPU-fast when you want it.** ~20 model architectures, HuggingFace-parity-gated, with
schema-constrained structured output. No Python, no llama.cpp, no CUDA toolkit.

goinfer is a pure-Go, no-cgo decoder-only LLM runtime that loads open-weight checkpoints
and runs them **in-process**. What makes it different — you don't have to choose:

- **One cgo-free static binary.** Pure Go, no cgo → cross-compiles to a single file
  (macOS / Linux / Windows, Intel + ARM). No Python, no llama.cpp `.so`, no CUDA toolkit,
  no provider API. The runtime *and*, if you want, the model in one file you `scp` and run
  offline.
- **Fast when you want it — still cgo-free.** The default build is pure-Go CPU
  (SIMD-accelerated, NEON / AVX2). Opt into a GPU backend and it *stays* `CGO_ENABLED=0`:
  **native CUDA** (cgo-free, driver-only — no toolkit; vs **current Ollama v0.32.5**: ~1.7× on
  tiny 0.5B, modestly ahead on 1.5B up to ~1k tokens of context and behind beyond it, and behind
  on prefill — see the table below), **native Metal** on Apple Silicon (**at parity on small models**, behind on larger),
  and a portable **WebGPU** backend (~60–70% of native, but runs on *any* GPU and streams
  bigger-than-VRAM MoE weights). Going fast never costs you the single binary.
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

> Not to be confused with provider-orchestration libraries (e.g. teilomillet/gollm)
> that call remote LLM APIs. goinfer runs the weights itself, locally, in-process.

Built on [`aikit`](https://github.com/townsendmerino/aikit)'s embedding and tensor
primitives.

Full, generated support map (every supported `model_type`, how each family is
configured — coverage axis, MoE, RoPE, norm, loaders, modality):
[docs/capability-matrix.md](docs/capability-matrix.md) (generated from the
registry; do not hand-edit).

The Go bindings for llama.cpp still ship a native library alongside the binary, and the
pure-Go ports I know of are no longer maintained. goinfer aims to be a pure-Go option you can
actually deploy. Measured numbers, every cell with provenance:
[docs/benchmarks.md](docs/benchmarks.md).

![Mellum2 — a 12B coding MoE running GPU-resident on an 8 GB card, in pure Go](docs/assets/mellum2-gpu.gif)

*Bigger than your VRAM: JetBrains **Mellum2** — a 12B sparse-MoE coding model — decoding
**GPU-resident on a consumer 8 GB card**. The int4 experts stream into VRAM through a
pure-Go WebGPU backend (no CUDA, no Python, no llama.cpp); a 12B that won't fit 8 GB at
int8 runs **fully resident** at int4, ~13–21 tok/s. It writes idiomatic Go. Prequant the
weights once to a `.giw` bundle and it reloads in ~13 s
([docs/mellum2-resident.md](docs/mellum2-resident.md)).*

*And bigger still — **Gemma 4 26B-A4B** (a 26B MoE whose ~11.4 GB of int4 experts **do not
fit 8 GB even at 4-bit**) decodes coherently on the same card at **~17 tok/s**, running
**fully GPU-resident** — every expert executes on the GPU, streamed from host RAM into a VRAM
cache (81% hit rate) over the cgo-free CUDA backend. (Current Ollama also runs this 26B on 8 GB,
but by offloading 58% to the CPU, at ~24.5 tok/s; goinfer's distinction is all-experts-on-GPU,
not that peers can't run it — [docs/task-moe-streaming.md](docs/task-moe-streaming.md).)*

## Try it: an LLM in one file

[`demo/chat`](demo/chat) is a local coding assistant that's a **single static
binary** — the runtime *and* the model in one file. Download it, run it, chat
offline: no install, no Python, no cgo, no model download.

Grab a binary from the [latest release](https://github.com/townsendmerino/goinfer/releases/latest)
(macOS / Linux / Windows, Intel + ARM), or run from source against your own GGUF:

```bash
go run ./demo/chat --model ~/models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf
```

### Run Gemma 4 (pure Go, no cgo)

goinfer runs Google's **Gemma 4** end to end in pure Go — including the
**E-models** (E2B/E4B) with their Per-Layer-Embedding stack, and the **12B
dense**. Grab the QAT GGUF and point the demo at it (the Gemma 4 chat template
is applied automatically):

```bash
# E2B (~3 GB Q4_0) — the small one; E4B and the 12B dense work the same way
go run ./demo/chat --model ~/models/gemma-4-E2B_q4_0-it.gguf
```

```
you> What is the capital of France?
The capital of France is Paris.
```

Every Gemma 4 forward is parity-gated against the HuggingFace bf16 reference
(argmax-exact + logit cosine). Text models only — the 26B-A4B MoE and 31B
multimodal vision towers are out of scope.

See [`demo/chat/README.md`](demo/chat/README.md) for commands, canned demos, and
how the single-file binary is built.

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

## OpenAI-compatible server

[`cmd/serve`](cmd/serve) is a pure-stdlib (`net/http`, no deps) OpenAI-compatible
server — point Open WebUI, LangChain, or the OpenAI SDKs at it:

```bash
go run ./cmd/serve --model ~/models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf
# OpenAI base URL: http://localhost:8080/v1
```

`/v1/chat/completions`, `/v1/completions`, `/v1/responses`, `/v1/messages`
(Anthropic — see below), `/v1/models`;
streaming (SSE); the sampling knobs (`temperature`/`top_p`/`top_k`/`seed`/
`frequency_penalty`/`presence_penalty`/`stop`/`logprobs`); and **`response_format`**
— `{"type":"json_schema", …}` or `{"type":"json_object"}` gives schema-constrained
output the model cannot violate (the same grammar as above). The chat template is
auto-detected per model.

**Multi-model.** `--model` is repeatable as `name=path` to serve a model zoo from
one process; requests route on the OpenAI `model` field, `/v1/models` lists all,
and distinct models run in parallel (per-model mutex). Resident int8 models are
expensive — prequant `.giw` maps weights zero-copy for a cheap zoo. With
`--allow-admin` (off by default — it loads attacker-named paths), `POST
/admin/models/{load,unload}` manage the registry at runtime (unload refuses a
busy model). `--max-queue N` (default 8) bounds each model's queue: a full queue
returns 429 + Retry-After (single decode worker per model; no continuous batching).

**Responses API.** `/v1/responses` honors `input` (string or message items),
`instructions`, `text.format` (→ the same constrained grammar), `tools`, and
streaming (`response.created`/`output_text.delta`/`completed`). `store` +
`previous_response_id` continue a conversation from an in-memory ring — by
construction a prompt-prefix extension, so it rides the warm-KV cache below.

**Anthropic Messages API.** `/v1/messages` and `/v1/messages/count_tokens` speak
the second de-facto standard (the one llama.cpp, Ollama, and LM Studio also
serve), so Anthropic-speaking tools — **Claude Code** included — can point at a
pure-Go single-binary runtime. It honors `system` (string or block array),
content blocks (`text`, `tool_use`/`tool_result` replay), `tools` (note:
`input_schema`), `tool_choice` (`auto`/`any`/`tool` — `any`/`tool` ride the same
constrained decoding, so a malformed tool call is impossible), `stop_sequences`,
and streaming (the named-event SSE protocol: `message_start` → `content_block_*`
→ `message_delta` → `message_stop`, no `[DONE]`). Point Claude Code at it — all
three env vars are required:

```bash
go run ./cmd/serve --model ~/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf
ANTHROPIC_BASE_URL=http://127.0.0.1:8080 ANTHROPIC_AUTH_TOKEN=goinfer \
  ANTHROPIC_MODEL=qwen2.5-coder-1.5b-instruct-q4_k_m claude
```

Compatible, not full-spec (llama.cpp's bar): `thinking` / `cache_control` /
`metadata` are accepted and ignored. Agentic use wants a roomy-context model
(≥32k).

**Vision (image→text), pure Go.** With a Gemma 3 VL checkpoint loaded behind
`--vision <dir>` (auto-discovered when `--model` is a VL dir), `cmd/serve` accepts
images on both surfaces — OpenAI `image_url` content parts and Anthropic `image`
blocks — **base64 / `data:` URIs only** (a remote URL is never fetched: an SSRF
guard, returns 400). An image runs through the pure-Go `vision` tower (SigLIP
encoder + projector, HF-parity-gated) into the decoder's embed-by-vector seam;
image tokens count in `usage`. `demo/agent`'s web UI takes a dropped/pasted image
too. Caveat: the SigLIP prefill is CPU-heavy (~3 min/image at 896²) — correct but
slow; an int8 tower is the planned speedup (`docs/completed/task-cpu-vision-prefill.md`).

```bash
go run ./cmd/serve --model ~/models/gemma-3-4b-it --vision ~/models/gemma-3-4b-it
# then POST an image_url data: URI to /v1/chat/completions, or an image block to /v1/messages
```

**Prompt-prefix KV caching.** Across requests the server reuses the KV cache for
the longest token prefix a new prompt shares with a recent one, prefilling only
the new suffix — so a continuing chat (or an agent loop with a fixed system
prompt + tool specs) skips re-encoding the whole history. Reuse is exact
(bit-identical to a cold prefill). `--kv-sessions N` sets how many conversations
to keep warm (default 4; 0 disables); `--session-dir DIR` persists the warm
sessions to disk and restores them on restart.

**Embeddings.** Point `--embed-model` at a [CodeRankEmbed](https://huggingface.co/nomic-ai/CodeRankEmbed)
HF snapshot to serve `/v1/embeddings` (`--embed-quant f32|q8`). `--model` and
`--embed-model` are each optional and can run together — generation and
embeddings from one process, or either alone:

```bash
go run ./cmd/serve --embed-model ~/models/coderankembed         # /v1/embeddings only
```

`input` (string or array), `encoding_format: float|base64`, and `dimensions`
(truncate + renormalize) follow the OpenAI shape; vectors are L2-normalized. For
this encoder's asymmetric query/document encoding, an optional `input_type:
"query"|"document"` (default `document`, the Cohere/Voyage convention) selects the
query instruction prefix.

## Status

Pre-1.0; the forward-pass / quantization contract is parity-gated and stable, the
loader and architecture-descriptor surface is still moving as new model families
land. See `CHANGELOG.md`.

## Install

```bash
go get github.com/townsendmerino/goinfer
```

### Modules

goinfer ships as **four Go modules**. The three GPU backends are separate modules so their
dependencies (`cogentcore/webgpu` and its cgo, `eitamring/gocudrv`, `ebitengine/purego`)
never enter the dependency graph of a build that doesn't ask for them.

| Module path | Contents |
|---|---|
| `github.com/townsendmerino/goinfer` | everything in the Packages table below except the three backends |
| `github.com/townsendmerino/goinfer/gpu` | WebGPU backend (`-tags gpu`) |
| `github.com/townsendmerino/goinfer/cuda` | native CUDA backend (`-tags cuda`) |
| `github.com/townsendmerino/goinfer/metal` | native Metal backend (`-tags metal`) |

**You normally name only the root.** Its `go.mod` requires the other three at versions known
to work with it, so `go get github.com/townsendmerino/goinfer` brings all four and a
`-tags cuda` build resolves without further action. Naming a backend module explicitly is
only needed to pin, vendor, or audit it:

```bash
go get github.com/townsendmerino/goinfer/cuda@v0.9.0
```

**The backend modules are versioned independently of the root and of each other** — they are
not in lockstep, since a root-only release doesn't retag them. Backend tags carry the module
path as a prefix — the three backends are currently tagged `gpu/v0.9.0`, `cuda/v0.9.0`, and
`metal/v0.9.0` — which is how Go's module proxy resolves a submodule tag; the bare `v0.9.x`
tags are the root's. When in doubt, take the root's requirement rather than picking a backend
version yourself.

## Packages

| Package | Purpose | Deps beyond stdlib |
|---|---|---|
| `decoder` | generic decoder-only forward pass; f32/bf16/f16 + int8/int4; safetensors/GGUF/GPTQ/AWQ; KV-cache; samplers | `aikit/embed`, `aikit/linalg`, `goinfer/tokenizer` |
| `tokenizer` | BPE tokenizers the decoder LLMs ship — byte-level + SentencePiece byte-fallback, from `tokenizer.json` or a bare `.gguf`; HF-exact id parity | `aikit/embed`, `golang.org/x/text` |
| `constrain` | constrained / structured decoding — a logit mask that forces output to satisfy a grammar; streaming JSON grammar + JSON Schema (and Go-struct) compiler | — |
| `chat` | chat-template detection + byte-exact native renderers (Gemma 3/4, ChatML/Qwen, Llama-3, Mistral) and per-family tool calling (render + parse) | — |
| `gpu` (opt-in, `-tags gpu`) | WebGPU compute backend for matmul (Metal / Vulkan / DX12) | `cogentcore/webgpu` (cgo), `aikit/encoder`, `goinfer/decoder` |
| `cuda` (opt-in, `-tags cuda`) | cgo-free native CUDA decode backend — dlopen libcuda + NVRTC, dense residency, `CGO_ENABLED=0` | `eitamring/gocudrv`, `goinfer/decoder` |
| `metal` (opt-in, `-tags metal`) | cgo-free native Metal decode backend — purego / Obj-C, MSL compiled at runtime, dense residency, darwin, `CGO_ENABLED=0` | `ebitengine/purego`, `goinfer/decoder` |

The cgo WebGPU dependency is confined to the `gpu` submodule; the two native GPU
backends (`cuda`, `metal`) are **cgo-free**. Either way the default build is pure Go,
no cgo — a backend is compiled only when you pass its build tag.

## Running on a GPU

The default build is pure-Go CPU. Three **opt-in** GPU backends accelerate decode.
Each lives in its own submodule with its own binaries under `<submodule>/cmd/` — the
pure-Go root module never imports them, so `go install …/cmd/serve` and any SBOM of the
root stay free of webgpu/purego/gocudrv (audit M-19). The `serve` and `chat` binaries
exist in each flavor:

| Backend | Binary (server / REPL) | Platform | cgo |
|---|---|---|---|
| WebGPU | `./gpu/cmd/serve`, `./gpu/cmd/chat` (`-tags gpu`) | any GPU (Metal / Vulkan / DX12) | yes (confined to the `gpu` submodule) |
| CUDA | `./cuda/cmd/serve`, `./cuda/cmd/chat` (`-tags cuda`) | NVIDIA — Linux / Windows x86-64 | **no** — `CGO_ENABLED=0`, dlopens the driver |
| Metal | `./metal/cmd/serve`, `./metal/cmd/chat` | Apple Silicon | **no** — `CGO_ENABLED=0`, purego / Obj-C |

The native **CUDA** and **Metal** backends need only the platform's GPU driver —
**no CUDA toolkit, no Xcode, no Python, no cgo** — and are selected at runtime with
`--backend`.

> **Upgrading from ≤ v0.9.x?** The old `go build -tags cuda …/cmd/serve` (the *root*
> command) no longer enables a backend — since v0.10.0 the root is pure-Go and the tag is a
> no-op. Build the **submodule entrypoint** instead (the commands below). Passing a backend
> tag to the root now fails the build with a message pointing here, rather than silently
> producing a CPU binary.

**Out-of-tree** (you `go get` goinfer, no checkout) — build the submodule entrypoint by its
full module path; nothing else is needed:

```bash
# CUDA server / REPL
CGO_ENABLED=0 go build -tags cuda github.com/townsendmerino/goinfer/cuda/cmd/serve
CGO_ENABLED=0 go build -tags cuda github.com/townsendmerino/goinfer/cuda/cmd/chat
# WebGPU (cgo)
go build -tags gpu   github.com/townsendmerino/goinfer/gpu/cmd/serve
# Metal (darwin; the module is darwin-gated, so no -tags)
go build             github.com/townsendmerino/goinfer/metal/cmd/serve
```

**In-tree** (a checkout) — run straight from the submodule; its `go.mod` resolves the root
via a `replace`, so no workspace setup is needed:

```bash
# NVIDIA — cgo-free native CUDA
cd cuda && CGO_ENABLED=0 go run -tags cuda ./cmd/serve --backend cuda \
    --model ~/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf     # or ./cmd/chat for the REPL

# Apple Silicon — cgo-free native Metal (darwin-gated; no -tags needed)
cd metal && CGO_ENABLED=0 go run ./cmd/serve --backend metal --quant int8int8 \
    --model ~/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf
```

### Measured throughput (server-to-server, q4_k_m 4-bit)

Both goinfer and the reference are driven through their own HTTP server — sampling,
detokenize, and JSON all included — so there's no methodology gap to discount:

Peers: **Ollama-CUDA v0.32.5** (2026-07) and **Ollama-Metal 0.32.0** (2026-07-16), each on
the same machine as its goinfer column. *(Earlier revisions of this table quoted Ollama 0.5.7
from 2025-01 — ~18 months stale — which inflated the CUDA ratios to ~2×; re-anchored below.)*

| Model | goinfer CUDA | Ollama-CUDA v0.32.5 | goinfer Metal | Ollama-Metal |
|---|---|---|---|---|
| Qwen2.5-Coder-0.5B | ~476 tok/s (**1.78×**) | ~268 | ~128 tok/s (**1.03×**) | ~124 |
| Qwen2.5-Coder-1.5B | *CUDA by KV depth ↓* | | ~61 tok/s (**0.77×**) | ~79 |

**Qwen2.5-Coder-1.5B, CUDA decode by KV depth** (same RTX 2070 SUPER, best-of-3):

| context | goinfer CUDA | Ollama-CUDA v0.32.5 | |
|---|---|---|---|
| 128 | ~226.6 tok/s | ~197.5 | goinfer **1.15×** |
| 512 | ~207.3 tok/s | ~191.7 | goinfer **1.08×** |
| 2048 | ~160.1 tok/s | ~186.6 | Ollama **1.17×** |
| 3900 | ~123.5 tok/s | ~180.7 | Ollama **1.46×** |

**What the curve shows:** on CUDA goinfer is ahead on **tiny models** (0.5B, launch-bound) and
on 1.5B up to **roughly 1000 tokens of context**; the 1.5B decode curve crosses over there —
ahead below it, behind above it, and the gap widens with depth (1.15× ahead at 128, 1.46× behind
at 3900), because Ollama's flash attention stays nearly flat as context grows. On **prefill**
goinfer is **~4–5× behind** current Ollama (`docs/benchmarks.md` §B2). Metal is at parity on 0.5B
and behind on 1.5B (issue-bound — no DP4A on Apple GPUs). Each engine is compared **only against
its peer on the same machine** — CUDA on an RTX 2070 SUPER, Metal on an M1 Pro — so the absolute
tok/s do *not* compare across the CUDA and Metal columns (that would compare two graphics cards,
not two engines). Best of 3 warm runs; full provenance — hardware, driver, peer versions, method — in
[docs/benchmarks.md](docs/benchmarks.md).

### What runs on the GPU

GPU-resident decode covers a subset of architectures. **Everything else runs on the
pure-Go CPU path automatically.** A shared feature taxonomy checks each model's required
features against what the backend implements; an unsupported architecture is *declined at
load and falls back to CPU* rather than run with a feature quietly dropped.

| Family | CUDA | Metal |
|---|---|---|
| Qwen2 · Qwen3 · Llama | ✅ resident | ✅ resident |
| Mistral · Phi-3-mini-4k | ✅ resident | ✅ resident¹ |
| Gemma 3 | ✅ resident³ | ✅ resident³ |
| MoE — Mixtral · Qwen2-MoE · Qwen3-MoE · GLM-MoE | ✅ resident⁴ | ✅ resident² |
| Gemma 4 · MLA · DeltaNet/YaRN | CPU fallback | CPU fallback |

The full per-family × 4-backend (CPU · WebGPU · CUDA · Metal) table is **generated** from the
residency predicate (`decoder.ResidentEligible`) and freshness-gated in CI, so it can never drift
from what a backend actually admits: [docs/hardware-matrix.md](docs/hardware-matrix.md).

¹ Metal Mistral-7B needs > 16 GB unified memory (int8 + int4). Both backends implement
qk-norm + sliding-window; Metal also does partial rotary, so a partial-rotary Phi
variant is resident on Metal but falls back on CUDA.

³ Gemma 3 (both backends) covers the sandwich-norm block, GeGLU, the (1+w) RMS offset, the
√hidden embedding scale, and Gemma's dual RoPE base — validated on a real gemma-3-4b-it against
the CPU path. Metal parity was gated on a GELU-tanh overflow fix (the `<bos>` massive-activation
gate drove `tanh`'s argument past its internal `exp` range → NaN; clamped). Gemma 4 stays on
CPU: it needs logit-softcap and has its own forward (per-layer head_dim, KV-sharing, PLE).

⁴ CUDA MoE runs Mixtral and GLM-MoE resident (on-GPU router, row-stacked int4 experts, ungated
shared expert). Qwen2-MoE / Qwen3-MoE decline to CPU on CUDA — their gated shared expert
(sigmoid-scaled) isn't built yet.

² Metal MoE (router + stacked experts + shared expert) is validated by assembly
equivalence (identical experts ≡ the dense FFN, cosine 1.0) + per-kernel parity vs CPU;
a real MoE checkpoint needs a Mac with enough unified memory (Qwen1.5-MoE-A2.7B is 14.3B
≈ 14 GB at int8 load), so the real-model e2e cross-check runs on the CUDA box. The
DeltaNet/Llama-4/Gemma hybrids stay on CPU (declined before residency).

An unlisted or unsupported model still runs — in pure Go on the CPU. The portable
WebGPU backend (`-tags gpu`) covers a broader resident set (MoE, MLA, SSM, YaRN); see
[docs/capability-matrix.md](docs/capability-matrix.md) for the full map.

> **Note:** in `cmd/serve`, a GPU-resident model skips prompt-prefix KV reuse and
> speculative decoding — the resident decode path is fast enough that the per-request
> session optimization isn't worth it. The OpenAI API is stateless (clients resend the
> whole conversation), so this is a throughput trade, not a correctness change.

## Prequantized weight bundles (`.giw`)

Loading a GGUF quantizes its weights on every launch. A **`.giw` bundle** stores the
already-quantized resident weights alongside a metadata-only GGUF (the source truncated at
the tensor-data boundary, so it still carries the tokenizer). Loading one skips
dequant/requant entirely — the weights are aliased straight from the file image rather than
copied into a multi-GB heap.

Build one with `cmd/prequant`:

```bash
go run ./cmd/prequant -o model.giw ~/models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf
```

| Flag | Meaning |
|---|---|
| `-o PATH` | output bundle path (**required**) |
| `-quant` | quant baked into the bundle: `int8int8` (default), `int8`, or `int4` |
| `-embed-int4` | with `-quant int4`, store the token-embedding / LM-head table at int4 too instead of pinning it at int8 — roughly halves the head's per-token traffic on a big-vocab model |

The quant is **baked in at build time**: a bundle made with `-quant int8int8` is an int8int8
model, and `serve --quant` cannot change it afterwards. Build a separate bundle per quant you
intend to serve.

Then serve it like any other model:

```bash
./serve --model model.giw
```

`serve --stream-weights` also produces these on demand — a plain `.gguf` is transcoded to a
sidecar `.giw` cache on first use, so the one-time cost is paid once rather than per launch.

## Quick start

See `demo/gemma` for a working CLI: load a tokenizer (GGUF or HF), load a decoder,
stream tokens with optional sampling and JSON-constrained output. `demo/chat` is a
single-binary local chat GUI, and `demo/agent` is a fully-local stdlib RAG coding
agent built on goinfer.
