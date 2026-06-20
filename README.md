# goinfer

![goinfer chat — an entire LLM in one file](docs/assets/demo.gif)

*An entire 1.5B LLM in one file — instant boot (~0.4s), <100 MB heap, runs offline. Writes correct generic Go and **cannot** emit invalid JSON. No cgo, no Python, no model download.*

**Run open-weight LLMs in pure Go.** No cgo, no Python, no llama.cpp — one
cross-compiled static binary, with HuggingFace logit parity.

goinfer is a pure-Go, no-cgo decoder-only LLM runtime. It loads open-weight
models — Gemma 3/4, Qwen 2.5/3, Llama 2/3, Mistral, Mixtral, Qwen-MoE,
GLM-4.5/4.6, Granite-4.0-H, Nemotron-H, DeepSeek-V2/V3 + Kimi K2 (incl. K2.7-Code) (MLA), Phi-3/Phi-4, GPT-2, Mellum2 —
directly from safetensors (single or
sharded), GGUF, GPTQ, or AWQ checkpoints and runs them in-process: f32/bf16/f16 plus
int8 and int4 quantization, KV-cache, all standard samplers, LoRA adapters (PEFT,
merged at load), and constrained/structured decoding (a model that *cannot* emit
malformed JSON). It runs the major modern attention/sequence-mixing families in one
binary — softmax/GQA, **gated-linear** (Gated DeltaNet), **state-space**
(Mamba-2), and **latent-KV** (DeepSeek MLA). Forward-pass numerics are parity-gated against HuggingFace;
matmul is SIMD-accelerated (NEON on arm64, AVX2/FMA on amd64). Because it's pure
Go with no cgo, it cross-compiles to a single static binary — no Python, no native
runtime, no provider API.

> Not to be confused with provider-orchestration libraries (e.g. teilomillet/gollm)
> that call remote LLM APIs. goinfer runs the weights itself, locally, in-process.

Built on [`aikit`](https://github.com/townsendmerino/aikit)'s embedding and tensor
primitives.

Full, generated support map (every supported `model_type`, how each family is
configured — coverage axis, MoE, RoPE, norm, loaders, modality):
[docs/capability-matrix.md](docs/capability-matrix.md) (generated from the
registry; do not hand-edit).

**The lane:** goinfer runs the weights *in-process in pure Go* — the single-file,
zero-install, HF-parity-gated lane no other maintained runtime occupies (the Go
llama.cpp bindings still ship a native `.so`; the pure-Go ports are archived toys).
On a GPU it decodes at **~60–70% of llama.cpp/Ollama-CUDA at equal 4-bit quant** — a
portable WebGPU backend vs years-tuned CUDA — in a static binary that boots in ~0.5 s.
Full capability matrix + measured numbers, every cell with provenance:
[docs/benchmarks.md](docs/benchmarks.md).

![Mellum2 — a 12B coding MoE running GPU-resident on an 8 GB card, in pure Go](docs/assets/mellum2-gpu.gif)

*Bigger than your VRAM: JetBrains **Mellum2** — a 12B sparse-MoE coding model — decoding
**GPU-resident on a consumer 8 GB card**. The int4 experts stream into VRAM through a
pure-Go WebGPU backend (no CUDA, no Python, no llama.cpp); a 12B that won't fit 8 GB at
int8 runs **fully resident** at int4, ~13–21 tok/s. It writes idiomatic Go and **cannot**
emit invalid JSON. Prequant the weights once to a `.giw` bundle and it reloads in ~13 s
([docs/mellum2-resident.md](docs/mellum2-resident.md)).*

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

## Packages

| Package | Purpose | Deps beyond stdlib |
|---|---|---|
| `decoder` | generic decoder-only forward pass; f32/bf16/f16 + int8/int4; safetensors/GGUF/GPTQ/AWQ; KV-cache; samplers | `aikit/embed`, `aikit/linalg`, `goinfer/tokenizer` |
| `tokenizer` | BPE tokenizers the decoder LLMs ship — byte-level + SentencePiece byte-fallback, from `tokenizer.json` or a bare `.gguf`; HF-exact id parity | `aikit/embed`, `golang.org/x/text` |
| `constrain` | constrained / structured decoding — a logit mask that forces output to satisfy a grammar; streaming JSON grammar + JSON Schema (and Go-struct) compiler | — |
| `chat` | chat-template detection + byte-exact native renderers (Gemma 3/4, ChatML/Qwen, Llama-3, Mistral) and per-family tool calling (render + parse) | — |
| `gpu` (opt-in, `-tags gpu`) | WebGPU compute backend for matmul (Metal / Vulkan / DX12) | `cogentcore/webgpu` (cgo), `aikit/encoder`, `goinfer/decoder` |

The cgo WebGPU dependency is confined to the `gpu` submodule; the default build is
pure Go, no cgo.

## Quick start

See `demo/gemma` for a working CLI: load a tokenizer (GGUF or HF), load a decoder,
stream tokens with optional sampling and JSON-constrained output. `demo/chat` is a
single-binary local chat GUI, and `demo/agent` is a fully-local stdlib RAG coding
agent built on goinfer.
