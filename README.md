# goinfer

**Run open-weight LLMs in pure Go.** No cgo, no Python, no llama.cpp — one
cross-compiled static binary, with HuggingFace logit parity.

goinfer is a pure-Go, no-cgo decoder-only LLM runtime. It loads open-weight
models — Gemma 3, Qwen 2.5/3, Llama 2/3, Mistral, Mixtral, GPT-2, Mellum —
directly from safetensors (single or sharded), GGUF, GPTQ, or AWQ checkpoints and
runs them in-process: f32/bf16/f16 plus int8 and int4 quantization, KV-cache, all
standard samplers, and constrained/structured decoding (a model that *cannot*
emit malformed JSON). Forward-pass numerics are parity-gated against HuggingFace;
matmul is SIMD-accelerated (NEON on arm64, AVX2/FMA on amd64). Because it's pure
Go with no cgo, it cross-compiles to a single static binary — no Python, no native
runtime, no provider API.

> Not to be confused with provider-orchestration libraries (e.g. teilomillet/gollm)
> that call remote LLM APIs. goinfer runs the weights itself, locally, in-process.

Built on [`aikit`](https://github.com/townsendmerino/aikit)'s embedding and tensor
primitives.

## Try it: an LLM in one file

[`demo/chat`](demo/chat) is a local coding assistant that's a **single static
binary** — the runtime *and* the model in one file. Download it, run it, chat
offline: no install, no Python, no cgo, no model download.

Grab a binary from the [latest release](https://github.com/townsendmerino/goinfer/releases/latest)
(macOS / Linux / Windows, Intel + ARM), or run from source against your own GGUF:

```bash
go run ./demo/chat --model ~/models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf
```

See [`demo/chat/README.md`](demo/chat/README.md) for commands, canned demos, and
how the single-file binary is built.

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
| `constrain` | constrained / structured decoding — a logit mask that forces output to satisfy a grammar; ships a streaming JSON grammar | — |
| `gpu` (opt-in, `-tags gpu`) | WebGPU compute backend for matmul (Metal / Vulkan / DX12) | `cogentcore/webgpu` (cgo), `aikit/encoder`, `goinfer/decoder` |

The cgo WebGPU dependency is confined to the `gpu` submodule; the default build is
pure Go, no cgo.

## Quick start

See `demo/gemma` for a working CLI: load a tokenizer (GGUF or HF), load a decoder,
stream tokens with optional sampling and JSON-constrained output.
