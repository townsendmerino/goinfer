# Changelog

All notable changes to goinfer are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

The forward-pass and quantization numerics are parity-gated against HuggingFace
and are the stable contract. The loader and architecture-descriptor surface is
pre-1.0 and may change as new model families and quant formats land.

## [Unreleased]

## [v0.5.0] — 2026-06-11

_A **minor bump** per SemVer: `constrain` schema validation carries a
user-visible behavior change (under Changed) — shipped alongside the Anthropic
Messages API (Claude Code can point at a pure-Go runtime), the f16 GPU KV cache,
the ~2.4× sparse-MoE / ~3.4× dense prefill speedups, and the now soak-verified
untrusted-input fuzzing hardening._

### Changed
- **`constrain` rejects unenforceable / unsatisfiable JSON Schemas at compile**
  instead of silently accepting them. Schemas that previously compiled but could
  never be enforced now return an error: a `required` entry naming a property
  absent from `properties`, an array with `maxItems` < `minItems`, and
  negative / non-integer array bounds. **Behavior change** — a caller that relied
  on the old silent-accept (and the non-conforming output it produced) now gets a
  compile error instead. (`3cb27e6`)
- **aikit `v1.1.1` → `v1.3.0`** — a hardened `embed.parseGGUF` (overflow-safe
  length checks, map/array pre-sizing capped to the remaining input, so a hostile
  GGUF header errors instead of panicking / OOM-ing one import down) and the
  scores·V attention vectorization (`v1.2.1`), plus `MatmulBTAcc64` — an
  f64-accumulating A·Bᵀ matmul (bit-identical to a sequential-order f64 reduction)
  that the MoE attention path needs (`v1.3.0`). Bumped in both the root and `gpu`
  modules. (`b77922c`)
- **MoE attention now runs through the f64-accumulating matmul** (decode AND
  prefill). It is more accurate than — and may differ at routing near-ties from —
  the prior f32/scalar path, but stays parity-gated against the HF bf16 oracle
  (Mellum2 logit/window parity unchanged). The discrete top-k router amplifies any
  attention reassociation, so the precision is load-bearing, not cosmetic.

### Added
- **Anthropic Messages API** (`POST /v1/messages`, `POST /v1/messages/count_tokens`)
  alongside the OpenAI surface — point **Claude Code** (or any tool that speaks
  `ANTHROPIC_BASE_URL` + `ANTHROPIC_AUTH_TOKEN` + `ANTHROPIC_MODEL`) at a pure-Go
  single-binary runtime. The second de-facto-standard chat surface, served by the
  same edge-translation trick llama.cpp/Ollama/LM Studio use: the request is
  mapped into the existing internal path (system + turns + sampling + tools) and
  the result mapped back out — `drive`/`prepare` unchanged. Honors `system`
  (string or text-block array), content blocks (`text`, plus `tool_use`/
  `tool_result` conversation replay), `tools` (`input_schema`) and `tool_choice`
  (`auto`/`any`/`tool` — `any`/`tool` reuse the constrained-decoding path, so a
  malformed tool call is physically impossible), `stop_sequences`, and the named-
  event streaming SSE protocol (`message_start` → `content_block_*` →
  `message_delta` → `message_stop`, no `[DONE]`). Compatible-not-full-spec (the
  llama.cpp bar): image blocks 400; `thinking` / `cache_control` / `metadata`
  accepted and ignored (Claude Code sends `cache_control` on every request).
  Pure stdlib `net/http`, no new deps. (`d0b0f66`)
- **`demo/agent`: a fully-local stdlib RAG coding agent** over goinfer + the
  `ken` MCP retrieval server — a CLI and a single-binary web UI, kept in its own
  module so its MCP dependency stays out of the goinfer root graph. (`36eae14`)
- **f16 GPU KV cache** (`--kv f16`, opt-in; default `f32` stays bit-exact) — halves
  per-token KV bytes on the full-residency path, unlocking **32k context for a
  7B-int4 on an 8 GB card** (32k f16 fits in the same VRAM as 16k f32 — measured
  6912 vs 6926 MiB on an RTX 2070 SUPER). Manual WGSL f16 via the core
  `pack2x16float`/`unpack2x16float` builtins (no `shader-f16` device feature, so the
  CI software adapter still compiles). Decode parity vs an f32 cache: argmax
  preserved, full-logit cosine ≥0.998 over an 8000-key context. (`138d5e0`,
  `b40d8dd`)
- **Chunked-parallel Gated DeltaNet scan kernel** (`deltanet_chunked.go`) — the
  reformulation the `qwen3_5_moe` perf rewrite needs, unrolling the gated
  delta-rule recurrence over a chunk into the matmul-friendly form. Proven
  algebraically equivalent to the sequential recurrence over random
  inputs/chunk sizes (`TestGatedDeltaNet_chunkedMatchesSequential`; self-contained,
  no checkpoint/torch). Not yet on the hot path — the forward is still
  single-token streaming — so it's a zero-regression-risk reference. (`750b4ac`)

### Security
- **Hostile model files / request bodies now error, never panic.** A fuzzing pass
  over the untrusted-input surface (Go native fuzzing, CI-enforced via committed
  seed corpora) found and fixed five panic/OOM vectors, each turned into a typed
  error with a regression test:
  - GGUF: a metadata `block_count` ≥ 2³¹ overflowed `int` to a negative
    `NumLayers` → `make([]LayerWeights, …)` panic. (`9cdf98d`)
  - `.giw` bundle: a near-`maxint64` v2 length overflowed the bound check in
    `cur.take` → slice-bounds panic. (`92f0d83`)
  - `tokenizer.json`: a hostile token id overflowed / OOM'd / negative-indexed the
    `idToPiece` allocation. (`8e0b8bd`)
  - GPTQ/AWQ: dims not a multiple of 8 slipped past the integer-division shape
    check → dequant indexed out of bounds. (`aebbd45`)
  - Serialized `.giw` weights: a corrupt blob could drive a multi-GB allocation;
    the layer/expert count gates were tightened to sane bounds. (`c6b5ff2`)
  Fuzz targets now cover `constrain`, the GGUF / serialized-weights / GPTQ-AWQ
  loaders, the `.giw` frame, `tokenizer.json`, and the serve request shapers, and a
  `-race` serve soak/chaos test exercises multi-model + admin + sessions under
  concurrency (zero goroutine leaks, byte-identical warm-KV restore). The GGUF
  interpretation fuzzer was re-enabled after the aikit hardening landed (`e3bc98d`).

### Performance
- **~3.4× prompt prefill** — vectorized the prefill attention (QKᵀ + scores·V)
  onto the SIMD path. (`7fa82c2`)
- **~1.7× Gemma 4 prefill** — the same treatment for gemma4 attention. (`88b7aaa`)
- `qwen3_5_moe`: softmax attention reuses the cache scratch for scores
  (`4fcc069`), and its M=1 projections route through the SIMD A·Bᵀ kernel
  (`443ab7a`).
- **~2.4× sparse-MoE prompt prefill** (Mellum2 12B-A2.5B: 3.36 → 8.11 tok/s at a
  1024-token context). `canBatchN` now admits standard sparse-MoE (Mellum / Mixtral),
  so their prefill batches the attention onto the SIMD A·Bᵀ kernel like the dense
  families — a CPU profile put the scalar per-token attention at ~83% of MoE prefill,
  the expert matmuls only ~17% (those stay per-token: the router picks different
  experts per row). The batched attention uses the f64-accumulating `MatmulBTAcc64`
  and decode is routed through the same kernel, so the result is **bit-identical to
  the sequential forward** (the discrete router won't tolerate an f32 matmul's ~1e-6
  reassociation). The qwen3_5_moe DeltaNet hybrid and Gemma 4 keep their own forwards.

## [v0.4.0] — 2026-06-09

### Added
- **GPU full-residency decode (W4A8 int4) — run bigger models pure-GPU.** Real
  int4 `.giw` models now decode entirely on the GPU through `decoder.Generate` on
  the `webgpu` backend (dense Qwen2/Llama): the full-token forward is the resident
  DecodeRunner, not the per-matmul staged path. The win is **footprint, not a
  speed record** — int4 halves resident weights, so a **7B int4 fits and decodes
  at ~51 tok/s on an 8 GB card** (the model class that does NOT fit at int8), at
  **~71% of llama.cpp-CUDA (q4) at equal 4-bit quant** (51.7 vs 72.8 tok/s; greedy
  output matches the CPU decode bit-for-bit on the first tokens). int8 residency
  peaks ~89.7 tok/s on the 1.5B (3.5× the staged hybrid; **61% of Ollama-q8** at
  equal int8 quant, 89.7 vs 147). v1 limits: **stateless `Generate` only**
  (`Session`/prefix-reuse/`GenerateSpeculative` fall back to the staged path),
  **16k context cap** (f32 KV), **eligible archs only** (dense Qwen2/Llama;
  MoE/Gemma/hybrid → staged). See `docs/gpu-assessment.md` §0.0 + the §1 decision
  matrix. (The `.giw` bundle's weights length is now u64 — v2 — so int4 models
  past 4 GiB, i.e. the 7B+ class, serialize without truncation; v1 bundles still
  load.)
- **cmd/serve: multi-model + admin + Responses API.** `--model` is now repeatable
  as `name=path` to serve a model zoo from one process; requests route on the
  OpenAI `model` field (exact match, or the sole model for single-model compat),
  unknown → an OpenAI-shaped 404, and `/v1/models` lists all. Each model has its
  own mutex (distinct models run in parallel) and warm-KV dir
  (`--session-dir/<fp>/`). **Admin API** (gated behind `--allow-admin`, default
  off — RCE-adjacent): `POST /admin/models/{load,unload}`, with unload refusing a
  busy model (409) and snapshotting its warm KV. **`/v1/responses`** (OpenAI
  Responses API): `input`/`instructions`/`text.format`(→constrain)/`tools`,
  streaming event shapes, and `store`/`previous_response_id` (an in-memory ring
  that rides the per-model sessionLRU for warm KV). **Backpressure**: a bounded
  per-model queue (`--max-queue`, default 8) returns 429 + Retry-After when full
  (single decode worker per model; not continuous batching). Internally the
  generative half is a `loadedModel` registry. Pure stdlib `net/http`, no deps.
- **Qwen3.5/3.6-MoE (`qwen3_5_moe`)** — the hybrid linear/softmax-attention MoE.
  Most layers are **Gated DeltaNet** (linear attention with a recurrent matrix
  state — short causal conv + gated delta rule + gated RMSNorm, its own forward
  primitive), the rest gated softmax attention (double-width q_proj → query‖gate,
  QK-norm, partial RoPE, output gate), over a 256-expert + shared MoE on every
  layer. A **hybrid cache** holds KV for the softmax layers + a fixed-size
  `deltaState` per linear layer; prefix-reuse / speculative fall back for the
  hybrid (recurrent state isn't position-truncatable). Loads from safetensors,
  bit-exact vs the HF oracle: the DeltaNet primitive op-for-op (cosine 1.0,
  `deltanet_test.go`) and the full model argmax + cosine 1.0
  (`qwen35_forward_test.go`). **Loads from both safetensors and GGUF** — the
  `qwen3_5_moe` loader reverses llama.cpp's fused/stacked transform back to the
  per-expert layout — with **real-checkpoint int8 parity on the 35B-A3B** (Gate 2:
  argmax **74/80**, sample cosine **0.99466** vs the banked HF bf16 golden), and the
  GGUF path proven by weight-diff against the safetensors load. **Honest scope:**
  this is the **text decoder of the Qwen3-VL 35B-A3B** model (the language tower),
  and the hybrid arch runs the **staged path — not GPU residency**. Parity-first f32
  forward. See `docs/qwen3_5_moe.md`.
- **Mellum2 chat template** — `chat.Mellum2()` (a named ChatML alias) + a `Detect`
  fingerprint (its distinctive `normalize_content` macro) so JetBrains Mellum2 is
  identified as `mellum2` by `cmd/serve` / `demo/chat` rather than falling through
  to generic ChatML. Its template is ChatML byte-for-byte (`<|im_start|>`/
  `<|im_end|>` turns, stop `<|im_end|>` = EOS id 28, Hermes `<tool_call>` tools),
  verified against HF `apply_chat_template` (`testdata/chat_goldens/mellum2.json`:
  system+user, multi-turn, no-system) — the same byte-exact gate as the other five
  families. Tools ride the existing ChatML Hermes `RenderTools`/`ParseToolCalls`.

### Changed
- **Mellum2 parity-gated** — Mellum2 is no longer the README family list's parity
  exception. `TestMellum2_logitParity` pins the forward (MoE 64/top-8, 3:1
  sliding/full interleave, YaRN-on-full RoPE, QK-norm) against the HF bf16 oracle
  on a chat-templated prompt: argmax exact (`Paris`), sample-256 cosine **0.99955**
  (int8int8 vs bf16). The sliding-window EVICTION path — untested when validation
  stayed under the 1024 window — now has a model-free unit proof
  (`TestMellum2_slidingWindowEviction`: a sliding layer's output is invariant to an
  out-of-window key, a full layer's isn't) plus a real-checkpoint past-window point
  (`TestMellum2_windowParity`: 1441-token prompt, cosine **0.99636**).
- **CPU int4 decode is now usable** — the `decoder.Generate` int4 path moved onto
  aikit v1.1.1's `MatmulBTW4A8` (int4×int8 integer decode) from the f32-activation
  `MatmulBTQ4`, which was dequant-bound at M=1. CPU int4 goes from **~20–45× slower
  than int8 to ~1.4–1.9× int8** (arm64 + amd64), so int4 is now a real CPU
  footprint option, not only a GPU one.
- **aikit → v1.1.1** — adds the `MatmulBTW4A8` CPU int4 kernel above (the
  `embed`/`linalg` deps; both modules track it).

## [v0.3.0] — 2026-06-07

### Added
- **Embeddings endpoint** — `cmd/serve` serves OpenAI `/v1/embeddings` from an
  `--embed-model` CodeRankEmbed encoder (`--embed-quant f32|q8`), alongside or
  instead of the generative `--model`; endpoints register for whatever loaded.
  Honors `input` (string|array), `encoding_format` (`float`|`base64`), and
  `dimensions` (truncate + renormalize); vectors are L2-normalized. An optional
  `input_type` (`query`|`document`, default document — the Cohere/Voyage
  convention) selects the encoder's asymmetric query instruction prefix.
  `usage.prompt_tokens` is counted via the encoder's tokenizer; the encoder is
  goroutine-safe, so embeddings serve without the decoder's lock. Rides aikit
  v1.0.0's frozen `encoder` API (`Encode`/`EncodeBatch`/`HiddenDim`).
- **Prompt-prefix KV caching / cross-call session reuse** — `decoder.Session`
  pairs a KV cache with the tokens materialized in it and reuses the KV of the
  longest token prefix a new prompt shares, prefilling only the divergent suffix
  (exact — bit-identical to a cold prefill). `cmd/serve` layers a session LRU
  (`--kv-sessions`, default 4; 0 disables) so a continuing chat — or an agent
  loop with a fixed system prompt + tool specs — skips re-encoding the whole
  history; `--session-dir` persists warm sessions to CRC+identity-guarded
  `.giw-kv` snapshots and restores them across restarts. Fixes
  `KVCache.TruncateTo` to derive each layer's stride from what it holds, so it is
  correct on Gemma 4's per-layer KV widths and KV-shared tail layers. Reuse
  parity is gated id-for-id vs a cold prefill (Qwen 2.5 + Gemma 4 E2B).
- **LoRA adapter loading** — `decoder.Options.LoRA` merges a PEFT adapter
  (`adapter_config.json` + `adapter_model.safetensors`) into the base weights at
  load: W′ = W + (α/r)·B·A (α/√r under rslora), applied to the f32 weight *before*
  quantization, so a merged model costs nothing extra at decode. Targets the
  per-layer attention/MLP projections; an unsupported target (e.g. `embed_tokens`)
  or a GGUF base is a loud error. `demo/chat` and `cmd/serve` gain `--lora`.
- **Tool calling** — composes chat templates + JSON-Schema constraint + a
  per-family parser (parse against the model's template, not a naive JSON scan).
  `chat.Template.RenderTools` declares tools in each family's syntax and
  `ParseToolCalls` recovers structured calls; supported: ChatML/Qwen (Hermes
  `<tool_call>`), Mistral (`[TOOL_CALLS]`), Llama-3 (bare `{name,parameters}`),
  and Gemma 4's bespoke `<|tool_call>` micro-language (incl. its declaration
  format, byte-exact). `constrain.ToolCallGrammar` constrains a call to a tool's
  schema (the family wrapper around `{"name":const,…:<args schema>}`). `cmd/serve`
  honors OpenAI `tools` / `tool_choice`: renders declarations, constrains the call
  when unambiguous (one tool or a forced function — Gemma 4 is parse-only, no
  logit constraint), and returns `tool_calls` with `finish_reason:"tool_calls"`.
  Tested: per-family parse + round-trip, the tool-grammar property (every
  constrained generation is a valid, schema-conforming call), and a server
  integration call (Qwen → `get_weather(...)`).
- **OpenAI-compatible HTTP server** (`cmd/serve`) — pure stdlib `net/http`, no
  deps. `/v1/chat/completions`, `/v1/completions`, `/v1/models`; streaming via
  SSE; the sampling knobs (temperature/top_p/top_k/seed/frequency_penalty/
  presence_penalty/stop/logprobs) map onto goinfer's sampler; and
  **`response_format`** (`json_object` / `json_schema`) rides the constrain
  grammar for output the model cannot violate. Chat templates auto-detected per
  model (the `chat` package). Point Open WebUI / LangChain / the OpenAI SDKs at
  `http://host:8080/v1`. (Its own release; depends on the chat-template,
  sampling, and JSON-Schema work above.)
- **K-quant coverage confirmed: Q6_K and Q4_K_S** now have dedicated logit-parity
  tests (TinyLlama, same harness as the other GGUF quant tests), alongside the
  existing Q2_K/Q3_K_M/Q4_K_M/Q5_K_M. The dequant paths (via `aikit/embed`) were
  already present — most HF repos ship Q5_K_M/Q6_K next to Q4_K_M and goinfer
  loads them all; these tests pin per-quant parity (argmax exact; cosine-floored).
- **JSON Schema constrained decoding** (the v0.2 flagship) — `constrain.JSONSchema`
  compiles a JSON Schema into the existing incremental byte-level `Grammar`, so the
  streaming logit masker drives it unchanged: the model **physically cannot** emit
  non-conforming JSON (invalid tokens are −∞, i.e. unreachable). Supported subset:
  objects (required + optional, `additionalProperties:false`), arrays
  (`items`/`minItems`/`maxItems`), `string`/`number`/`integer`/`boolean`/`null`,
  `enum`/`const`, arbitrary nesting; unsupported keywords are a loud compile error.
  **`constrain.GrammarFromStruct(v)`** derives the schema from a Go struct's json
  tags — "a struct the model cannot violate": constrain → `json.Unmarshal` always
  succeeds. `demo/chat --schema <file.json>` constrains the demo. A property-based
  test asserts every constrained generation validates against its schema; the
  unconstrained path is untouched (purely additive).
- **`chat` package** — chat templating as a library feature (no Jinja engine).
  `chat.Detect(meta)` fingerprints the GGUF/HF `tokenizer.chat_template` string
  (falling back to a vocab-marker heuristic for bare checkpoints) and returns a
  `Template`; `Template.Render(system, turns)` produces the exact prompt string
  and `Template.Stops()` the turn-stop markers. Native Go renderers for **Gemma 3,
  Gemma 4, ChatML (Qwen), Llama-3, and Mistral**, each **byte-exact against HF
  `apply_chat_template`** (golden fixtures in `testdata/chat_goldens`). An
  unrecognized template is an explicit `ErrUnknownTemplate` (caller falls back to
  raw completion). New tokenizer accessors `ChatTemplate()`, `Has()`, `TokenID()`.
- **Sampling completeness** in `SamplingParams` / `Sampler` — the standard
  controls a llama.cpp/Ollama/vLLM user expects:
  - **min-p** (`MinP`) — keep tokens with prob ≥ min-p·max-prob (a relative
    floor), composable with top-k/top-p.
  - **repetition controls** — llama.cpp-style `RepeatPenalty` (scales repeated
    tokens' logits) plus OpenAI-style `PresencePenalty` and `FrequencyPenalty`,
    over a `RepeatLastN` window (the prompt is seeded so it counts). `Generate`
    wires the history automatically.
  - **`LogitBias`** — per-token logit offsets (force or ban specific tokens).
  - **logprobs out** — `Sampler.SampleWithInfo` and `SamplingParams.Logprobs` /
    `TopLogprobs` report the chosen token's log-probability and the top
    alternatives; `Generation.Logprobs` collects them per emitted token.
  - `Seed` (already present) gives reproducible draws.
  `Sample(logits)` and the greedy parity path are unchanged. `demo/chat` gains
  `--min-p`, `--repeat-penalty`, `--presence-penalty`, `--frequency-penalty`,
  `--repeat-last-n`.

### Changed
- Depends on **aikit v1.0.0** (was v0.5.2): its now-frozen Hard tier covers the
  embedding encoder behind `/v1/embeddings`. The `goinfer/gpu` submodule is
  bumped to match, so both modules ship against the same published aikit.
- `cmd/serve` `--model` is now optional when `--embed-model` is given (and vice
  versa); at least one is required. The single-model flag set moved to a config
  struct as the server grew a generative and an embedding half.
- `demo/chat` and `demo/gemma-web` now render prompts via the `chat` package
  (was duplicated per-demo). The Gemma 4 demo render matches HF exactly,
  including the `<|channel>thought` scaffold (so the model may emit reasoning).

### Removed
- `tokenizer.ChatStyle` (enum + method, shipped in v0.2.0) — superseded by
  `chat.Detect`, whose fallback subsumes the old vocab-marker heuristic.

## [v0.2.0] — 2026-06-06

### Added
- **Gemma 4 support** (HF `model_type` `gemma4_unified_text`; GGUF arch
  `gemma4`) — the **E2B** and **E4B** "E-models" plus the **12B dense**, all
  parity-gated against the HF bf16 oracle. Gemma 4 is a meaningfully new
  architecture on top of the Gemma 3 stack, driven entirely through the
  `Architecture` descriptor:
  - per-layer **head_dim** (256 local / 512 global) and per-layer **KV-head
    count**, scale-less **v-norm**, **proportional (partial-rotary) RoPE** on the
    global layers, dual-base RoPE, and the final-logit softcap (30);
  - the E-model additions — **Per-Layer Embeddings (PLE)** branch, **cross-layer
    KV sharing**, variable per-layer **FFN width**, and a per-layer output
    scalar (no AltUp/Laurel — those are Gemma 3n);
  - the 12B's **`attention_k_eq_v`** (V reuses K's projection on the global
    layers).

  Greedy generation is coherent ("Paris"). Gates: `TestGemma4_logitParity`
  (E2B, sample-256 cosine **0.99938** vs HF bf16) and `TestGemma4_12B_logitParity`
  (12B, argmax exact + cosine **0.990**). Out of scope: the 26B-A4B MoE and 31B
  multimodal vision towers (text-only runtime).
- **Gemma 4 chat template** in the tokenizer + `demo/chat` — Gemma 4 replaced
  Gemma 3's `<start_of_turn>`/`<end_of_turn>` with new `<|turn>`/`<turn|>`
  markers. `Tokenizer.ChatStyle` now detects them (`ChatStyleGemma4`), the GGUF
  loader resolves the turn-stop token, and the demo renders the right template
  so replies are clean and stop correctly.

## [v0.1.3] — 2026-06-05

### Added
- **`decoder.LoadGGUFBytes`** / **`tokenizer.LoadGGUFBytes`** — load a GGUF model
  and its tokenizer from an in-memory `[]byte`, touching no filesystem. The
  shared GGUF build core is reused by the path-based `Load` / `LoadGGUF` (both
  unchanged); EOS ids resolve from the GGUF's own metadata.
- **`decoder.SerializeWeights` / `LoadSerializedWeights`** + `Model.Weights()` /
  `Model.Quant()` / `NewModel` — a versioned binary format (`.giw`: magic +
  version + config + quant guard + CRC, lazy fallback like ken's index) for an
  already-quantized weight bundle. On load the big int8/int4 arrays are **aliased
  zero-copy** out of the blob (float scales are copied for alignment), so a
  prequant build skips dequant+requant *and* the resident-weight heap copy.
- **`cmd/prequant`** — build-time generator: turns a GGUF into a `.giw` bundle
  (serialized int8 weights + a metadata-only GGUF carrying the tokenizer).
- **`decoder.GenerateSpeculative`** + **`demo/chat --draft <gguf>`** — greedy
  speculative decoding: a small draft model proposes K tokens, the target
  verifies them in one batched pass (`forwardN`), keeping the matching prefix.
  Output is **token-identical to plain target greedy** (gated by
  `TestSpeculativeGreedyParity`); `cache.TruncateTo` does the KV rollback. On the
  pure-Go CPU int8 kernel it's ~break-even (decode is ~half compute, which
  doesn't amortize at M=K), so it's a forward-looking / bandwidth-bound-backend
  feature — correct and exact, not a CPU speedup.
- **`decoder.SetDecodeParallelThreshold`** / `DefaultDecodeParallelThreshold` —
  goinfer-owned tuning of aikit's matmul parallelism crossover for batch=1
  decode (the demo sets it; aikit's library default stays conservative).

### Changed
- **Faster decode (~48 → ~70 tok/s on the 0.5B, +42%; ~36 on the 1.5B; M-series
  CPU)** via aikit v0.5.x: a per-stream `Workspace` makes steady-state decode
  **zero-alloc** (4660 → 19 allocs/op), q/k/v and gate/up run as **batched**
  matmuls, and a decode-tuned parallelism threshold parallelizes the per-token
  weight matmuls. Numerics bit-identical (`TestDecodeParity`).
- **Batched prompt prefill** — the prompt now runs through one `M=len(prompt)`
  pass instead of sequential single-token forwards, **~1.9–2.9× faster
  time-to-first-token** on the 1.5B (e.g. ~2.7 s → ~1.3 s for an 80-token
  prompt). Seed token bit-identical.
- Requires **aikit ≥ v0.5.2** (the `Workspace` / batched / `Into` W8A8 matmul API
  and the column-blocked W8A8 kernel that reuses weights across M>1 rows).
- `demo/chat` (embed build) loads the baked-in model **from memory by default** —
  no temp file, so the single-file binary runs on a read-only / `FROM scratch`
  filesystem. `--model-tmp` / `GOINFER_MODEL_TMP=1` opts back into a temp-file +
  mmap load (lower peak RAM for large models; tmpfs caveat documented). Load
  progress prints to stderr.
- `demo/chat` gains a **`-tags prequant`** build (now `build-embed.sh`'s default)
  that embeds a `.giw` bundle and maps the int8 weights from the binary image.
  Measured on the 0.5B (Qwen2.5-Coder, M-series CPU): cold start **2.3 s → 0.48 s**
  and resident heap (`phys_footprint`) **772 MB → 78 MB**. The win scales with
  model size — a 4B int8 model no longer needs a ~5 GB weight heap per launch.
  Quant is fixed at bundle-build time; the runtime `--quant` flag and the GGUF
  fallback apply only to the `--model <gguf>` path.
- The embed build now bakes the GGUF **uncompressed** (dropping zstd — q4 weights
  are high-entropy, so it shaved only ~3% while costing inflate time + a
  full-size heap buffer). Removes the `klauspost/compress` dependency; the
  default module graph is back to `aikit` + `x/text`.
- `demo/chat` now ships in **two size tiers** built from the same program:
  Qwen2.5-Coder **0.5B** (~617 MB, ~57 tok/s — the headline) and **1.5B**
  (~1.7 GB, ~26 tok/s — bigger, smarter, still one file). `build-embed.sh --name`
  parameterizes the output basename so the tiers build side by side without
  clobbering. Prequant keeps the 1.5B's resident heap at ~87 MB (≈ the 0.5B), and
  the 1.5B binary stays under GitHub's 2 GiB asset cap.

## [v0.1.2] — 2026-06-04

### Added
- **`demo/chat`** — "an entire LLM in one file": an interactive REPL with a
  zstd-`//go:embed`-ed Qwen2.5-Coder-0.5B-Instruct model baked into a static,
  no-cgo, cross-compiled binary (macOS / Linux / Windows × amd64/arm64).
  Defaults to the fast int8×int8 (W8A8) kernel; ships live prompt-tuning
  slash-commands and canned `/demo` prompts.

### Changed
- Requires `aikit ≥ v0.4.1`, which platform-guards the embed loader's mmap so the
  decoder (and the demo binary) cross-compiles to `windows/amd64`.

## [v0.1.1] — 2026-06-04

### Added
- **`decoder`** — generic decoder-only forward pass with HuggingFace logit
  parity: Gemma 3, Qwen 2.5/3, Llama 2/3, Mistral, Mixtral, GPT-2, Mellum.
  Loads safetensors (single or sharded), GGUF, GPTQ, and AWQ checkpoints;
  f32/bf16/f16 plus int8 (weight-only and W8A8) and group-wise int4
  quantization; KV-cache; the standard samplers; and a pluggable matmul
  `Backend` (default pure-Go SIMD).
- **`tokenizer`** — byte-level BPE and SentencePiece byte-fallback tokenizers
  for the decoder LLMs, from `tokenizer.json` or a bare `.gguf`, id-exact
  against HuggingFace.
- **`constrain`** — constrained / structured decoding via a logit mask that
  forces output to satisfy a grammar; ships a streaming JSON grammar.
- **`gpu`** (opt-in submodule, `-tags gpu`) — WebGPU compute backend
  (Metal / Vulkan / DX12) that registers a resident-weight matmul into both
  the goinfer `decoder` and the aikit `encoder`. The cgo
  `github.com/cogentcore/webgpu` dependency is confined to this submodule;
  the default goinfer build is pure Go, no cgo.
- `demo/gemma` and `demo/gemma-web` — CLI and HTTP examples wiring tokenizer +
  decoder with optional sampling and JSON-constrained output.

### Notes
- goinfer is extracted from [`aikit`](https://github.com/townsendmerino/aikit)'s
  `decoder` / `tokenizer` / `constrain` packages so the LLM runtime can release
  on its own cadence, independent of aikit's stable retrieval core. It depends
  inward on `aikit/embed` + `aikit/linalg` (≥ v0.4.0, which promoted `linalg`
  to public and gave `encoder` a pluggable GPU `Backend`). The split is tracked
  in `aikit/docs/aikit-module-split-plan.md`. Pre-extraction history for these
  packages lives in the aikit repository.
