# Changelog

All notable changes to goinfer are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

The forward-pass and quantization numerics are parity-gated against HuggingFace
and are the stable contract. The loader and architecture-descriptor surface is
pre-1.0 and may change as new model families and quant formats land.

## [Unreleased]

### Added
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
